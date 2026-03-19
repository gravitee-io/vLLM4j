/*
 * Copyright © 2015 The Gravitee team (http://gravitee.io)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package io.gravitee.vllm.binding;

import io.gravitee.vllm.platform.PlatformResolver;
import java.lang.foreign.Arena;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;

/**
 * Reflection-based dispatch layer for the jextract-generated CPython FFM bindings.
 *
 * <p>At build time, jextract generates a {@code CPython} class under a platform-specific
 * package (e.g. {@code io.gravitee.vllm.macosx.aarch64.CPython} or
 * {@code io.gravitee.vllm.linux.x86_64.CPython}). The generated class contains static
 * methods that correspond to CPython C API functions, but the struct layouts and calling
 * conventions differ between platforms.
 *
 * <p>This class uses {@link java.lang.reflect.Method#invoke reflection} to dispatch
 * calls to the correct platform-specific class at runtime, so that consumer code
 * (binding helpers, engine, template renderer) is platform-agnostic.
 *
 * <h2>Architecture</h2>
 * <pre>{@code
 * Consumer Code (PythonCall, PythonTypes, VllmEngine, etc.)
 *         │
 *         ▼
 *     CPythonBinding (static methods wrapping reflection calls)
 *         │
 *         ▼
 *     invoke("CPython", methodName, parameterTypes, args)
 *         │
 *         ▼
 *     Class.forName("io.gravitee.vllm.<os>.<arch>.CPython")
 *         .getMethod(methodName, parameterTypes)
 *         .invoke(null, args)
 *         │
 *         ▼
 *     jextract-generated CPython class → FFM downcall → libpython
 * }</pre>
 *
 * <h2>Variadic Functions</h2>
 * <p>Some CPython functions are variadic (e.g. {@code PyObject_CallMethodObjArgs}).
 * jextract generates inner classes with {@code makeInvoker(MemoryLayout...)} factories.
 * These are accessed via {@link #makeVariadicInvoker} and wrapped in
 * {@link VariadicInvoker}.
 *
 * @see PlatformResolver
 */
public final class CPythonBinding {

  private static final String pkg = PlatformResolver.platform().getPackage();
  private static final String runtime = PlatformResolver.platform().runtime();
  private static final String basePackage = "io.gravitee.vllm.%s.".formatted(
    pkg
  );

  /** The C_POINTER layout constant from the generated CPython$shared class. */
  private static volatile MemoryLayout cachedCPointer;

  private CPythonBinding() {}

  // ── Core reflection dispatch ────────────────────────────────────────────

  /**
   * Invokes a static method on a jextract-generated class via reflection.
   *
   * <p>Resolves the full class name from
   * {@code io.gravitee.vllm.<os>.<arch>.<classNameSuffix>} and calls
   * the given static method.
   *
   * @param classNameSuffix the simple class name (e.g. {@code "CPython"})
   * @param methodName      the static method name (e.g. {@code "Py_InitializeEx"})
   * @param parameterTypes  the method parameter types
   * @param args            the arguments
   * @param <T>             the return type
   * @return the method's return value
   * @throws IllegalStateException if the class/method cannot be found or invocation fails
   */
  @SuppressWarnings("unchecked")
  public static <T> T invoke(
    String classNameSuffix,
    String methodName,
    Class<?>[] parameterTypes,
    Object... args
  ) {
    try {
      String fullClassName = basePackage + classNameSuffix;
      Class<?> targetClass = Class.forName(fullClassName);
      Method method = targetClass.getMethod(methodName, parameterTypes);
      return (T) method.invoke(null, args);
    } catch (ClassNotFoundException e) {
      throw new IllegalStateException(
        "Class not found for runtime " + runtime + ": " + e.getMessage(),
        e
      );
    } catch (NoSuchMethodException e) {
      throw new IllegalStateException(
        "Method not found for runtime " + runtime + ": " + e.getMessage(),
        e
      );
    } catch (InvocationTargetException e) {
      if (e.getTargetException() instanceof RuntimeException re) {
        throw re;
      }
      throw new IllegalStateException(
        "Error invoking method for runtime " + runtime + ": " + e.getMessage(),
        e
      );
    } catch (IllegalAccessException e) {
      throw new IllegalStateException(
        "Illegal access to method for runtime " +
          runtime +
          ": " +
          e.getMessage(),
        e
      );
    }
  }

  /**
   * Shorthand for invoking a method on the main {@code CPython} class.
   *
   * @see #invoke(String, String, Class[], Object...)
   */
  @SuppressWarnings("unchecked")
  public static <T> T cpython(
    String methodName,
    Class<?>[] parameterTypes,
    Object... args
  ) {
    return invoke("CPython", methodName, parameterTypes, args);
  }

  // ── C_POINTER constant ──────────────────────────────────────────────────

  /**
   * Returns the platform-specific {@code C_POINTER} layout constant from the
   * jextract-generated class hierarchy.
   *
   * <p>This is the canonical {@code void*} layout for the target platform,
   * needed when constructing variadic call descriptors.
   *
   * @return the {@code AddressLayout} for a C pointer
   */
  public static MemoryLayout C_POINTER() {
    if (cachedCPointer == null) {
      synchronized (CPythonBinding.class) {
        if (cachedCPointer == null) {
          try {
            String fullClassName = basePackage + "CPython";
            Class<?> targetClass = Class.forName(fullClassName);
            cachedCPointer = (MemoryLayout) targetClass
              .getField("C_POINTER")
              .get(null);
          } catch (Exception e) {
            throw new IllegalStateException(
              "Cannot resolve C_POINTER for runtime " + runtime,
              e
            );
          }
        }
      }
    }
    return cachedCPointer;
  }

  // ── Variadic function support ───────────────────────────────────────────

  /**
   * Creates a variadic invoker for a jextract-generated inner class.
   *
   * <p>jextract generates inner classes for variadic C functions (e.g.
   * {@code CPython.PyObject_CallMethodObjArgs}). Each inner class has a
   * static {@code makeInvoker(MemoryLayout...)} factory that returns an
   * instance with an {@code apply(...)} method.
   *
   * <p>This method:
   * <ol>
   *   <li>Resolves the inner class via reflection</li>
   *   <li>Calls {@code makeInvoker(layouts)}</li>
   *   <li>Wraps the result in a {@link VariadicInvoker}</li>
   * </ol>
   *
   * @param innerClassName the simple name of the inner class (e.g. {@code "PyObject_CallMethodObjArgs"})
   * @param layouts        the variadic argument layouts
   * @return a wrapped invoker
   */
  public static VariadicInvoker makeVariadicInvoker(
    String innerClassName,
    MemoryLayout... layouts
  ) {
    try {
      String fullClassName = basePackage + "CPython";
      Class<?> outerClass = Class.forName(fullClassName);

      // Find the inner class
      Class<?> innerClass = null;
      for (Class<?> inner : outerClass.getDeclaredClasses()) {
        if (innerClassName.equals(inner.getSimpleName())) {
          innerClass = inner;
          break;
        }
      }
      if (innerClass == null) {
        throw new IllegalStateException(
          "Inner class " + innerClassName + " not found in " + fullClassName
        );
      }

      // Call makeInvoker(MemoryLayout...)
      Method makeInvokerMethod = innerClass.getMethod(
        "makeInvoker",
        MemoryLayout[].class
      );
      Object invokerInstance = makeInvokerMethod.invoke(null, (Object) layouts);

      return new VariadicInvoker(invokerInstance, innerClass);
    } catch (IllegalStateException e) {
      throw e;
    } catch (Exception e) {
      throw new IllegalStateException(
        "Failed to create variadic invoker for " +
          innerClassName +
          " on runtime " +
          runtime,
        e
      );
    }
  }

  /**
   * Wrapper around a jextract-generated variadic invoker instance.
   *
   * <p>Delegates to the {@code apply()} method on the invoker via reflection.
   * The invoker instance is created by {@link #makeVariadicInvoker}.
   */
  public static final class VariadicInvoker {

    private final Object invokerInstance;
    private final Method applyMethod;

    VariadicInvoker(Object invokerInstance, Class<?> invokerClass) {
      this.invokerInstance = invokerInstance;
      // Find the apply method — it takes (MemorySegment, MemorySegment, Object...)
      // for PyObject_CallMethodObjArgs
      Method found = null;
      for (Method m : invokerClass.getMethods()) {
        if ("apply".equals(m.getName())) {
          found = m;
          break;
        }
      }
      if (found == null) {
        throw new IllegalStateException(
          "No apply() method found on " + invokerClass.getName()
        );
      }
      this.applyMethod = found;
    }

    /**
     * Calls the variadic function.
     *
     * @param obj     first fixed argument (PyObject* self)
     * @param name    second fixed argument (PyObject* method name)
     * @param varArgs the variadic arguments (must match the layouts passed to makeInvoker)
     * @return the return value
     */
    public MemorySegment apply(
      MemorySegment obj,
      MemorySegment name,
      Object... varArgs
    ) {
      try {
        return (MemorySegment) applyMethod.invoke(
          invokerInstance,
          obj,
          name,
          varArgs
        );
      } catch (InvocationTargetException e) {
        if (e.getTargetException() instanceof RuntimeException re) {
          throw re;
        }
        throw new IllegalStateException(
          "Variadic apply() failed for runtime " + runtime,
          e
        );
      } catch (IllegalAccessException e) {
        throw new IllegalStateException(
          "Cannot access variadic apply() for runtime " + runtime,
          e
        );
      }
    }
  }

  // ── Convenience methods for all CPython API functions used ──────────────
  //
  // These provide type-safe, compile-time-checked wrappers around the
  // generic invoke() method. They also serve as documentation of which
  // CPython functions the project uses.

  // -- Interpreter lifecycle ------------------------------------------------

  public static void Py_InitializeEx(int initsigs) {
    cpython("Py_InitializeEx", new Class<?>[] { int.class }, initsigs);
  }

  public static MemorySegment PyEval_SaveThread() {
    return cpython("PyEval_SaveThread", new Class<?>[] {});
  }

  // -- GIL management -------------------------------------------------------

  public static int PyGILState_Ensure() {
    return cpython("PyGILState_Ensure", new Class<?>[] {});
  }

  public static void PyGILState_Release(int state) {
    cpython("PyGILState_Release", new Class<?>[] { int.class }, state);
  }

  // -- Error handling -------------------------------------------------------

  public static MemorySegment PyErr_Occurred() {
    return cpython("PyErr_Occurred", new Class<?>[] {});
  }

  public static void PyErr_Fetch(
    MemorySegment pType,
    MemorySegment pValue,
    MemorySegment pTraceback
  ) {
    cpython(
      "PyErr_Fetch",
      new Class<?>[] {
        MemorySegment.class,
        MemorySegment.class,
        MemorySegment.class,
      },
      pType,
      pValue,
      pTraceback
    );
  }

  public static void PyErr_NormalizeException(
    MemorySegment pType,
    MemorySegment pValue,
    MemorySegment pTraceback
  ) {
    cpython(
      "PyErr_NormalizeException",
      new Class<?>[] {
        MemorySegment.class,
        MemorySegment.class,
        MemorySegment.class,
      },
      pType,
      pValue,
      pTraceback
    );
  }

  public static void PyErr_Clear() {
    cpython("PyErr_Clear", new Class<?>[] {});
  }

  // -- Import ---------------------------------------------------------------

  public static MemorySegment PyImport_ImportModule(MemorySegment name) {
    return cpython(
      "PyImport_ImportModule",
      new Class<?>[] { MemorySegment.class },
      name
    );
  }

  // -- Object protocol ------------------------------------------------------

  public static MemorySegment PyObject_GetAttrString(
    MemorySegment obj,
    MemorySegment name
  ) {
    return cpython(
      "PyObject_GetAttrString",
      new Class<?>[] { MemorySegment.class, MemorySegment.class },
      obj,
      name
    );
  }

  public static int PyObject_SetAttrString(
    MemorySegment obj,
    MemorySegment name,
    MemorySegment value
  ) {
    return cpython(
      "PyObject_SetAttrString",
      new Class<?>[] {
        MemorySegment.class,
        MemorySegment.class,
        MemorySegment.class,
      },
      obj,
      name,
      value
    );
  }

  public static int PyObject_IsTrue(MemorySegment obj) {
    return cpython(
      "PyObject_IsTrue",
      new Class<?>[] { MemorySegment.class },
      obj
    );
  }

  public static MemorySegment PyObject_Str(MemorySegment obj) {
    return cpython("PyObject_Str", new Class<?>[] { MemorySegment.class }, obj);
  }

  public static MemorySegment PyObject_GetIter(MemorySegment obj) {
    return cpython(
      "PyObject_GetIter",
      new Class<?>[] { MemorySegment.class },
      obj
    );
  }

  // -- Reference counting ---------------------------------------------------

  public static void Py_IncRef(MemorySegment obj) {
    cpython("Py_IncRef", new Class<?>[] { MemorySegment.class }, obj);
  }

  public static void Py_DecRef(MemorySegment obj) {
    cpython("Py_DecRef", new Class<?>[] { MemorySegment.class }, obj);
  }

  public static int Py_IsNone(MemorySegment obj) {
    return cpython("Py_IsNone", new Class<?>[] { MemorySegment.class }, obj);
  }

  // -- Unicode --------------------------------------------------------------

  public static MemorySegment PyUnicode_FromString(MemorySegment str) {
    return cpython(
      "PyUnicode_FromString",
      new Class<?>[] { MemorySegment.class },
      str
    );
  }

  public static MemorySegment PyUnicode_AsUTF8(MemorySegment obj) {
    return cpython(
      "PyUnicode_AsUTF8",
      new Class<?>[] { MemorySegment.class },
      obj
    );
  }

  // -- Tuple ----------------------------------------------------------------

  public static MemorySegment PyTuple_New(long len) {
    return cpython("PyTuple_New", new Class<?>[] { long.class }, len);
  }

  public static void PyTuple_SetItem(
    MemorySegment tuple,
    long pos,
    MemorySegment item
  ) {
    cpython(
      "PyTuple_SetItem",
      new Class<?>[] { MemorySegment.class, long.class, MemorySegment.class },
      tuple,
      pos,
      item
    );
  }

  public static MemorySegment PyTuple_GetItem(MemorySegment tuple, long pos) {
    return cpython(
      "PyTuple_GetItem",
      new Class<?>[] { MemorySegment.class, long.class },
      tuple,
      pos
    );
  }

  // -- List -----------------------------------------------------------------

  public static MemorySegment PyList_New(long len) {
    return cpython("PyList_New", new Class<?>[] { long.class }, len);
  }

  public static long PyList_Size(MemorySegment list) {
    return cpython("PyList_Size", new Class<?>[] { MemorySegment.class }, list);
  }

  public static MemorySegment PyList_GetItem(MemorySegment list, long index) {
    return cpython(
      "PyList_GetItem",
      new Class<?>[] { MemorySegment.class, long.class },
      list,
      index
    );
  }

  public static int PyList_Append(MemorySegment list, MemorySegment item) {
    return cpython(
      "PyList_Append",
      new Class<?>[] { MemorySegment.class, MemorySegment.class },
      list,
      item
    );
  }

  // -- Dict -----------------------------------------------------------------

  public static MemorySegment PyDict_New() {
    return cpython("PyDict_New", new Class<?>[] {});
  }

  public static int PyDict_SetItemString(
    MemorySegment dict,
    MemorySegment key,
    MemorySegment val
  ) {
    return cpython(
      "PyDict_SetItemString",
      new Class<?>[] {
        MemorySegment.class,
        MemorySegment.class,
        MemorySegment.class,
      },
      dict,
      key,
      val
    );
  }

  // -- Numeric --------------------------------------------------------------

  public static MemorySegment PyLong_FromLong(long val) {
    return cpython("PyLong_FromLong", new Class<?>[] { long.class }, val);
  }

  public static long PyLong_AsLong(MemorySegment obj) {
    return cpython(
      "PyLong_AsLong",
      new Class<?>[] { MemorySegment.class },
      obj
    );
  }

  public static MemorySegment PyFloat_FromDouble(double val) {
    return cpython("PyFloat_FromDouble", new Class<?>[] { double.class }, val);
  }

  public static double PyFloat_AsDouble(MemorySegment obj) {
    return cpython(
      "PyFloat_AsDouble",
      new Class<?>[] { MemorySegment.class },
      obj
    );
  }

  public static MemorySegment PyBool_FromLong(long val) {
    return cpython("PyBool_FromLong", new Class<?>[] { long.class }, val);
  }

  // -- Sequence / Iterator --------------------------------------------------

  public static MemorySegment PySequence_GetItem(
    MemorySegment seq,
    long index
  ) {
    return cpython(
      "PySequence_GetItem",
      new Class<?>[] { MemorySegment.class, long.class },
      seq,
      index
    );
  }

  public static MemorySegment PyIter_Next(MemorySegment iter) {
    return cpython("PyIter_Next", new Class<?>[] { MemorySegment.class }, iter);
  }
}
