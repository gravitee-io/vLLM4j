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

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;

/**
 * Low-level CPython call helpers.
 *
 * <p>Wraps the variadic invokers and function-call patterns needed to drive
 * Python objects through the C API. All methods are static and require the
 * GIL to be held by the calling thread.
 */
public final class PythonCall {

  /** Lazily resolved MethodHandle for {@code PyObject_Call}. */
  private static volatile MethodHandle pyObjectCallHandle;

  private PythonCall() {}

  /**
   * Calls {@code obj.method(*objArgs)} using {@code PyObject_CallMethodObjArgs}.
   *
   * <p>The generated binding for {@code PyObject_CallMethodObjArgs} is a
   * variadic invoker class. We use {@code makeInvoker} with one
   * {@code C_POINTER} layout per extra argument, then append a {@code NULL}
   * sentinel as required by the C varargs contract.
   *
   * @param obj     the Python object to call the method on
   * @param name    a Python unicode string naming the method (new reference — caller owns)
   * @param objArgs zero or more positional arguments (borrowed references)
   * @return new reference to the return value, or NULL on error
   */
  public static MemorySegment callMethodObjArgs(
    MemorySegment obj,
    MemorySegment name,
    MemorySegment... objArgs
  ) {
    var layouts = new java.lang.foreign.MemoryLayout[objArgs.length + 1];
    for (int i = 0; i < objArgs.length; i++) layouts[i] =
      CPythonBinding.C_POINTER();
    layouts[objArgs.length] = CPythonBinding.C_POINTER(); // NULL sentinel

    var invoker = CPythonBinding.makeVariadicInvoker(
      "PyObject_CallMethodObjArgs",
      layouts
    );

    Object[] varArgs = new Object[objArgs.length + 1];
    for (int i = 0; i < objArgs.length; i++) varArgs[i] = objArgs[i];
    varArgs[objArgs.length] = MemorySegment.NULL;

    return invoker.apply(obj, name, varArgs);
  }

  /**
   * Calls {@code callable(**kwargs)} using an empty positional-args tuple.
   *
   * @return new reference to the return value
   */
  public static MemorySegment callWithKwargs(
    MemorySegment callable,
    MemorySegment kwargs
  ) {
    MemorySegment emptyTuple = makeTuple();
    MemorySegment result = pyObjectCall(callable, emptyTuple, kwargs);
    PythonTypes.decref(emptyTuple);
    return result;
  }

  /**
   * Calls {@code callable(arg)} with a single positional argument.
   *
   * @return new reference to the return value
   */
  public static MemorySegment callOneArg(
    MemorySegment callable,
    MemorySegment arg
  ) {
    MemorySegment args = makeTuple(arg);
    MemorySegment result = pyObjectCall(callable, args, MemorySegment.NULL);
    PythonTypes.decref(args);
    return result;
  }

  /**
   * Creates a Python tuple from zero or more elements.
   *
   * <p>Note: {@code PyTuple_SetItem} steals a reference, so we incref each
   * element so the caller's reference survives.
   */
  public static MemorySegment makeTuple(MemorySegment... elements) {
    MemorySegment tuple = CPythonBinding.PyTuple_New(elements.length);
    for (int i = 0; i < elements.length; i++) {
      PythonTypes.incref(elements[i]); // PyTuple_SetItem steals
      CPythonBinding.PyTuple_SetItem(tuple, i, elements[i]);
    }
    return tuple;
  }

  /**
   * Calls {@code PyObject_Call(callable, args, kwargs)}.
   *
   * <p>{@code PyObject_Call} is not included in the filtered jextract output.
   * We resolve it lazily via {@link SymbolLookup#loaderLookup()} and cache
   * the {@link MethodHandle}.
   */
  public static MemorySegment pyObjectCall(
    MemorySegment callable,
    MemorySegment args,
    MemorySegment kwargs
  ) {
    if (pyObjectCallHandle == null) {
      synchronized (PythonCall.class) {
        if (pyObjectCallHandle == null) {
          var addr = SymbolLookup.loaderLookup()
            .or(Linker.nativeLinker().defaultLookup())
            .find("PyObject_Call")
            .orElseThrow(() ->
              new VllmException("Cannot locate PyObject_Call in libpython")
            );
          var desc = FunctionDescriptor.of(
            ValueLayout.ADDRESS, // PyObject* return
            ValueLayout.ADDRESS, // PyObject* callable
            ValueLayout.ADDRESS, // PyObject* args
            ValueLayout.ADDRESS // PyObject* kwargs (may be NULL)
          );
          pyObjectCallHandle = Linker.nativeLinker().downcallHandle(addr, desc);
        }
      }
    }
    try {
      MemorySegment kw = (kwargs == null) ? MemorySegment.NULL : kwargs;
      return (MemorySegment) pyObjectCallHandle.invokeExact(callable, args, kw);
    } catch (Error | RuntimeException ex) {
      throw ex;
    } catch (Throwable t) {
      throw new VllmException("PyObject_Call failed", t);
    }
  }

  /**
   * Imports {@code moduleName} and returns {@code getattr(module, className)}.
   *
   * @param arena      arena for native string allocation
   * @param moduleName Python module path, e.g. {@code "vllm.engine.arg_utils"}
   * @param className  attribute name to retrieve from the module
   * @return new reference to the class object
   */
  public static MemorySegment importClass(
    Arena arena,
    String moduleName,
    String className
  ) {
    MemorySegment module = CPythonBinding.PyImport_ImportModule(
      arena.allocateFrom(moduleName)
    );
    PythonErrors.checkPythonError("import " + moduleName);
    MemorySegment cls = PythonTypes.getAttr(arena, module, className);
    PythonErrors.checkPythonError(
      "getattr(" + moduleName + ", " + className + ")"
    );
    PythonTypes.decref(module);
    return cls;
  }
}
