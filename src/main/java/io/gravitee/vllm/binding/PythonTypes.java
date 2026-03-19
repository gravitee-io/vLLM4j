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
 * Python type conversion and object utilities.
 *
 * <p>Provides static helpers for converting between Java types and Python
 * objects via the CPython C API, plus reference-counting and attribute access.
 * All methods require the GIL to be held by the calling thread.
 */
public final class PythonTypes {

  private PythonTypes() {}

  // ── Reference counting ─────────────────────────────────────────────────

  /**
   * Decrements the Python reference count of {@code obj}.
   * No-op for null/NULL.
   */
  public static void decref(MemorySegment obj) {
    if (obj == null || obj.address() == 0) return;
    CPythonBinding.Py_DecRef(obj);
  }

  /**
   * Increments the Python reference count of {@code obj}.
   * No-op for null/NULL.
   */
  public static void incref(MemorySegment obj) {
    if (obj == null || obj.address() == 0) return;
    CPythonBinding.Py_IncRef(obj);
  }

  // ── Singleton accessors ────────────────────────────────────────────────

  /**
   * Returns the Python singleton {@code None} as a new reference.
   *
   * <p>{@code _Py_NoneStruct} depends on internal CPython types excluded by
   * jextract. We obtain {@code None} via {@code builtins.None} instead.
   */
  public static MemorySegment pyNone() {
    return builtinSingleton("None");
  }

  /** Returns the Python singleton {@code True} as a new reference. */
  public static MemorySegment pyTrue() {
    return builtinSingleton("True");
  }

  /** Returns the Python singleton {@code False} as a new reference. */
  public static MemorySegment pyFalse() {
    return builtinSingleton("False");
  }

  // ── String conversion ──────────────────────────────────────────────────

  /**
   * Allocates a Python unicode object from a Java string.
   *
   * @param arena arena for the native C string
   * @param s     the Java string
   * @return new Python unicode reference
   */
  public static MemorySegment pyStr(Arena arena, String s) {
    return CPythonBinding.PyUnicode_FromString(arena.allocateFrom(s));
  }

  /**
   * Converts a Python unicode object to a Java {@code String}.
   *
   * @return the Java string, or empty string if the input is null/NULL
   */
  public static String pyUnicodeToString(MemorySegment pyStr) {
    if (isNull(pyStr)) return "";
    MemorySegment cStr = CPythonBinding.PyUnicode_AsUTF8(pyStr);
    if (isNull(cStr)) return "";
    return cStr.reinterpret(Long.MAX_VALUE).getString(0);
  }

  // ── Attribute access ───────────────────────────────────────────────────

  /** Gets a Python attribute as a new reference. */
  public static MemorySegment getAttr(
    Arena arena,
    MemorySegment obj,
    String name
  ) {
    return CPythonBinding.PyObject_GetAttrString(obj, arena.allocateFrom(name));
  }

  /** Reads a string attribute. */
  public static String getStrAttr(Arena arena, MemorySegment obj, String name) {
    MemorySegment attr = getAttr(arena, obj, name);
    String result = pyUnicodeToString(attr);
    decref(attr);
    return result;
  }

  /** Reads a boolean attribute. */
  public static boolean getBoolAttr(
    Arena arena,
    MemorySegment obj,
    String name
  ) {
    MemorySegment attr = getAttr(arena, obj, name);
    boolean result = CPythonBinding.PyObject_IsTrue(attr) != 0;
    decref(attr);
    return result;
  }

  /** Reads a long attribute. */
  public static long getLongAttr(Arena arena, MemorySegment obj, String name) {
    MemorySegment attr = getAttr(arena, obj, name);
    long result = CPythonBinding.PyLong_AsLong(attr);
    decref(attr);
    return result;
  }

  /** Reads a double attribute. */
  public static double getDoubleAttr(
    Arena arena,
    MemorySegment obj,
    String name
  ) {
    MemorySegment attr = getAttr(arena, obj, name);
    double result = CPythonBinding.PyFloat_AsDouble(attr);
    decref(attr);
    return result;
  }

  // ── Dict helpers ───────────────────────────────────────────────────────

  /** Puts a float value into a Python dict. */
  public static void putDictFloat(
    Arena arena,
    MemorySegment dict,
    String key,
    double value
  ) {
    MemorySegment pyVal = CPythonBinding.PyFloat_FromDouble(value);
    CPythonBinding.PyDict_SetItemString(dict, arena.allocateFrom(key), pyVal);
    decref(pyVal);
  }

  /** Puts a long value into a Python dict. */
  public static void putDictInt(
    Arena arena,
    MemorySegment dict,
    String key,
    long value
  ) {
    MemorySegment pyVal = CPythonBinding.PyLong_FromLong(value);
    CPythonBinding.PyDict_SetItemString(dict, arena.allocateFrom(key), pyVal);
    decref(pyVal);
  }

  /** Puts an existing Python object into a dict (does NOT steal the reference). */
  public static void putDictObj(
    Arena arena,
    MemorySegment dict,
    String key,
    MemorySegment obj
  ) {
    CPythonBinding.PyDict_SetItemString(dict, arena.allocateFrom(key), obj);
  }

  // ── Null / None checks ─────────────────────────────────────────────────

  /** Returns {@code true} if {@code obj} is a null pointer. */
  public static boolean isNull(MemorySegment obj) {
    return obj == null || obj.address() == 0;
  }

  /** Returns {@code true} if {@code obj} is Python's {@code None}. */
  public static boolean isNone(MemorySegment obj) {
    if (isNull(obj)) return true;
    return CPythonBinding.Py_IsNone(obj) != 0;
  }

  // ── Bytes conversion ──────────────────────────────────────────────────

  /** Lazily resolved MethodHandle for {@code PyBytes_FromStringAndSize}. */
  private static volatile MethodHandle pyBytesFromStringAndSizeHandle;

  /**
   * Creates a Python {@code bytes} object from a Java byte array.
   *
   * <p>{@code PyBytes_FromStringAndSize} is not included in the filtered
   * jextract output. We resolve it lazily via {@link SymbolLookup} and cache
   * the {@link MethodHandle}.
   *
   * @param arena arena for native memory allocation
   * @param data  the Java byte array
   * @return new Python bytes reference
   */
  public static MemorySegment pyBytes(Arena arena, byte[] data) {
    if (pyBytesFromStringAndSizeHandle == null) {
      synchronized (PythonTypes.class) {
        if (pyBytesFromStringAndSizeHandle == null) {
          var addr = SymbolLookup.loaderLookup()
            .or(Linker.nativeLinker().defaultLookup())
            .find("PyBytes_FromStringAndSize")
            .orElseThrow(() ->
              new VllmException(
                "Cannot locate PyBytes_FromStringAndSize in libpython"
              )
            );
          var desc = FunctionDescriptor.of(
            ValueLayout.ADDRESS, // PyObject* return
            ValueLayout.ADDRESS, // const char* v
            ValueLayout.JAVA_LONG // Py_ssize_t len
          );
          pyBytesFromStringAndSizeHandle = Linker.nativeLinker().downcallHandle(
            addr,
            desc
          );
        }
      }
    }
    try {
      MemorySegment nativeData = arena.allocate(data.length);
      nativeData.copyFrom(MemorySegment.ofArray(data));
      return (MemorySegment) pyBytesFromStringAndSizeHandle.invokeExact(
        nativeData,
        (long) data.length
      );
    } catch (Error | RuntimeException ex) {
      throw ex;
    } catch (Throwable t) {
      throw new VllmException("PyBytes_FromStringAndSize failed", t);
    }
  }

  // ── Internal ───────────────────────────────────────────────────────────

  /** Obtains a Python builtin singleton (None, True, False) as a new reference. */
  private static MemorySegment builtinSingleton(String name) {
    try (Arena tmp = Arena.ofConfined()) {
      MemorySegment builtins = CPythonBinding.PyImport_ImportModule(
        tmp.allocateFrom("builtins")
      );
      MemorySegment result = CPythonBinding.PyObject_GetAttrString(
        builtins,
        tmp.allocateFrom(name)
      );
      CPythonBinding.Py_DecRef(builtins);
      return result;
    }
  }
}
