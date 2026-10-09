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
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Recursive conversion between JSON-like Java values and Python objects.
 *
 * <p>Java → Python:
 * <ul>
 *   <li>{@code null} → None</li>
 *   <li>{@code String} → str</li>
 *   <li>{@code Integer}, {@code Long} → int</li>
 *   <li>{@code Double}, {@code Float} → float</li>
 *   <li>{@code Boolean} → bool</li>
 *   <li>{@code Map<String, ?>} → dict (insertion order kept)</li>
 *   <li>{@code Collection<?>} → list</li>
 *   <li>anything else → {@code str(value.toString())}</li>
 * </ul>
 *
 * <p>Python → Java: None → {@code null}, str → {@code String}, bool →
 * {@code Boolean}, int → {@code Long}, float → {@code Double}, dict →
 * {@code LinkedHashMap<String, Object>} (keys through {@code str()}),
 * list / tuple → {@code List<Object>}, anything else → its {@code str()}.
 *
 * <p>All methods require the GIL to be held by the calling thread.
 */
public final class PythonObjects {

  /** Builtin types, resolved once; type objects live as long as the interpreter. */
  private static volatile Builtins builtins;

  private record Builtins(
    MemorySegment dict,
    MemorySegment list,
    MemorySegment tuple,
    MemorySegment str,
    MemorySegment bool,
    MemorySegment integer,
    MemorySegment floating
  ) {}

  private PythonObjects() {}

  /**
   * Converts a Java value to a Python object.
   *
   * @param arena arena for native string allocation
   * @param value the Java value
   * @return new reference
   */
  public static MemorySegment toPython(Arena arena, Object value) {
    if (value == null) {
      return PythonTypes.pyNone();
    }
    if (value instanceof String s) {
      return PythonTypes.pyStr(arena, s);
    }
    if (value instanceof Integer i) {
      return CPythonBinding.PyLong_FromLong(i);
    }
    if (value instanceof Long l) {
      return CPythonBinding.PyLong_FromLong(l);
    }
    if (value instanceof Double d) {
      return CPythonBinding.PyFloat_FromDouble(d);
    }
    if (value instanceof Float f) {
      return CPythonBinding.PyFloat_FromDouble(f.doubleValue());
    }
    if (value instanceof Boolean b) {
      return b ? PythonTypes.pyTrue() : PythonTypes.pyFalse();
    }
    if (value instanceof Map<?, ?> m) {
      return toPyDict(arena, m);
    }
    if (value instanceof Collection<?> c) {
      MemorySegment pyList = CPythonBinding.PyList_New(0);
      for (Object item : c) {
        MemorySegment pyItem = toPython(arena, item);
        CPythonBinding.PyList_Append(pyList, pyItem);
        PythonTypes.decref(pyItem);
      }
      return pyList;
    }
    // Fallback: convert to string
    return PythonTypes.pyStr(arena, value.toString());
  }

  /**
   * Converts a Java map to a Python dict, converting values with
   * {@link #toPython}. Keys go through {@code String.valueOf}.
   *
   * @param arena arena for native string allocation
   * @param map   the Java map
   * @return new Python dict reference
   */
  public static MemorySegment toPyDict(Arena arena, Map<?, ?> map) {
    MemorySegment pyDict = CPythonBinding.PyDict_New();
    for (var entry : map.entrySet()) {
      MemorySegment pyValue = toPython(arena, entry.getValue());
      CPythonBinding.PyDict_SetItemString(
        pyDict,
        arena.allocateFrom(String.valueOf(entry.getKey())),
        pyValue
      );
      PythonTypes.decref(pyValue);
    }
    return pyDict;
  }

  /**
   * Converts a Python object to a Java value.
   *
   * @param obj borrowed reference
   * @return the Java value, {@code null} for None or NULL
   */
  public static Object toJava(MemorySegment obj) {
    if (PythonTypes.isNone(obj)) {
      return null;
    }
    Builtins types = builtins();
    // bool before int: bool is a subclass of int.
    if (PythonTypes.isInstance(obj, types.bool())) {
      return CPythonBinding.PyObject_IsTrue(obj) == 1;
    }
    if (PythonTypes.isInstance(obj, types.integer())) {
      return CPythonBinding.PyLong_AsLong(obj);
    }
    if (PythonTypes.isInstance(obj, types.floating())) {
      return CPythonBinding.PyFloat_AsDouble(obj);
    }
    if (PythonTypes.isInstance(obj, types.str())) {
      return PythonTypes.pyUnicodeToString(obj);
    }
    if (PythonTypes.isInstance(obj, types.dict())) {
      return dictToJava(obj);
    }
    if (
      PythonTypes.isInstance(obj, types.list()) ||
      PythonTypes.isInstance(obj, types.tuple())
    ) {
      return iterableToJava(obj);
    }
    return strOf(obj);
  }

  private static Map<String, Object> dictToJava(MemorySegment dict) {
    Map<String, Object> map = new LinkedHashMap<>();
    try (Arena tmp = Arena.ofConfined()) {
      MemorySegment name = PythonTypes.pyStr(tmp, "items");
      MemorySegment items = PythonCall.callMethodObjArgs(dict, name);
      PythonTypes.decref(name);
      PythonErrors.checkPythonError("dict.items()");
      MemorySegment iter = CPythonBinding.PyObject_GetIter(items);
      PythonErrors.checkPythonError("iter(dict.items())");
      MemorySegment pair;
      while (!PythonTypes.isNull(pair = CPythonBinding.PyIter_Next(iter))) {
        MemorySegment key = CPythonBinding.PyTuple_GetItem(pair, 0); // borrowed
        MemorySegment value = CPythonBinding.PyTuple_GetItem(pair, 1); // borrowed
        map.put(strOf(key), toJava(value));
        PythonTypes.decref(pair);
      }
      PythonErrors.checkPythonError("iterating dict.items()");
      PythonTypes.decref(iter);
      PythonTypes.decref(items);
    }
    return map;
  }

  private static List<Object> iterableToJava(MemorySegment seq) {
    List<Object> list = new ArrayList<>();
    MemorySegment iter = CPythonBinding.PyObject_GetIter(seq);
    PythonErrors.checkPythonError("iter(sequence)");
    MemorySegment item;
    while (!PythonTypes.isNull(item = CPythonBinding.PyIter_Next(iter))) {
      list.add(toJava(item));
      PythonTypes.decref(item);
    }
    PythonErrors.checkPythonError("iterating sequence");
    PythonTypes.decref(iter);
    return list;
  }

  private static String strOf(MemorySegment obj) {
    MemorySegment str = CPythonBinding.PyObject_Str(obj);
    PythonErrors.checkPythonError("str(obj)");
    String s = PythonTypes.pyUnicodeToString(str);
    PythonTypes.decref(str);
    return s;
  }

  private static Builtins builtins() {
    if (builtins == null) {
      synchronized (PythonObjects.class) {
        if (builtins == null) {
          try (Arena tmp = Arena.ofConfined()) {
            builtins = new Builtins(
              PythonCall.importClass(tmp, "builtins", "dict"),
              PythonCall.importClass(tmp, "builtins", "list"),
              PythonCall.importClass(tmp, "builtins", "tuple"),
              PythonCall.importClass(tmp, "builtins", "str"),
              PythonCall.importClass(tmp, "builtins", "bool"),
              PythonCall.importClass(tmp, "builtins", "int"),
              PythonCall.importClass(tmp, "builtins", "float")
            );
          }
        }
      }
    }
    return builtins;
  }
}
