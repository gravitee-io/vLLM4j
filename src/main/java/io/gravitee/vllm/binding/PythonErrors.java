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
import java.lang.foreign.ValueLayout;

/**
 * Python error handling utilities.
 *
 * <p>Checks for, fetches, and converts Python exceptions into Java
 * {@link VllmException} instances. All methods require the GIL to be held.
 */
public final class PythonErrors {

  private PythonErrors() {}

  /**
   * Checks whether a Python exception is set. If so, fetches and clears it,
   * then throws {@link VllmException} with the string representation.
   *
   * @param context a human-readable description of the operation that may have failed
   * @throws VllmException if a Python exception was pending
   */
  public static void checkPythonError(String context) {
    MemorySegment errOccurred = CPythonBinding.PyErr_Occurred();
    if (PythonTypes.isNull(errOccurred)) return;

    try (Arena tmp = Arena.ofConfined()) {
      MemorySegment pType = tmp.allocate(ValueLayout.ADDRESS);
      MemorySegment pValue = tmp.allocate(ValueLayout.ADDRESS);
      MemorySegment pTraceback = tmp.allocate(ValueLayout.ADDRESS);
      CPythonBinding.PyErr_Fetch(pType, pValue, pTraceback);
      CPythonBinding.PyErr_NormalizeException(pType, pValue, pTraceback);

      MemorySegment valueObj = pValue.get(ValueLayout.ADDRESS, 0);
      MemorySegment typeObj = pType.get(ValueLayout.ADDRESS, 0);

      String detail = "<unknown Python exception>";
      MemorySegment target = PythonTypes.isNull(valueObj) ? typeObj : valueObj;
      if (!PythonTypes.isNull(target)) {
        MemorySegment strObj = CPythonBinding.PyObject_Str(target);
        if (!PythonTypes.isNull(strObj)) {
          detail = PythonTypes.pyUnicodeToString(strObj);
          PythonTypes.decref(strObj);
        }
      }

      // If detail is empty, try to get the exception type name for better diagnostics
      if (detail.isEmpty() || detail.isBlank()) {
        if (!PythonTypes.isNull(typeObj)) {
          MemorySegment typeName = CPythonBinding.PyObject_Str(typeObj);
          if (!PythonTypes.isNull(typeName)) {
            String typeStr = PythonTypes.pyUnicodeToString(typeName);
            PythonTypes.decref(typeName);
            if (!typeStr.isEmpty()) {
              detail = typeStr + " (no detail message)";
            }
          }
        }
      }

      // Also try to get the traceback string for full diagnostics
      MemorySegment tbObj = pTraceback.get(ValueLayout.ADDRESS, 0);
      String tracebackStr = "";
      if (!PythonTypes.isNull(tbObj)) {
        try {
          MemorySegment tbModule = CPythonBinding.PyImport_ImportModule(
            tmp.allocateFrom("traceback")
          );
          if (!PythonTypes.isNull(tbModule)) {
            MemorySegment formatTb = PythonTypes.getAttr(
              tmp,
              tbModule,
              "format_exception"
            );
            if (!PythonTypes.isNull(formatTb)) {
              MemorySegment pyResult = PythonCall.callMethodObjArgs(
                tbModule,
                PythonTypes.pyStr(tmp, "format_exception"),
                typeObj,
                valueObj,
                tbObj
              );
              if (!PythonTypes.isNull(pyResult)) {
                // format_exception returns a list of strings
                MemorySegment joined = PythonTypes.pyStr(tmp, "");
                MemorySegment joinMethod = PythonTypes.pyStr(tmp, "join");
                MemorySegment tbStr = PythonCall.callMethodObjArgs(
                  joined,
                  joinMethod,
                  pyResult
                );
                if (!PythonTypes.isNull(tbStr)) {
                  tracebackStr = PythonTypes.pyUnicodeToString(tbStr);
                  PythonTypes.decref(tbStr);
                }
                PythonTypes.decref(joinMethod);
                PythonTypes.decref(joined);
                PythonTypes.decref(pyResult);
              }
              PythonTypes.decref(formatTb);
            }
            PythonTypes.decref(tbModule);
          }
        } catch (Exception ignored) {
          // traceback extraction is best-effort
        }
        CPythonBinding.PyErr_Clear(); // clear any errors from traceback extraction itself
      }

      PythonTypes.decref(pType.get(ValueLayout.ADDRESS, 0));
      PythonTypes.decref(pValue.get(ValueLayout.ADDRESS, 0));
      PythonTypes.decref(pTraceback.get(ValueLayout.ADDRESS, 0));
      CPythonBinding.PyErr_Clear();

      String message = context + ": " + detail;
      if (!tracebackStr.isEmpty()) {
        message += "\n" + tracebackStr;
      }
      throw new VllmException(message);
    }
  }
}
