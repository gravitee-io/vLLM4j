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
package io.gravitee.vllm.decision;

import io.gravitee.vllm.binding.CPythonBinding;
import io.gravitee.vllm.binding.PythonCall;
import io.gravitee.vllm.binding.PythonErrors;
import io.gravitee.vllm.binding.PythonObjects;
import io.gravitee.vllm.binding.PythonTypes;
import io.gravitee.vllm.binding.VllmException;
import io.gravitee.vllm.runtime.GIL;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.Map;

/**
 * Runs a vLLM Semantic Router decision model in-process and answers typed
 * questions about a state in one forward pass, without generating tokens.
 *
 * <p>Decision models (Decision 1.0 and 2.0, {@code vllm-sr/Decision-1.0-*} and
 * {@code vllm-sr/Decision-2.0-*}, and Decision 3.0, {@code vllm-sr/d3*}) are not served by vLLM's
 * {@code LLMEngine}: vLLM Semantic Router runs them on its own model runtime,
 * {@code vllm_srun}. This class drives that runtime in the embedded CPython,
 * the same way {@link io.gravitee.vllm.engine.VllmEngine} drives
 * {@code LLMEngine}: no HTTP server, no extra process. A request goes through
 * the runtime's {@code decisions} surface, so the answers are the ones its
 * {@code POST /v1/decisions} would return.
 *
 * <h2>Usage</h2>
 * <pre>{@code
 * try (var decider = VllmDecider.builder()
 *         .model("vllm-sr/Decision-2.0-Kai-0.6B")
 *         .build()) {
 *     Map<String, String> intents = new LinkedHashMap<>();   // options keep their order
 *     intents.put("billing", "Payment or invoice issue");
 *     intents.put("account", "Login or access issue");
 *     DecisionResponse response = decider.decide(
 *         DecisionRequest.builder(Map.of("request", "I was charged twice this month."))
 *             .choice("intent", "What does the user want?", intents)
 *             .build());
 *     var intent = (Answer.Choice) response.answer("intent");
 * }
 * }</pre>
 *
 * <h2>Lifecycle</h2>
 * {@link VllmDeciderBuilder#build()} loads the model in the foreground: it
 * resolves the pinned revision, verifies the files, loads the weights and
 * checks the model against its golden answers. It returns once the model is
 * ready, or throws.
 *
 * <h2>Thread safety</h2>
 * Thread-safe. The GIL is released while a request waits for the model, so
 * concurrent {@link #decide} calls are batched by the runtime's scheduler.
 *
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public final class VllmDecider implements AutoCloseable {

  /**
   * Python side: the runtime, and an event loop on a daemon thread to run its
   * coroutines. {@code exit_on_device_error} is forced off — it would
   * {@code os._exit} the JVM on a device failure; the model reports
   * {@code degraded} instead.
   *
   * <p>{@code close()} refuses new calls, stops the runtime with the loop still
   * running so the calls it answers can complete, then cancels whatever is
   * left before stopping the loop: a caller is never left waiting on a loop
   * that no longer runs. A refused or cancelled call is answered as the
   * runtime answers a model that is not ready (503 {@code not_ready}).
   */
  private static final String SHIM_SOURCE = """
    import asyncio
    import concurrent.futures
    import gc
    import threading

    from vllm_srun.config import ModelConfig, ServeConfig
    from vllm_srun.runtime import Runtime

    CLOSED = (503, {"error": {"code": "not_ready", "message": "the decider is closed"}})
    DRAIN_SECONDS = 5


    class Vllm4jDecider:
        def __init__(self, model, serve):
            serve = dict(serve)
            serve["exit_on_device_error"] = False
            self.runtime = Runtime(ServeConfig(models=(ModelConfig(**model),), **serve))
            self.runtime.start(background=False)
            self.lock = threading.Lock()
            self.pending = set()
            self.closing = False
            self.loop = asyncio.new_event_loop()
            self.thread = threading.Thread(
                target=self.loop.run_forever, name="vllm4j-decider", daemon=True
            )
            self.thread.start()

        def decide(self, body):
            with self.lock:
                if self.closing:
                    return CLOSED
                future = asyncio.run_coroutine_threadsafe(
                    self.runtime.call("decisions", body), self.loop
                )
                self.pending.add(future)
            try:
                # result() waits on a lock, which releases the GIL.
                return future.result()
            except concurrent.futures.CancelledError:
                return CLOSED
            finally:
                with self.lock:
                    self.pending.discard(future)

        def health(self):
            return (self.runtime.health.state, self.runtime.health.reason)

        def close(self):
            with self.lock:
                self.closing = True
            try:
                self.runtime.stop()
                with self.lock:
                    pending = list(self.pending)
                concurrent.futures.wait(pending, timeout=DRAIN_SECONDS)
            finally:
                with self.lock:
                    pending = list(self.pending)
                for future in pending:
                    future.cancel()
                self.loop.call_soon_threadsafe(self.loop.stop)
                self.thread.join(DRAIN_SECONDS)
                # The runtime froze the heap after loading (gc.freeze), which
                # also hides a VllmEngine's objects in this interpreter from
                # the cycle collector.
                gc.unfreeze()
    """;

  /** Cached {@code Vllm4jDecider} class, shared by every decider. */
  private static volatile MemorySegment shimClass;

  private final String model;
  private MemorySegment decider;

  /** Guards {@link #closed} and {@link #inFlight}. */
  private final Object lifecycle = new Object();
  private boolean closed = false;
  /** Calls using {@link #decider}; {@link #close()} releases it at zero. */
  private int inFlight = 0;

  /** Obtain via {@link #builder()}. */
  VllmDecider(Arena arena, VllmDeciderBuilder builder) {
    this.model = builder.model();
    try (var gil = GIL.acquire()) {
      MemorySegment cls = ensureShimClass(arena);
      MemorySegment modelKwargs = PythonObjects.toPyDict(
        arena,
        builder.modelConfig()
      );
      MemorySegment serveKwargs = PythonObjects.toPyDict(
        arena,
        builder.serveConfig()
      );
      MemorySegment args = PythonCall.makeTuple(modelKwargs, serveKwargs);
      MemorySegment instance = PythonCall.pyObjectCall(
        cls,
        args,
        MemorySegment.NULL
      );
      PythonTypes.decref(args);
      PythonTypes.decref(serveKwargs);
      PythonTypes.decref(modelKwargs);
      PythonErrors.checkPythonError("loading decision model " + model);
      this.decider = instance;
    }
  }

  /** Returns a builder for a decider. */
  public static VllmDeciderBuilder builder() {
    return new VllmDeciderBuilder();
  }

  /**
   * Answers the questions of one request.
   *
   * @param request the state and questions
   * @return the answers, one per question; a question that failed is an {@link Answer.Failed}
   * @throws DecisionException if the runtime refused the request as a whole
   */
  public DecisionResponse decide(DecisionRequest request) {
    return DecisionResponse.fromMap(decide(request.toMap()));
  }

  /**
   * Answers a raw {@code /v1/decisions} body, for fields this binding does
   * not type (extra {@code states}, {@code set} and {@code span} questions…).
   *
   * @param body the request body, as JSON-like Java values
   * @return the response body
   * @throws DecisionException if the runtime refused the request as a whole
   */
  @SuppressWarnings("unchecked")
  public Map<String, Object> decide(Map<String, Object> body) {
    enter();
    Object result;
    try (var gil = GIL.acquire(); Arena local = Arena.ofConfined()) {
      MemorySegment pyBody = PythonObjects.toPyDict(local, body);
      MemorySegment name = PythonTypes.pyStr(local, "decide");
      MemorySegment pyResult = PythonCall.callMethodObjArgs(
        decider,
        name,
        pyBody
      );
      PythonTypes.decref(name);
      PythonTypes.decref(pyBody);
      try {
        PythonErrors.checkPythonError("decider.decide()");
        result = PythonObjects.toJava(pyResult);
      } finally {
        PythonTypes.decref(pyResult);
      }
    } finally {
      exit();
    }
    List<Object> outcome = (List<Object>) result;
    int status = ((Number) outcome.get(0)).intValue();
    Map<String, Object> response = (Map<String, Object>) outcome.get(1);
    if (status != 200) {
      throw toException(status, response);
    }
    return response;
  }

  /**
   * The model's state: {@code ready}, {@code degraded} (after a device
   * failure) or {@code failed}.
   */
  public String state() {
    return (String) health().get(0);
  }

  /** Why the model is not ready, or {@code null}. */
  public String reason() {
    return (String) health().get(1);
  }

  /** The model this decider serves, as given to the builder. */
  public String model() {
    return model;
  }

  /**
   * Stops the runtime's scheduler and event loop. Idempotent.
   *
   * <p>New calls are refused with {@link IllegalStateException}. Calls already
   * waiting get their answer if the runtime gives it while stopping, or a
   * {@link DecisionException} (503 {@code not_ready}) otherwise; this method
   * returns once they have all returned.
   */
  @Override
  public void close() {
    synchronized (lifecycle) {
      if (closed) return;
      closed = true;
    }
    try {
      // Without waiting for the calls in flight: they need the GIL to return.
      try (var gil = GIL.acquire(); Arena local = Arena.ofConfined()) {
        MemorySegment name = PythonTypes.pyStr(local, "close");
        MemorySegment ignored = PythonCall.callMethodObjArgs(decider, name);
        PythonTypes.decref(name);
        PythonTypes.decref(ignored);
        PythonErrors.checkPythonError("decider.close()");
      }
    } finally {
      awaitNoCallInFlight();
      try (var gil = GIL.acquire()) {
        PythonTypes.decref(decider);
        decider = null;
      }
    }
  }

  // ── Internal ───────────────────────────────────────────────────────────

  @SuppressWarnings("unchecked")
  private List<Object> health() {
    enter();
    try (var gil = GIL.acquire(); Arena local = Arena.ofConfined()) {
      MemorySegment name = PythonTypes.pyStr(local, "health");
      MemorySegment pyHealth = PythonCall.callMethodObjArgs(decider, name);
      PythonTypes.decref(name);
      try {
        PythonErrors.checkPythonError("decider.health()");
        return (List<Object>) PythonObjects.toJava(pyHealth);
      } finally {
        PythonTypes.decref(pyHealth);
      }
    } finally {
      exit();
    }
  }

  /** Registers a call using {@link #decider}; refused once closed. */
  private void enter() {
    synchronized (lifecycle) {
      if (closed) {
        throw new IllegalStateException("VllmDecider is closed");
      }
      inFlight++;
    }
  }

  private void exit() {
    synchronized (lifecycle) {
      if (--inFlight == 0) {
        lifecycle.notifyAll();
      }
    }
  }

  /** Waits until no call uses {@link #decider}, keeping any interrupt. */
  private void awaitNoCallInFlight() {
    boolean interrupted = false;
    synchronized (lifecycle) {
      while (inFlight > 0) {
        try {
          lifecycle.wait();
        } catch (InterruptedException e) {
          interrupted = true;
        }
      }
    }
    if (interrupted) {
      Thread.currentThread().interrupt();
    }
  }

  @SuppressWarnings("unchecked")
  static DecisionException toException(int status, Map<String, Object> body) {
    if (body != null && body.get("error") instanceof Map<?, ?> error) {
      Map<String, Object> e = (Map<String, Object>) error;
      return new DecisionException(
        status,
        String.valueOf(e.get("code")),
        String.valueOf(e.get("message"))
      );
    }
    return new DecisionException(
      status,
      "internal_error",
      String.valueOf(body)
    );
  }

  /**
   * Defines the shim class on first use, with {@code builtins.exec} into a
   * private namespace, as {@link io.gravitee.vllm.engine.VllmEngine} does for
   * its step packer.
   */
  private static MemorySegment ensureShimClass(Arena arena) {
    if (shimClass != null) {
      return shimClass;
    }
    synchronized (VllmDecider.class) {
      if (shimClass != null) {
        return shimClass;
      }
      MemorySegment exec = PythonCall.importClass(arena, "builtins", "exec");
      MemorySegment namespace = CPythonBinding.PyDict_New();
      MemorySegment source = PythonTypes.pyStr(arena, SHIM_SOURCE);
      MemorySegment args = PythonCall.makeTuple(source, namespace);
      MemorySegment ignored = PythonCall.pyObjectCall(
        exec,
        args,
        MemorySegment.NULL
      );
      PythonErrors.checkPythonError(
        "defining the decider shim (is vllm-srun installed in the venv?)"
      );
      PythonTypes.decref(ignored);
      PythonTypes.decref(args);
      PythonTypes.decref(source);
      PythonTypes.decref(exec);

      MemorySegment get = PythonTypes.pyStr(arena, "get");
      MemorySegment key = PythonTypes.pyStr(arena, "Vllm4jDecider");
      MemorySegment cls = PythonCall.callMethodObjArgs(namespace, get, key);
      PythonErrors.checkPythonError("namespace.get(Vllm4jDecider)");
      PythonTypes.decref(key);
      PythonTypes.decref(get);
      PythonTypes.decref(namespace);

      if (PythonTypes.isNull(cls) || PythonTypes.isNone(cls)) {
        throw new VllmException("Could not define the vLLM4j decider shim");
      }
      shimClass = cls;
      return cls;
    }
  }
}
