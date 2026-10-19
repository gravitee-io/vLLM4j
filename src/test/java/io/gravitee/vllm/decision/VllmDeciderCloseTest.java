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

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import io.gravitee.vllm.binding.CPythonBinding;
import io.gravitee.vllm.binding.PythonCall;
import io.gravitee.vllm.binding.PythonErrors;
import io.gravitee.vllm.binding.PythonTypes;
import io.gravitee.vllm.runtime.GIL;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * {@link VllmDecider#close()} with calls in flight: every caller returns, none
 * is left waiting on a stopped event loop. Each test loads its own decider.
 *
 * <p>Runs in its own JVM (the profile forks per class), so stubbing the
 * runtime here reaches no other test.
 */
@Tag("decision-integration")
class VllmDeciderCloseTest {

  private static VllmDecider newDecider() {
    return VllmDecider.builder()
      .model("vllm-sr/Decision-2.0-Kai-0.6B")
      .device("cpu")
      .build();
  }

  /** Enough questions to keep the model busy while close() runs. */
  private static DecisionRequest slowRequest() {
    var builder = DecisionRequest.builder(
      Map.of("request", "I was charged twice for my subscription this month.")
    );
    for (int i = 0; i < 32; i++) {
      Map<String, String> intents = new LinkedHashMap<>();
      intents.put("billing", "Payment or invoice issue");
      intents.put("account", "Login or access issue");
      builder.choice(
        "intent" + i,
        "What does the user want? (" + i + ")",
        intents
      );
    }
    return builder.build();
  }

  /** An answer, a refusal on entry, or a 503 not_ready from the closing runtime. */
  private static void assertAnsweredOrRefused(Future<?> future)
    throws Exception {
    try {
      assertThat(future.get(60, TimeUnit.SECONDS)).isNotNull();
    } catch (ExecutionException e) {
      assertThat(e.getCause()).satisfiesAnyOf(
        cause -> assertThat(cause).isInstanceOf(IllegalStateException.class),
        cause ->
          assertThat(cause).isInstanceOfSatisfying(
            DecisionException.class,
            d -> {
              assertThat(d.status()).isEqualTo(503);
              assertThat(d.code()).isEqualTo("not_ready");
            }
          )
      );
    }
  }

  /** Makes every later {@code Runtime.call} wait forever, as a stuck forward would. */
  private static void stallRuntimeCalls() {
    String source = """
      import asyncio
      from vllm_srun.runtime import Runtime

      async def stalled(self, *args, **kwargs):
          await asyncio.Event().wait()

      Runtime.call = stalled
      """;
    try (var gil = GIL.acquire(); Arena arena = Arena.ofConfined()) {
      MemorySegment exec = PythonCall.importClass(arena, "builtins", "exec");
      MemorySegment code = PythonTypes.pyStr(arena, source);
      MemorySegment namespace = CPythonBinding.PyDict_New();
      MemorySegment args = PythonCall.makeTuple(code, namespace);
      MemorySegment result = PythonCall.pyObjectCall(
        exec,
        args,
        MemorySegment.NULL
      );
      PythonErrors.checkPythonError("stalling Runtime.call");
      PythonTypes.decref(result);
      PythonTypes.decref(args);
      PythonTypes.decref(namespace);
      PythonTypes.decref(code);
      PythonTypes.decref(exec);
    }
  }

  @Test
  void calls_the_runtime_never_answers_are_released_by_close()
    throws Exception {
    var decider = newDecider();
    stallRuntimeCalls();
    // Daemon threads, shut down without waiting: a regression must fail the
    // test, not hang the JVM on callers that never return.
    var pool = Executors.newFixedThreadPool(4, task -> {
      var thread = new Thread(task);
      thread.setDaemon(true);
      return thread;
    });
    try {
      List<Future<DecisionResponse>> futures = new ArrayList<>();
      for (int i = 0; i < 4; i++) {
        futures.add(pool.submit(() -> decider.decide(slowRequest())));
      }
      Thread.sleep(500);
      assertThat(futures).noneMatch(Future::isDone);

      CompletableFuture.runAsync(decider::close).get(60, TimeUnit.SECONDS);
      for (var future : futures) {
        assertThatThrownBy(() -> future.get(30, TimeUnit.SECONDS))
          .isInstanceOf(ExecutionException.class)
          .cause()
          .isInstanceOfSatisfying(DecisionException.class, d -> {
            assertThat(d.status()).isEqualTo(503);
            assertThat(d.code()).isEqualTo("not_ready");
          });
      }
    } finally {
      pool.shutdownNow();
    }
  }

  @Test
  void calls_racing_close_all_return() throws Exception {
    var decider = newDecider();
    try (var pool = Executors.newFixedThreadPool(8)) {
      List<Future<Integer>> loops = new ArrayList<>();
      for (int i = 0; i < 8; i++) {
        loops.add(
          pool.submit(() -> {
            int answered = 0;
            while (true) {
              decider.decide(slowRequest());
              answered++;
            }
          })
        );
      }
      Thread.sleep(500);
      decider.close();
      for (var loop : loops) {
        assertAnsweredOrRefused(loop);
      }
    }
  }

  @Test
  void a_closed_decider_refuses_calls() {
    var decider = newDecider();
    decider.close();
    decider.close();
    assertThatThrownBy(() -> decider.decide(slowRequest())).isInstanceOf(
      IllegalStateException.class
    );
    assertThatThrownBy(decider::state).isInstanceOf(
      IllegalStateException.class
    );
  }
}
