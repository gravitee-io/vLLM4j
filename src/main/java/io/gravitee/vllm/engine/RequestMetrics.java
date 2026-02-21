package io.gravitee.vllm.engine;

/**
 * Timing and stats for a single request, extracted from vLLM's
 * {@code RequestMetrics} Python object.
 *
 * <p>All time fields are in epoch seconds (double). A value of {@code -1}
 * indicates the field was unavailable for this request/vLLM version.
 *
 * @param arrivalTime         wall-clock time when the request arrived
 * @param lastTokenTime       wall-clock time when the last token was generated, or -1
 * @param firstScheduledTime  wall-clock time when the request was first scheduled, or -1
 * @param firstTokenTime      wall-clock time when the first token was generated, or -1
 * @param timeInQueue         time spent waiting in the queue (seconds), or -1
 * @param finishedTime        wall-clock time when the request finished, or -1
 */
public record RequestMetrics(
        double arrivalTime,
        double lastTokenTime,
        double firstScheduledTime,
        double firstTokenTime,
        double timeInQueue,
        double finishedTime) {

    /** A sentinel for when metrics are unavailable. */
    public static final RequestMetrics EMPTY = new RequestMetrics(0, -1, -1, -1, -1, -1);

    /**
     * Backward-compatible constructor with the original three fields.
     *
     * @deprecated Use the full constructor. This maps the legacy fields:
     * {@code firstTokenLatency} → computed from arrival + first token times,
     * {@code numGenerationTokens} → not tracked here (available in CompletionOutput).
     */
    public RequestMetrics(double arrivalTime, double firstTokenLatency, int numGenerationTokens) {
        this(arrivalTime,
             -1,
             -1,
             firstTokenLatency >= 0 ? arrivalTime + firstTokenLatency : -1,
             -1,
             -1);
    }

    /** Time-to-first-token in seconds, or -1 if unavailable. */
    public double firstTokenLatency() {
        if (firstTokenTime >= 0 && arrivalTime > 0) {
            return firstTokenTime - arrivalTime;
        }
        return -1;
    }

    /** Time-to-first-token in milliseconds, or -1 if unavailable. */
    public double ttftMs() {
        double latency = firstTokenLatency();
        return latency >= 0 ? latency * 1000.0 : -1;
    }

    /** Total generation time in seconds (arrival → finished), or -1 if unavailable. */
    public double totalTimeSeconds() {
        if (finishedTime >= 0 && arrivalTime > 0) {
            return finishedTime - arrivalTime;
        }
        return -1;
    }

    /** Total generation time in milliseconds, or -1 if unavailable. */
    public double totalTimeMs() {
        double total = totalTimeSeconds();
        return total >= 0 ? total * 1000.0 : -1;
    }
}
