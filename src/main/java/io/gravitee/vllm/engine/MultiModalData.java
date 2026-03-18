package io.gravitee.vllm.engine;

import io.gravitee.vllm.binding.PythonCall;
import io.gravitee.vllm.binding.PythonErrors;
import io.gravitee.vllm.binding.PythonTypes;

import org.vllm.python.CPython;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Multimodal data to attach to a {@link VllmRequest}.
 *
 * <p>Collects images and/or audio data that will be passed to vLLM as
 * {@code multi_modal_data} alongside the text prompt. The prompt should
 * contain placeholder tokens (e.g. {@code <image>}) matching the model's
 * expected format.
 *
 * <h2>Usage</h2>
 * <pre>{@code
 * var mmData = new MultiModalData()
 *         .addImage(imageBytes)
 *         .addImage(Path.of("photo.jpg"));
 *
 * var request = new VllmRequest("req-1", "Describe this image: <image>", sp, mmData);
 * }</pre>
 *
 * <p><b>Note:</b> Multimodal inference requires a vision-language model (VLM)
 * and a backend that supports it. Currently works on CUDA; vllm-metal does
 * not yet support VLMs.
 */
public final class MultiModalData {

    private final List<byte[]> images = new ArrayList<>();
    private final List<byte[]> audios = new ArrayList<>();

    /**
     * Adds an image from raw bytes (JPEG, PNG, etc.).
     *
     * @param imageData the raw image bytes
     * @return this (fluent)
     */
    public MultiModalData addImage(byte[] imageData) {
        if (imageData == null || imageData.length == 0) {
            throw new IllegalArgumentException("imageData must not be null or empty");
        }
        images.add(imageData);
        return this;
    }

    /**
     * Adds an image from a file path.
     *
     * @param imagePath path to the image file
     * @return this (fluent)
     * @throws IOException if the file cannot be read
     */
    public MultiModalData addImage(Path imagePath) throws IOException {
        return addImage(Files.readAllBytes(imagePath));
    }

    /**
     * Adds audio data from raw bytes (WAV, MP3, etc.).
     *
     * @param audioData the raw audio bytes
     * @return this (fluent)
     */
    public MultiModalData addAudio(byte[] audioData) {
        if (audioData == null || audioData.length == 0) {
            throw new IllegalArgumentException("audioData must not be null or empty");
        }
        audios.add(audioData);
        return this;
    }

    /**
     * Adds audio data from a file path.
     *
     * @param audioPath path to the audio file
     * @return this (fluent)
     * @throws IOException if the file cannot be read
     */
    public MultiModalData addAudio(Path audioPath) throws IOException {
        return addAudio(Files.readAllBytes(audioPath));
    }

    /** Returns the number of images added. */
    public int imageCount() {
        return images.size();
    }

    /** Returns the number of audio clips added. */
    public int audioCount() {
        return audios.size();
    }

    /** Returns {@code true} if any multimodal data has been added. */
    public boolean hasData() {
        return !images.isEmpty() || !audios.isEmpty();
    }

    /**
     * Builds the Python {@code multi_modal_data} dict for vLLM.
     *
     * <p>Structure: {@code {"image": [PIL.Image, ...], "audio": [bytes, ...]}}
     *
     * <p>Images are converted from raw bytes to PIL Image objects via
     * {@code PIL.Image.open(io.BytesIO(data))}.
     *
     * @param arena arena for native memory allocation
     * @return new Python dict reference, or {@code null} if no data
     */
    MemorySegment toPythonDict(Arena arena) {
        if (!hasData()) return null;

        MemorySegment pyDict = CPython.PyDict_New();

        if (!images.isEmpty()) {
            MemorySegment pyImageList = buildImageList(arena);
            // If single image, pass directly; if multiple, pass as list
            if (images.size() == 1) {
                MemorySegment pySingle = CPython.PyList_GetItem(pyImageList, 0);
                PythonTypes.incref(pySingle); // borrowed → new ref
                PythonTypes.putDictObj(arena, pyDict, "image", pySingle);
                PythonTypes.decref(pySingle);
            } else {
                PythonTypes.putDictObj(arena, pyDict, "image", pyImageList);
            }
            PythonTypes.decref(pyImageList);
        }

        if (!audios.isEmpty()) {
            MemorySegment pyAudioList = CPython.PyList_New(0);
            for (byte[] audioData : audios) {
                MemorySegment pyBytes = PythonTypes.pyBytes(arena, audioData);
                CPython.PyList_Append(pyAudioList, pyBytes);
                PythonTypes.decref(pyBytes);
            }
            if (audios.size() == 1) {
                MemorySegment pySingle = CPython.PyList_GetItem(pyAudioList, 0);
                PythonTypes.incref(pySingle);
                PythonTypes.putDictObj(arena, pyDict, "audio", pySingle);
                PythonTypes.decref(pySingle);
            } else {
                PythonTypes.putDictObj(arena, pyDict, "audio", pyAudioList);
            }
            PythonTypes.decref(pyAudioList);
        }

        return pyDict;
    }

    /**
     * Converts raw image bytes to PIL Image objects via
     * {@code PIL.Image.open(io.BytesIO(data))}.
     */
    private MemorySegment buildImageList(Arena arena) {
        // import io, PIL.Image (fresh each time — PyImport_ImportModule is
        // backed by sys.modules so repeated imports are just a dict lookup)
        MemorySegment ioModule = CPython.PyImport_ImportModule(arena.allocateFrom("io"));
        PythonErrors.checkPythonError("import io");
        MemorySegment pilImageModule = CPython.PyImport_ImportModule(arena.allocateFrom("PIL.Image"));
        PythonErrors.checkPythonError("import PIL.Image");

        MemorySegment bytesIOClass = PythonTypes.getAttr(arena, ioModule, "BytesIO");
        MemorySegment imageOpen = PythonTypes.getAttr(arena, pilImageModule, "open");

        MemorySegment pyList = CPython.PyList_New(0);
        for (byte[] imageData : images) {
            // bytesIO = io.BytesIO(pyBytes)
            MemorySegment pyBytes = PythonTypes.pyBytes(arena, imageData);
            MemorySegment bytesIO = PythonCall.callOneArg(bytesIOClass, pyBytes);
            PythonErrors.checkPythonError("io.BytesIO()");
            PythonTypes.decref(pyBytes);

            // img = PIL.Image.open(bytesIO)
            MemorySegment img = PythonCall.callOneArg(imageOpen, bytesIO);
            PythonErrors.checkPythonError("PIL.Image.open()");
            PythonTypes.decref(bytesIO);

            CPython.PyList_Append(pyList, img);
            PythonTypes.decref(img);
        }

        PythonTypes.decref(imageOpen);
        PythonTypes.decref(bytesIOClass);
        PythonTypes.decref(pilImageModule);
        PythonTypes.decref(ioModule);

        return pyList;
    }
}
