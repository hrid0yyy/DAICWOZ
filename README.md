# DAIC-WOZ Patient Audio Spectrogram Pipeline

This guide explains how to process the DAIC-WOZ dataset to extract patient speech, segment it, and generate spectrograms.

## 1. Download and Extract Data

- Download the required interview zip file from the DAIC-WOZ dataset (e.g., `450_p.zip`).
- Extract the contents to the following structure:
  ```
  interviews/
    450/
        450_TRANSCRIPT.csv
        450_AUDIO.wav
        ...
  ```
  - The `interviews` directory should be in your project root.

## 2. Run the Processing Notebook

Open `speech_processing.ipynb` and run the following blocks in order:

### Block 1: Patient Speech Segmentation

- Segments the patient's speech from the full interview audio.
- Output: `Segmented/{interview_id}_PATIENT.wav`

### Block 2: Chunking

- Splits the segmented patient audio into 8-second chunks.
- Output: `Chunks/{interview_id}_clip_{n}.wav`

### Block 3: Spectrogram Generation

- Generates a spectrogram for each audio chunk.
- Output: `Spectrograms/{interview_id}_clip_{n}.wav.npy`

## 3. Output

After running all blocks, you will have:
- Patient-only audio files in the `Segmented` folder.
- 8-second audio chunks in the `Chunks` folder.
- Spectrograms (as numpy arrays) in the `Spectrograms` folder.

---

**Note:**  
- Make sure all required Python packages are installed (`pandas`, `pydub`, `librosa`, `matplotlib`, `numpy`).
- Some DLL dependencies come from the Microsoft Visual C++ runtime.  
  If you see errors about missing DLLs (such as `msvcp140.dll` or similar), download and install the [Microsoft Visual C++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170) (choose the appropriate version, usually x64 for modern systems).

