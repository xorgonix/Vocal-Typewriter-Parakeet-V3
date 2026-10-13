# How to Fix "Exit Code 1" & OOM Build Crashes on Hugging Face Spaces

## The Problem
You are seeing a build error that ends with `Job failed with exit code: 1` during a `RUN` command (like `cargo build` or `make`). This usually happens after a "cache miss".

On cloud platforms like **Hugging Face Spaces (Free Tier)**, the build environment is extremely limited in **RAM (Memory)**. 

When you try to **compile code from source** (using Rust, C++, or compiling Python libraries), the compiler spawns multiple parallel threads to speed things up. These threads consume more RAM than the container has (often < 16GB), causing the system to kill the process to save itself.

## Why it works "Here" (Python TTS Project)
In this Python TTS project, we rarely compile anything.
1. We use `pip install`.
2. Python packages usually come as **Wheels (`.whl`)**.
3. A "Wheel" is a **pre-compiled binary**. Someone else with a powerful computer already did the hard work.
4. We just download and unzip it. This takes almost **zero RAM**.

## Why it fails "There" (Rust/Parakeet Project)
In your Rust project, the Dockerfile runs:
```dockerfile
RUN cargo build --release
```
This forces the weak cloud computer to compile the **entire application from scratch**. 
*   Rust is famous for high memory usage during compilation.
*   By default, it tries to use all CPU cores.
*   **Result:** 4 cores * 2GB RAM per core = 8GB RAM required -> **CRASH**.

---

## Solution 1: The "Quick Fix" (Limit Concurrency)
You can tell the compiler to "slow down" and use only **one core** at a time. This drastically reduces memory usage, preventing the crash.

**Add this to your Dockerfile** immediately before the build command:

```dockerfile
# Force Cargo to use only 1 CPU core to save RAM
ENV CARGO_BUILD_JOBS=1

# Now run the build
RUN cargo build --release
```

*   **Pros:** Very easy to add.
*   **Cons:** The build will take longer (but it will actually finish).

## Solution 2: The "Pro Fix" (Cross-Compilation)
Do exactly what Python "Wheels" do: **Build it on your powerful Mac, then upload the finished file.**

1.  **On your Mac (Local Machine):**
    Install the Linux cross-compiler target:
    ```bash
    rustup target add x86_64-unknown-linux-gnu
    # You may need a cross-linker tool like 'messense/homebrew-macos-cross-toolchains' or run this inside a local Docker container.
    ```
    
    *Easier method:* Run a local Docker container to build it, then copy the file out.
    ```bash
    # Run a builder container locally
    docker run --rm -v "$PWD":/usr/src/myapp -w /usr/src/myapp rust:latest cargo build --release
    ```

2.  **In your Dockerfile (Cloud):**
    Delete the `RUN cargo build` step. Instead, copy the file you just built locally.
    ```dockerfile
    COPY ./target/release/my-app /app/server
    ```

*   **Pros:** Zero RAM usage on the server. Instant deployment.
*   **Cons:** Requires setting up cross-compilation tools locally.

---

# Runtime Crashes: "Metadata" & Model Mismatches

Sometimes the app builds fine but crashes immediately when starting (Runtime Error / Exit Code 255). This often happens when the **Code** expects one thing, but the **Model** provides another.

## The Error: `window_size does not exist`
You might see logs like:
> `INFO:parakeet-server:Initializing model...`
> `Error: 'window_size' does not exist in the metadata`

## The Cause: Online vs. Offline Mismatch
You are trying to use an **Offline Model** (like Parakeet TDT) with an **Online (Streaming) Recognizer**.
*   **Online Models** (Streaming) need specific metadata like `window_size` and `chunk_size` to know how to process audio in small pieces.
*   **Offline Models** (File-based) process the entire audio at once. They do not have this metadata.
*   When the code uses `OnlineRecognizer` to load an offline model, it panics because it can't find the streaming settings.

## The Fix: Architecture Matching + VAD
To fix this, we must align the code with the model's actual architecture while keeping the "live" user experience.

### 1. Switch to OfflineRecognizer
We updated `app.py` to use `sherpa_onnx.OfflineRecognizer`. This class is designed for the Parakeet TDT model and does not require streaming metadata.

### 2. Add Voice Activity Detection (VAD)
Since `OfflineRecognizer` can't handle a continuous open mic stream, we added **Silero VAD** (Voice Activity Detector).
*   **How it works:** The VAD listens to the microphone. When it detects a pause in speech, it "cuts" the audio segment.
*   **Processing:** This finished segment is then sent to the `OfflineRecognizer` as a complete file.
*   **Result:** The user gets a "streaming-like" experience (text appears sentence-by-sentence) using a high-quality offline model.

---

# The "Final Boss": Output Dimension Mismatch (8198 vs 8193)

Even after fixing the architecture, we hit a critical crash:
> `vocab_size: 8193 != output_size: 8198`

## The Problem
The Parakeet TDT 0.6b-v3 model was exported with an output dimension of **8198**. However, the actual vocabulary (tokens used) plus the blank symbol only added up to **8193**.
*   **Sherpa-ONNX Rule**: The model's output dimension MUST exactly equal `vocab_size + 1`.
*   **Conflict**: We set `vocab_size=8192` (so expected 8193), but the model kept giving 8198.

## The Solution: Physical ONNX Surgery

We couldn't just change a config file. We had to physically modify the model's computational graph to discard the extra 5 outputs.

### Step 1: `slice_joiner.py` (The Surgeon)
We wrote a Python script using the `onnx` library to:
1.  Load `joiner.int8.onnx`.
2.  Find the final output node.
3.  **Inject a `Slice` Operator**: We inserted a new node into the graph that takes the original output and "slices" it, keeping only indices `0` to `8192` (total 8193).
4.  Rewrite the output shape metadata to match.
5.  Save the modified model.

### Step 2: Build-Time Execution (The Deployment)
Originally, we ran this script when the app started (`CMD` in Dockerfile). This caused issues because sometimes the model wasn't downloaded yet, or the changes weren't persisted.

**The Fix:** We moved the fix commands to the **Build Phase** (`RUN` in Dockerfile).
```dockerfile
# Dockerfile
# ...
# Run setup script to download models
RUN chmod +x setup_models.sh && ./setup_models.sh

# Verify download
RUN echo "Verifying model files..." && ls -R models/

# Fix metadata and tokens (Run during build to bake into image)
# 1. slice_joiner.py -> Cuts model from 8198 to 8193
# 2. fix_metadata.py -> Injects missing TDT fields
# 3. fix_tokens.py -> Generates matching token list
RUN python slice_joiner.py && python fix_metadata.py && python fix_tokens.py
```

Now, the Docker image contains the **pre-fixed, perfectly aligned model**. When the app starts, it just loads and runs.
