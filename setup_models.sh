#!/bin/bash
set -e

echo "Creating models directory..."
mkdir -p models

echo "Downloading Official Sherpa-ONNX Parakeet TDT Model..."
# Use wget instead of curl (curl might be missing)
wget -q --show-progress -O model.tar.bz2 "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2"

echo "Extracting Parakeet Model..."
tar xvf model.tar.bz2 -C models/

# Verify extraction
echo "Directory content after extraction:"
ls -la models/

# Move to expected directory
# Expected: models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8
if [ -d "models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8" ]; then
    echo "Moving model directory..."
    rm -rf models/parakeet-tdt-0.6b-v3-int8
    mv models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8 models/parakeet-tdt-0.6b-v3-int8
else
    echo "ERROR: Expected extracted directory 'models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8' not found!"
    echo "Contents of models/:"
    ls -la models/
    exit 1
fi

rm model.tar.bz2

echo "Downloading Silero VAD Model..."
wget -q --show-progress -O models/silero_vad.onnx https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx

echo "Final models directory structure:"
ls -R models/

echo "Model setup complete."
