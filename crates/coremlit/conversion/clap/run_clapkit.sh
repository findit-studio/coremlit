#!/bin/bash
# clapkit T1 conversion pipeline — re-derives both CLAP encoders deterministically.
# Toolchain is the conv/ phase venv (coremltools 9.0, torch 2.5.1, transformers
# 5.14.0, numpy 1.26.4, python 3.11.15). Source: laion/clap-htsat-unfused pinned to
# revision 8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a (see scripts/_clap_common.py).
set -e
CONV=/private/tmp/claude-501/-Users-al-Developer-findit-studio-coremlit/2e543e17-c5e2-4187-be75-b6b4fafe4418/scratchpad/conv
CK="$CONV/clapkit"
PY="$CONV/.venv/bin/python"
cd "$CK"

# 1. Convert both towers -> fp16 (shipped) + fp32 (verification reference) mlpackages.
#    convert_audio.py installs the exact bicubic->matmul resize shim (coremltools 9.0
#    lacks upsample_bicubic2d) and asserts it reproduces the un-patched embedding.
#    Both scripts register the `new_ones` custom op (_custom_ops.py).
"$PY" -u scripts/convert_audio.py
"$PY" -u scripts/convert_text.py

# 2. Compile every mlpackage -> mlmodelc (the audit + tests consume model.mil).
cd "$CK/staging"
for p in clap_audio clap_text clap_audio_fp32 clap_text_fp32; do
  xcrun coremlcompiler compile "$p.mlpackage" .
done

# 3. fp16 guard audit on the shipped fp16 models (must print RESULT: ALL CLEAN).
mkdir -p audit_fp16 && cd audit_fp16
ln -sfn ../clap_audio.mlmodelc clap_audio.mlmodelc
ln -sfn ../clap_text.mlmodelc  clap_text.mlmodelc
"$CONV/fp16audit/target/release/fp16audit" .
cd "$CK"

# 4. Verify: PyTorch fp32 vs CoreML fp32 (CPU), and CoreML fp16 vs fp32 per unit.
"$PY" -u scripts/verify_encoders.py

# 5. Mel-in-graph decision probe (records why the mel frontend stays in Rust).
#    The probe converts BOTH precisions (clap_audio_melgraph_fp32/fp16.mlpackage)
#    and fails closed (nonzero exit) if either conversion fails, so `set -e` halts
#    before measurement can read a stale artifact. measure_melgraph then loads
#    both melgraph mlpackages (no separate compile step) and the shipped fp32
#    spectrogram graph, feeding all arms the identical repeat-tiled 480k input.
"$PY" -u scripts/mel_in_graph_probe.py
"$PY" -u scripts/measure_melgraph.py

echo "clapkit T1 pipeline complete."
