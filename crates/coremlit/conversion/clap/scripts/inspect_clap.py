"""Inspect laion/clap-htsat-unfused: config, feature extractor, module structure,
and the I/O contract of both towers. Downloads the model on first run (cached
under ~/.cache/huggingface). Read-only; writes nothing but stdout."""
import json
import numpy as np
import torch
from transformers import ClapModel, ClapProcessor, AutoConfig

MODEL_ID = "laion/clap-htsat-unfused"

cfg = AutoConfig.from_pretrained(MODEL_ID)
print("=== TOP CONFIG ===")
print("projection_dim:", cfg.projection_dim)
print("logit_scale_init_value:", getattr(cfg, "logit_scale_init_value", None))
print("hidden_size (text?):", getattr(cfg.text_config, "hidden_size", None))
print("=== AUDIO CONFIG (subset) ===")
ac = cfg.audio_config
for k in ["hidden_size", "num_mel_bins", "spec_size", "patch_size", "patch_stride",
          "window_size", "num_classes", "depths", "num_attention_heads",
          "d_proj", "projection_hidden_act", "enable_fusion", "fusion_type",
          "hidden_act", "layer_norm_eps"]:
    print(f"  {k}:", getattr(ac, k, "<none>"))
print("=== TEXT CONFIG (subset) ===")
tc = cfg.text_config
for k in ["model_type", "hidden_size", "num_hidden_layers", "num_attention_heads",
          "max_position_embeddings", "vocab_size", "pad_token_id",
          "layer_norm_eps", "hidden_act", "projection_hidden_act"]:
    print(f"  {k}:", getattr(tc, k, "<none>"))

print("\n=== PROCESSOR / FEATURE EXTRACTOR ===")
proc = ClapProcessor.from_pretrained(MODEL_ID)
fe = proc.feature_extractor
for k in ["feature_size", "sampling_rate", "hop_length", "fft_window_size",
          "n_fft", "frequency_min", "frequency_max", "top_db", "truncation",
          "padding", "max_length_s", "nb_max_samples", "nb_max_frames"]:
    print(f"  {k}:", getattr(fe, k, "<none>"))
print("  full fe dict keys:", sorted(vars(fe).keys()))

print("\n=== MODEL LOAD ===")
model = ClapModel.from_pretrained(MODEL_ID).eval()
print("audio_projection:", model.audio_projection)
print("text_projection:", model.text_projection)
print("logit_scale_a:", float(model.logit_scale_a.exp()) if hasattr(model, "logit_scale_a") else "n/a")
print("logit_scale_t:", float(model.logit_scale_t.exp()) if hasattr(model, "logit_scale_t") else "n/a")

print("\n=== AUDIO TOWER top-level children ===")
for n, m in model.audio_model.named_children():
    print(" ", n, type(m).__name__)
print("=== TEXT TOWER top-level children ===")
for n, m in model.text_model.named_children():
    print(" ", n, type(m).__name__)

# Probe feature extractor output shape on 10s of noise at 48k.
print("\n=== FEATURE EXTRACTOR OUTPUT ON 10s@48k ===")
rng = np.random.RandomState(0)
audio = rng.randn(480000).astype(np.float32) * 0.1
feats = fe(audio, sampling_rate=48000, return_tensors="pt")
for k, v in feats.items():
    print(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype}")

# get_audio_features / get_text_features output shapes + norm check.
print("\n=== get_audio_features ===")
with torch.no_grad():
    af = model.get_audio_features(input_features=feats["input_features"],
                                  is_longer=feats.get("is_longer"))
print("  audio_features shape:", tuple(af.shape), "norm:", float(af.norm(dim=-1)[0]))

print("\n=== get_text_features ===")
tok = proc.tokenizer(["a dog barking", "rain on a roof"], padding=True, return_tensors="pt")
for k, v in tok.items():
    print(f"  tok {k}: shape={tuple(v.shape)} dtype={v.dtype}")
with torch.no_grad():
    tf = model.get_text_features(input_ids=tok["input_ids"],
                                 attention_mask=tok["attention_mask"])
print("  text_features shape:", tuple(tf.shape), "norm[0]:", float(tf.norm(dim=-1)[0]))
print("DONE")
