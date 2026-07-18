"""Real audio clips (resampled to 48 kHz, deterministic 10 s window) + a varied
text prompt set, for the conversion-verification chain. No randomness: each clip
is sliced to exactly the first 480_000 samples (or repeat-padded by the HF feature
extractor if shorter), so ClapFeatureExtractor never hits its rand_trunc branch."""
import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal as sps

TARGET_SAMPLES = 480_000
SR = 48_000

# Diverse real audio: music, dialogue, SFX, ambient, speech (multiple languages).
AUDIO_CLIPS = [
    ("music_multi_instrument", "/Users/al/Developer/findit-studio/indexer/assets/audios/04_音乐_三轨道不同乐器.wav"),
    ("voice_dialogue", "/Users/al/Developer/findit-studio/indexer/assets/audios/01_人声_自录双人对话.wav"),
    ("sfx_service_bell", "/Users/al/Developer/findit-studio/indexer/assets/audios/02_音效_服务铃.wav"),
    ("sfx_canned", "/Users/al/Developer/findit-studio/indexer/assets/audios/02_音效_罐头音效.wav"),
    ("ambient_thunderstorm", "/Users/al/Developer/findit-studio/indexer/assets/audios/03_环境声_雷雨.wav"),
    ("ambient_airport", "/Users/al/Developer/findit-studio/indexer/assets/videos/01_airport.mp4.wav"),
    ("clap_sample_48k", "/Users/al/Developer/findit-studio/textclap/tests/fixtures/sample.wav"),
    ("speech_dialogue_16k", "/Users/al/Developer/findit-studio/diarization/tests/parity/fixtures/01_dialogue/clip_16k.wav"),
    ("speech_ted_16k", "/Users/al/Developer/findit-studio/coremlit/crates/whisperkit/tests/fixtures/audio/ted_60.wav"),
    ("speech_mrbeast_16k", "/Users/al/Developer/findit-studio/diarization/tests/parity/fixtures/09_mrbeast_dollar_date/clip_16k.wav"),
]

TEXT_PROMPTS = [
    "a dog barking",
    "gentle rain on a tin roof",
    "a violin playing a slow melody in a concert hall",
    "the sound of a vacuum cleaner",
    "people talking in a busy restaurant",
    "a car engine revving loudly",
    "birds chirping in a forest at dawn",
    "electronic dance music with a heavy bass line",
    "a baby crying",
    "waves crashing on a rocky shore",
    "一只猫在喵喵叫",                      # CJK: a cat meowing
    "applause and cheering from a large crowd",
]


def _load_wav_48k(path):
    sr, data = wavfile.read(path)
    a = data.astype(np.float32)
    if a.ndim > 1:
        a = a.mean(axis=1)
    if np.issubdtype(data.dtype, np.integer):
        a = a / float(np.iinfo(data.dtype).max + 1)
    if sr != SR:
        from math import gcd
        g = gcd(int(sr), SR)
        a = sps.resample_poly(a, SR // g, int(sr) // g).astype(np.float32)
    # Deterministic: first 10 s (or the whole clip if shorter; the fe repeat-pads).
    if len(a) > TARGET_SAMPLES:
        a = a[:TARGET_SAMPLES]
    return a.astype(np.float32)


def audio_clips():
    """Yield (name, samples_f32_48k) for every real clip."""
    for name, path in AUDIO_CLIPS:
        yield name, _load_wav_48k(path)


def input_features(processor, samples):
    """HF mel [1,1,1001,64] for a 48 kHz mono clip."""
    feats = processor.feature_extractor(samples, sampling_rate=SR, return_tensors="pt")
    return feats["input_features"], feats.get("is_longer")
