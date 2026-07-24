"""The committed siglip corpus registry: 6 distinct CC0 / public-domain photos
(Wikimedia Commons) and 8 texts, with each image's exact source URL + license
recorded (the ``ImageGolden.source``/``license`` schema requires it).

The PNGs live under ``$SIGLIP_GOLDENS/images/<id>.png`` (committed). Downloaded
once at implementation time from the ``source`` URLs below, converted to 8-bit
RGB, and downscaled to the recorded geometry (one at exactly 320x240 so its patch
grid is (19, 26) at the 512 budget — the cross-link to the committed Rust
budget-solver oracle). Aspect spread: landscape, near-square, and portrait.
"""
import os

from PIL import Image

# id -> (source page URL, license, caption text-id). Distinct, unambiguous subjects
# so caption->image top-1 is far inside siglip2-base ability.
IMAGES = [
    {
        "id": "cat",
        "source": "https://commons.wikimedia.org/wiki/File:Closeup_of_a_cat_with_green_eyes%27_face_looking_at_the_viewer.jpg",
        "license": "Public domain",
        "caption_id": "cap_cat",
    },
    {
        "id": "dog",
        "source": "https://commons.wikimedia.org/wiki/File:Fawn_and_white_Welsh_Corgi_puppy_standing_on_rear_legs_and_sticking_out_the_tongue.jpg",
        "license": "CC0",
        "caption_id": "cap_dog",
    },
    {
        "id": "bus",
        "source": "https://commons.wikimedia.org/wiki/File:Double-Decker_Bus_-_DPLA_-_8969d2d710cec47760196b71049b450c.jpg",
        "license": "Public domain",
        "caption_id": "cap_bus",
    },
    {
        "id": "sunflower",
        "source": "https://commons.wikimedia.org/wiki/File:Sunflower_field_at_sunset.jpg",
        "license": "CC0",
        "caption_id": "cap_sunflower",
    },
    {
        "id": "mountain",
        "source": "https://commons.wikimedia.org/wiki/File:Snow-capped_mountain_range.jpg",
        "license": "CC0",
        "caption_id": "cap_mountain",
    },
    {
        "id": "sailboat",
        "source": "https://commons.wikimedia.org/wiki/File:Sonic_23_sailboat_Gyp_Sea_5121.jpg",
        "license": "CC0",
        "caption_id": "cap_sailboat",
    },
]

# 8 texts: the 6 lowercase-ASCII captions (each some image's caption_id), one
# MixedCase twin of cap_cat (its window must equal the lowercase twin's — the
# lowercase non-vacuity pair), and one >64-token sentence (sticky-EOS truncation
# proof: id[63] == eos, no pad). ASCII only, so tokenizers.Lowercase == str.lower.
TEXTS = [
    {"id": "cap_cat", "text": "a photo of a cat"},
    {"id": "cap_dog", "text": "a photo of a corgi puppy"},
    {"id": "cap_bus", "text": "a red double-decker bus on a city street"},
    {"id": "cap_sunflower", "text": "a field of sunflowers at sunset"},
    {"id": "cap_mountain", "text": "a snow-capped mountain range"},
    {"id": "cap_sailboat", "text": "a sailboat on the water"},
    {"id": "mixedcase_cat", "text": "A Photo Of A Cat"},
    {
        "id": "long_truncated",
        "text": (
            "a very long and extremely detailed descriptive sentence about an "
            "ordinary domestic cat sitting quietly on a warm windowsill in the late "
            "afternoon sunlight, watching the small birds and the slowly falling "
            "autumn leaves outside the window while the busy city street far below "
            "fills up with hurrying people, passing cars, red buses, bicycles, and "
            "the faint distant sound of somebody playing music somewhere nearby"
        ),
    },
]


def goldens_dir():
    d = os.environ.get("SIGLIP_GOLDENS")
    if not d:
        raise SystemExit("required environment variable SIGLIP_GOLDENS is unset")
    return d


def image_path(image_id):
    return os.path.join(goldens_dir(), "images", f"{image_id}.png")


def load_pil(image_id):
    """Load a committed corpus PNG as an 8-bit RGB PIL image (the exact bytes the
    Rust side decodes)."""
    return Image.open(image_path(image_id)).convert("RGB")
