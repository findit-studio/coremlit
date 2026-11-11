"""The R1 corpus: what the goldens embed, under which prompt.

A text item is ``(id, prompt, text)``, where ``prompt`` is one of
``("bare",)``, ``("named", key)``, ``("document", title)`` or
``("custom", prefix)``. Its ids are the joint encoding of ``prefix + text``.

FluidUse's 30 tokenizer cases are read from FluidUse's own fixture (FluidUse
``b51f7ad``, ``Tests/FluidUseTests/Fixtures/embeddinggemma2-tokenizer-cases.json``)
and embedded bare: their texts already carry any prefix they test.
"""
import json

FLUIDUSE_REV = "b51f7adfbade197f0a9c802e34ba8a98cd4ecbc9"
FLUIDUSE_CASES = "Tests/FluidUseTests/Fixtures/embeddinggemma2-tokenizer-cases.json"

# config.json's eight prompts.
NAMED = {
    "document": "title: none | text: ",
    "search_query": "task: search result | query: ",
    "question_answering": "task: question answering | query: ",
    "fact_checking": "task: fact checking | query: ",
    "code_retrieval": "task: code retrieval | query: ",
    "classification": "task: classification | query: ",
    "clustering": "task: clustering | query: ",
    "sentence_similarity": "task: sentence similarity | query: ",
}

BARE = ("bare",)


def prefix_of(prompt):
    kind = prompt[0]
    if kind == "bare":
        return ""
    if kind == "named":
        return NAMED[prompt[1]]
    if kind == "document":
        return f"title: {prompt[1]} | text: "
    if kind == "custom":
        return prompt[1]
    raise ValueError(prompt)


def prompt_record(prompt):
    kind = prompt[0]
    if kind == "bare":
        return {"kind": "bare"}
    if kind == "named":
        return {"kind": "named", "key": prompt[1]}
    if kind == "document":
        return {"kind": "document", "title": prompt[1]}
    return {"kind": "custom", "prefix": prompt[1]}


# Ours: English, multilingual, code, emoji with ZWJ, whitespace, scalars the
# vocabulary lacks, literal special strings and the one bridging merge.
OURS = [
    ("en-query", ("named", "search_query"), "waterproof shoes for trail running"),
    ("en-doc", ("named", "document"), "The northern lights are caused by charged particles from the sun."),
    ("en-question", ("named", "search_query"), "What causes the northern lights?"),
    ("en-bare", BARE, "A quick brown fox jumps over the lazy dog."),
    (
        "en-titled",
        ("document", "Q3 earnings call"),
        "Quarterly revenue grew 12% year over year, driven by subscriptions.",
    ),
    ("de-doc", ("named", "document"), "Der schnelle braune Fuchs springt über den faulen Hund."),
    ("fr-cluster", ("named", "clustering"), "Le renard brun rapide saute par-dessus le chien paresseux."),
    ("zh-doc", ("named", "document"), "敏捷的棕色狐狸跳过了懒狗。"),
    ("ja-query", ("named", "search_query"), "素早い茶色の狐がのろまな犬を飛び越える。"),
    ("ru-bare", BARE, "Быстрая коричневая лиса прыгает через ленивую собаку."),
    ("ar-doc", ("named", "document"), "مرحبا بالعالم، هذا اختبار."),
    ("hi-cluster", ("named", "clustering"), "नमस्ते दुनिया, यह एक परीक्षण है।"),
    ("ko-bare", BARE, "안녕하세요 세계, 이것은 테스트입니다."),
    ("code-rust", ("named", "document"), 'fn main() {\n    println!("hello");\n}'),
    ("code-sql", ("named", "search_query"), "SELECT id, name FROM users WHERE age > 30 ORDER BY name;"),
    ("code-py", BARE, "def f(x):\n\treturn x ** 2  # square"),
    ("emoji-family", ("named", "document"), "family: 👨‍👩‍👧‍👦 and flags 🇯🇵🇫🇷"),
    ("emoji-tones", BARE, "skin tones 👋🏽👋🏿 and keycap 1️⃣"),
    ("ws-tabs", ("named", "document"), "tab\tseparated\tvalues\nnew line\r\nwindows line"),
    ("ws-edges", BARE, "   three leading, two trailing  "),
    ("ws-run", BARE, "a     b"),
    ("ws-newlines", BARE, "\n\n\n"),
    ("ws-space", BARE, " "),
    ("fallback-rare", BARE, "rare scalars: \U00013000 \U0001D518\U0001D52B \U000F0000 \U0010FFFD"),
    ("fallback-control", BARE, "control \u0001\u001f and delete \u007f"),
    ("bom-twin", BARE, "﻿BOM-prefixed and plain BOM-prefixed"),
    ("combining", BARE, "é against é"),
    ("special-image-open", BARE, "<|image>"),
    ("special-image-placeholder", BARE, "<|image|>"),
    ("special-image-close", BARE, "<image|>"),
    ("special-inline", ("named", "document"), "look at <|image> and <|audio> here"),
    ("special-adjacent", BARE, "x<eos>y<bos>z"),
    ("special-spaced", BARE, " <pad> "),
    ("special-run", BARE, "<|image|><|audio|><|video|>"),
    ("special-turns", ("named", "search_query"), "<turn|> <|turn> <|tool_call>"),
    ("special-mask", ("named", "clustering"), "<mask> is not a word"),
    ("special-incomplete", BARE, "<|image and <image"),
    ("bridge", BARE, "<b> </b>"),
    ("bridge-inline", ("named", "document"), "use <b> </b> tags: <i> </i>"),
    ("numbers", ("named", "document"), "3.14159 2.71828 1,000,000 0x1F 1e-7"),
    ("url", ("named", "search_query"), "https://example.com/path?query=a&b=c#frag"),
    ("custom-plain", ("custom", "task: legal clause retrieval | query: "), "termination for convenience"),
]

# The items whose vectors are also golden under `split_special_tokens = true`.
SPECIAL_IDS = {
    "special-image-open",
    "special-image-placeholder",
    "special-image-close",
    "special-inline",
    "special-adjacent",
    "special-spaced",
    "special-run",
    "special-turns",
    "special-mask",
    "special-incomplete",
}

# The four custom-prefix boundary cases, each golden under both settings:
#   1. a prefix without a trailing space, before content beginning `</`: the
#      colon and `</` merge across the seam;
#   2. a prefix ending in `>`, before content beginning ` </`: the bridging
#      merge (`>`, `▁</`) across the seam;
#   3. content beginning with spaces after a prefix without one: `▁▁bar`;
#   4. a special literal spanning the seam: `<|ima` + `ge>`.
BOUNDARY = [
    ("custom-colon-seam", ("custom", "task: legal clause retrieval | query:"), "</task> hi"),
    ("custom-bridge-seam", ("custom", "markup: <b>"), " </b> bold"),
    ("custom-space-seam", ("custom", "task: search result | query:"), "  bar"),
    ("custom-special-seam", ("custom", "look <|ima"), "ge> now"),
]


def long_texts():
    """Four texts past 512 ids, built deterministically."""
    english = "\n\n".join(
        f"Paragraph {i}. The committee reviewed item {i} of the agenda at length, "
        f"weighing the costs of option {chr(65 + i % 26)} against its benefits. "
        f"Members noted that the {i}th revision introduced new constraints, and that "
        f"the timeline would slip by {i % 7 + 1} weeks if nothing changed. "
        f"A vote was deferred to the next session."
        for i in range(1, 25)
    )
    japanese = "".join(
        f"第{i}章では、市場の動向と技術の進歩について詳しく説明します。"
        f"特に{i % 5 + 1}つの重要な要素が挙げられ、それぞれが将来に影響を与えます。"
        for i in range(1, 30)
    )
    code = "\n".join(
        f"pub fn handler_{i}(input: &str) -> Result<usize, Error> {{\n"
        f"    let parsed = input.trim().parse::<usize>()?;\n"
        f"    Ok(parsed.saturating_mul({i}))\n}}\n"
        for i in range(1, 40)
    )
    unbroken = "token" * 600
    return [
        ("long-en", ("named", "document"), english),
        ("long-ja", ("named", "document"), japanese),
        ("long-code", BARE, code),
        ("long-unbroken", BARE, unbroken),
    ]


def fluiduse_cases(path):
    with open(path, encoding="utf-8") as f:
        cases = json.load(f)
    return [(f"fluiduse-{i:02}", BARE, case["text"], case["ids"]) for i, case in enumerate(cases)]


# Audio: committed 16 kHz mono clips, relative to coremlit/tests/. The jfk cut
# is the first 55,680 samples of jfk.wav (3.48 s, 87 tokens).
CLIPS = [
    ("jfk", "whisper/fixtures/audio/jfk.wav", None),
    ("jfk-3.48s", "whisper/fixtures/audio/jfk.wav", 55_680),
    ("sine440", "ced/fixtures/mel/sine440_10s.wav", None),
    ("noise", "ced/fixtures/mel/noise_10s.wav", None),
    ("silence", "ced/fixtures/mel/silence_2s.wav", None),
    ("ted60", "whisper/fixtures/audio/ted_60.wav", None),
]

# Images: siglip's six committed PNGs, relative to coremlit/tests/, and one
# synthetic 48 x 96 pattern.
IMAGES = [
    ("bus", "siglip/fixtures/goldens/images/bus.png"),
    ("cat", "siglip/fixtures/goldens/images/cat.png"),
    ("dog", "siglip/fixtures/goldens/images/dog.png"),
    ("mountain", "siglip/fixtures/goldens/images/mountain.png"),
    ("sailboat", "siglip/fixtures/goldens/images/sailboat.png"),
    ("sunflower", "siglip/fixtures/goldens/images/sunflower.png"),
    ("pattern-48x96", None),
]


def pattern_pixel(x, y, c):
    """The synthetic images' rule: deterministic, edge-rich, every channel
    distinct. Rust regenerates it from the same rule."""
    return (x * 7 + y * 13 + c * 101 + ((x // 5) ^ (y // 3)) * 29) % 256
