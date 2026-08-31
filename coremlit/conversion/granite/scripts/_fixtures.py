"""The committed granite golden corpus: 16 raw multilingual strings.

These are the EXACT texts (and ids) in
``$GRANITE_GOLDENS/corpus.json`` — the entry ids are the join key across
``corpus.json``, ``driver_crosscheck.json``, and every Rust granite gate, so an id
or an ordering change here silently invalidates all of them.

The strings are fed RAW. granite-embedding r2 retrieval is prompt-free
(``config_sentence_transformers.json`` carries empty query/document prompts), so
there is no task prefix to strip or add. Coverage is deliberate: Latin, CJK,
RTL, Cyrillic, hangul, emoji, mixed-script, code, digits/URLs, typographic
specials, a 4-token minimum, and one over-length input that truncates to
exactly 512 tokens.
"""
import os

CORPUS = [
    {
        "id": "en_q",
        "text": "how do I build a Rust CoreML inference library for text embeddings?",
        "note": "English question — the retrieval query side",
    },
    {
        "id": "en_doc",
        "text": (
            "Rust crates that wrap CoreML provide native on-device "
            "text-embedding inference."
        ),
        "note": "English passage — the retrieval document side",
    },
    {
        "id": "zh",
        "text": "北京是中华人民共和国的首都，也是重要的文化和政治中心。",
        "note": "Simplified Chinese (CJK, no whitespace segmentation)",
    },
    {
        "id": "ja",
        "text": "東京は日本の首都であり、世界最大級の都市圏を形成している。",
        "note": "Japanese (kana + kanji mix)",
    },
    {
        "id": "ar",
        "text": "باريس هي عاصمة فرنسا وأكبر مدنها من حيث عدد السكان.",
        "note": "Arabic — RIGHT-TO-LEFT script",
    },
    {
        "id": "ko",
        "text": "서울은 대한민국의 수도이자 최대 도시입니다.",
        "note": "Korean (hangul)",
    },
    {
        "id": "ru",
        "text": "Москва — столица России и крупнейший город страны.",
        "note": "Russian (Cyrillic)",
    },
    {
        "id": "es",
        "text": "La fotosíntesis convierte la luz solar en energía química en las plantas.",
        "note": "Spanish (Latin-1 diacritics)",
    },
    {
        "id": "de",
        "text": "Die Relativitätstheorie wurde von Albert Einstein entwickelt.",
        "note": "German (compound nouns, umlaut)",
    },
    {
        "id": "emoji",
        "text": "best pizza recipe 🍕🍅🧀 with a crispy crust and fresh basil 🌿",
        "note": "emoji — multi-codepoint / astral-plane bytes",
    },
    {
        "id": "mixed",
        "text": "使用 transformers 库加载 granite embedding model for retrieval 检索任务。",
        "note": "mixed-script: CJK + Latin in one string",
    },
    {
        "id": "code",
        "text": "def cosine(a, b): return (a @ b) / (norm(a) * norm(b))",
        "note": "source code — punctuation-dense, low natural-language signal",
    },
    {
        "id": "url_num",
        "text": "Order #A1234-99 shipped 2026-07-18; track at https://ex.com/t/A1234.",
        "note": "URL + order/date numerals — digit and delimiter handling",
    },
    {
        "id": "near512",
        "text": (
            "Quantum entanglement is a physical phenomenon that occurs when a "
            "group of particles are generated or interact in ways such that the "
            "quantum state of each particle cannot be described independently "
            "of the others, including when the particles are separated by a "
            "large distance. Quantum entanglement is a physical phenomenon that "
            "occurs when a group of particles are generated or interact in ways "
            "such that the quantum state of each particle cannot be described "
            "independently of the others, including when the particles are "
            "separated by a large distance. Quantum entanglement is a physical "
            "phenomenon that occurs when a group of particles are generated or "
            "interact in ways such that the quantum state of each particle "
            "cannot be described independently of the others, including when "
            "the particles are separated by a large distance. Quantum "
            "entanglement is a physical phenomenon that occurs when a group of "
            "particles are generated or interact in ways such that the quantum "
            "state of each particle cannot be described independently of the "
            "others, including when the particles are separated by a large "
            "distance. Quantum entanglement is a physical phenomenon that "
            "occurs when a group of particles are generated or interact in ways "
            "such that the quantum state of each particle cannot be described "
            "independently of the others, including when the particles are "
            "separated by a large distance. Quantum entanglement is a physical "
            "phenomenon that occurs when a group of particles are generated or "
            "interact in ways such that the quantum state of each particle "
            "cannot be described independently of the others, including when "
            "the particles are separated by a large distance. Quantum "
            "entanglement is a physical phenomenon that occurs when a group of "
            "particles are generated or interact in ways such that the quantum "
            "state of each particle cannot be described independently of the "
            "others, including when the particles are separated by a large "
            "distance. Quantum entanglement is a physical phenomenon that "
            "occurs when a group of particles are generated or interact in ways "
            "such that the quantum state of each particle cannot be described "
            "independently of the others, including when the particles are "
            "separated by a large distance. Quantum entanglement is a physical "
            "phenomenon that occurs when a group of particles are generated or "
            "interact in ways such that the quantum state of each particle "
            "cannot be described independently of the others, including when "
            "the particles are separated by a large distance. Quantum "
            "entanglement is a physical phenomenon that occurs when a group of "
            "particles are generated or interact in ways such that the quantum "
            "state of each particle cannot be described independently of the "
            "others, including when the particles are separated by a large "
            "distance. Quantum entanglement is a physical phenomenon that "
            "occurs when a group of particles are generated or interact in ways "
            "such that the quantum state of each particle cannot be described "
            "independently of the others, including when the particles are "
            "separated by a large distance. Quantum entanglement is a physical "
            "phenomenon that occurs when a group of particles are generated or "
            "interact in ways such that the quantum state of each particle "
            "cannot be described independently of the others, including when "
            "the particles are separated by a large distance. Quantum "
            "entanglement is a physical phenomenon that occurs when a group of "
            "particles are generated or interact in ways such that the quantum "
            "state of each particle cannot be described independently of the "
            "others, including when the particles are separated by a large "
            "distance. Quantum entanglement is a physical phenomenon that "
            "occurs when a group of particles are generated or interact in ways "
            "such that the quantum state of each particle cannot be described "
            "independently of the others, including when the particles are "
            "separated by a large distance. "
        ),
        "note": "OVER-LENGTH: 675 raw tokens, truncated to exactly 512 — the only entry "
                "that exercises the truncation path and a fully-populated window",
    },
    {
        "id": "short",
        "text": "hello world",
        "note": "minimal input — 4 tokens including the specials",
    },
    {
        "id": "special",
        "text": "Café — naïve façade; ½ + ¼ = ¾. \"Quotes\" & <tags> and\ttabs.",
        "note": "typographic specials: em dash, vulgar fractions, quotes, angle brackets, a literal TAB",
    },
]


def goldens_dir():
    """The committed goldens directory (``coremlit/tests/granite/fixtures/goldens``).

    Required — never defaulted to a repo-relative guess, so a recipe run can only ever
    write the goldens the caller pointed it at."""
    d = os.environ.get("GRANITE_GOLDENS")
    if not d:
        raise SystemExit("required environment variable GRANITE_GOLDENS is unset")
    return d
