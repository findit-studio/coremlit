# The licence row, the gate and the `MODELS_LOCK` table — written out before they could land, and what changed when they did

**Status: LANDED.** `FinDIT-Studio/facekit-coreml` is published (private) at revision
`70e212696bd3c472e28718e2e39c79467b97805e`, and every edit below is in the tree: the
`commercial-face-arcface` feature, the `#[cfg]`-gated `arcface` manifest module, the licence
row, the `MODELS_LOCK` table, the `ci.yml` shard and four gated suites. This file is kept as
the record of what was written out AHEAD of the revision and of the one field that had to
change when it arrived — see "What the landing changed" below. It is not a to-do list any
more.

The reason it was written ahead at all is unchanged and worth keeping: every edit needed the
artifact repository's **immutable revision**, which is a publishing action and not an edit,
and inventing one would have been the "record a version nobody ran" defect the rest of this
recipe is built to prevent.

## What the landing changed

**One field, and it was not the revision.** §4 below writes `kit = "face"`. That is wrong,
and the register says so mechanically:
`every_rows_loader_module_is_the_kit_its_lock_table_names` asserts that a row's `kit` EQUALS
the module its `loader` names, and §2's loader is `src/embeddings/face/mod.rs::arcface`. The
staged table is therefore `kit = "arcface"`, the `ci.yml` shard is `- kit: arcface`, and
`every_pin_locator_belongs_to_the_kit_and_bundle_it_is_read_for` then requires the row's pin
to live under a path component of that name — so the suite's shared module is
`tests/face/arcface/mod.rs` rather than `tests/face/common/mod.rs`. Naming the table `face`
would have pointed the row at the plain, UNGATED `face` feature, which is exactly the
reading direction 2 exists to refuse.

Three smaller things moved with it, none of them a correction:

* §1's SECOND sentence says the feature adds "a loader"; what it adds is a MANIFEST — four
  constants, no loading logic — so the shipped comment says so. The first sentence, which
  is the one the register's rule checks, is verbatim.
* §3's `pin` is `tests/face/arcface/mod.rs::ARTIFACT_SHA256`, a per-file manifest in
  `tests/identity`'s shape, rather than a scalar `ARCFACE_WEIGHTS_SHA256`. A manifest pin is
  what makes the row cover the whole bundle rather than the one file it keys on
  (`row_coverage`).
* §6's three suites became four: `model_io`, `parity`, `known_pairs` and `placement`. The
  parity suite needed one piece of new committed DATA — `tests/face/fixtures/
  onnx_reference.json`, cut by `scripts/write_onnx_reference.py` — for the reason §6 already
  gave, that a gate cannot depend on a live `onnxruntime`.

This is the same shape `conversion/redimnet/LICENCE_ROW.md` records for ReDimNet-B5, and the
coupling it describes is unchanged: a licence row whose `staged_by` names no `MODELS_LOCK`
table is a hard failure; a `MODELS_LOCK` table whose `kit` has no `ci.yml` shard is a hard
failure; and there is no sanctioned "not published yet" exemption.

**One thing here is NOT like ReDimNet, and it is the whole point of the file.** B5's row is
`Terms::unresolved` at the weights layer and `Terms::attribution` at the corpus layer, so it
needs no `commercial-` gate and `identity` stays a plain feature. This row is
**`Terms::research_only` at BOTH layers**. It is the register's first such row, and it is the
first artifact that turns directions 2 and 3 from mechanisms with nothing to bind into
mechanisms that bind something.

---

## 1 · The feature — `commercial-face-arcface`

```toml
# Requires a commercial licence for the ArcFace `w600k_r50` weights and their WebFace600K
# corpus. InsightFace publishes both for non-commercial research only, and offers no
# commercial grant over them, so what this feature adds is a loader for an artifact that may
# be used to develop, evaluate and test — never to ship. It is never in `default`, no other
# feature may pull it in, and this repository still redistributes no weight bytes: the
# artifact lives in a PRIVATE Hugging Face repository that CI fetches with a read token, the
# same posture `identity` already has (see NOTICE, "CI DOWNLOADS; IT DOES NOT
# REDISTRIBUTE"). The `commercial-` prefix reads backwards on purpose — it looks like
# "cleared for commercial use" — which is why the sentence you just read has to be the first
# one.
commercial-face-arcface = ["face"]
```

**The first sentence is written to the rule, not near it.** `model_licences.rs`'s
`every_commercial_feature_says_it_requires_a_commercial_licence_first` normalises the block's
first sentence (`normalise_spelling`: lowercase, `licence` → `license`), requires it to
**begin with** one of `COMMERCIAL_DOC_OPENINGS`, and then refuses it if `negation_in` finds
any word of `NEGATIONS` anywhere in that same sentence. Checked against the sentence above:

| check | result |
|---|---|
| normalised first sentence | `requires a commercial license for the arcface \`w600k_r50\` weights and their webface600k corpus.` |
| begins with a `COMMERCIAL_DOC_OPENINGS` entry | yes — `requires a commercial license` |
| `negation_in` over its words | none of `no not never neither nor none without unless cannot cant dont doesnt isnt wont except` appears |

The sentence that carries the "InsightFace offers no commercial grant" fact is the SECOND
one, and that is not stylistic: `no` in the first sentence would red the rule, correctly —
a first sentence that warns and then qualifies has not warned anybody.

### `face` stays plain, and this is a NEW feature rather than the rename `FEATURE_MAP.md` predicts

`coremlit/FEATURE_MAP.md` currently says of `face`: *"the feature is plain `face` … and
`commercial-face` … arrives with the artifact it protects, not before."* Read literally that
is a rename, and **a rename would be wrong.** Two thirds of `face` is not encumbered by
anything:

* `embeddings::face::align` is `f64` arithmetic and an integer resampler with a synthetic
  golden. It contains no weights and no photograph, and its correctness matters to any face
  model at all;
* `embeddings::face::embed` is a manifest-driven CoreML embedder over a **caller-supplied**
  path. It loads whatever it is pointed at, including a model whose licence permits a
  product.

Pushing either behind a gate whose documentation says "requires a commercial licence" would
make the register say something false about code that requires nothing. What is encumbered
is exactly the **staged-artifact loader** for these weights, so that is what the new feature
gates. `FEATURE_MAP.md`'s `face` paragraph needs rewriting in the same change.

## 2 · The `#[cfg]` in `src/` — a gate that gates nothing is a name

`every_commercial_feature_gates_an_artifact_with_no_shipping_grant` refuses a `commercial-`
feature that **no `#[cfg(feature = ...)]` under `src/` names**, and
`every_rows_gate_matches_the_cfg_that_guards_its_loader` derives the row's real gate from the
`#[cfg]` chain on the `mod` the row's `loader` field names — not from the row's own claim. So
the row below is only true if this declaration exists in `coremlit/src/embeddings/face/mod.rs`:

```rust
/// The staged `w600k_r50` artifact: a [`FaceModel`] and the path CI downloads it to.
///
/// Behind `commercial-face-arcface` because the weights and their corpus are
/// research-only. Everything else in this module works on any artifact the
/// caller supplies and is not gated.
#[cfg(feature = "commercial-face-arcface")]
pub mod arcface;
```

and `arcface.rs` holds the loader — the `FaceModel { dim: 512, preprocessing:
Preprocessing::ARCFACE }` this recipe measured, plus the `Models/facekit/w600k_r50.mlmodelc`
path the shard stages. `loader_gates` reads the `#[cfg]` **as a cfg expression**, so the
positive form above is what derives the gate; a `#[cfg(not(feature = …))]` would be read as
that feature's gate while compiling precisely when it is off (a known defect of the reader,
recorded in issue #115's own "known gaps" comment), so the positive form is the only one to
write.

## 3 · The licence row

Paste into `ARTIFACTS` in `coremlit/tests/model_licences.rs`. It needs one new restrictions
constant beside `RETAIN_NOTICE` and friends:

```rust
/// Terms that confine use to non-commercial research and grant no
/// redistribution. Not an SPDX identifier and deliberately not rounded to one:
/// InsightFace's zoo and the WebFace260M agreement are two separate documents
/// that happen to impose the same two obligations, and neither is a licence
/// anybody can look up by name.
const RESEARCH_ONLY_NO_REDISTRIBUTION: &[&str] = &[
  "non-commercial-research-use-only",
  "no-commercial-use",
  "no-redistribution-of-the-weights",
];
```

```rust
  // --- face (commercial-face-arcface) ---------------------------------------
  Artifact {
    file: "facekit/w600k_r50.mlmodelc/weights/weight.bin",
    // MEASURED on the conversion run recorded in conversion/face/README.md's observed-
    // toolchain table. It must be RE-READ from the PUBLISHED artifact before this row
    // lands: the .mlmodelc is produced by `xcrun coremlcompiler`, whose output the Python
    // toolchain does not pin. The ReDimNet recipe measured which files that affects —
    // `model.mil`, `weights/weight.bin` and `metadata.json` re-derive byte for byte, while
    // both `coremldata.bin`s differ between two compiles of the SAME .mlpackage — so this
    // key is on the deterministic half, and the publishing run's bytes are still the ones a
    // pin may name.
    key: Key::Sha256("aa08d7826a70f9bc237ea0532a5eec12cb83b8375148a1b0650f104cbb2ff492"),
    pin: "tests/face/common/mod.rs::ARCFACE_WEIGHTS_SHA256",  // does not exist yet — step 4
    staged_by: "FinDIT-Studio/facekit-coreml",                // repository not published yet
    loader: "src/embeddings/face/mod.rs::arcface",            // module not written yet — §2
    gate: "commercial-face-arcface",
    weights: Terms::research_only(
      "InsightFace model licence (non-commercial research)",
      RESEARCH_ONLY_NO_REDISTRIBUTION,
      "InsightFace's model zoo states \"ALL models are available for non-commercial research \
       purposes only\", and `buffalo_l` is one of its packaged models. No commercial licence \
       is offered for these weights — issue #115's census could not find one to buy, and the \
       owner's decision was to use them for CI and development on the standing basis that \
       this repository redistributes nothing. A conversion does not lift the restriction: \
       re-encoding a graph produces a derivative of the weights, not a new work, so this \
       bundle carries their terms exactly. This is the register's FIRST research-only weights \
       layer; every earlier restricted layer was a corpus one.",
    ),
    corpus: Terms::research_only(
      "WebFace260M/WebFace600K licence agreement",
      RESEARCH_ONLY_NO_REDISTRIBUTION,
      "`w600k_r50` is trained on WebFace600K, the 600K-identity subset of WebFace260M, which \
       is released under a signed licence agreement confining it to non-commercial academic \
       research. The corpus layer would disqualify the shipping path on its own even if a \
       commercial grant over the weights appeared, which is exactly why this register asks \
       the two questions separately — and it is the reason issue #115's census ended with no \
       shippable candidate at all rather than with a purchase order.",
    ),
    source: "conversion/face/README.md; InsightFace model zoo (\"non-commercial research \
             purposes only\"); WebFace260M licence agreement",
  },
```

### What this row turns on, checked mechanism by mechanism

| check | before this row | with this row |
|---|---|---|
| `Terms::forbids_commercial_use` | false for every row in the register | **true**, for the first time — both layers |
| direction 2, `research_only_reachable` | binds nothing (no research-only row exists) | binds: `commercial-face-arcface` must be absent from `default`'s closure **and** from every other feature's closure |
| direction 3, `every_commercial_feature_gates_an_artifact_with_no_shipping_grant` | binds nothing (no `commercial-` feature exists) | binds: the gate stands on a found prohibition rather than an open question |
| `every_commercial_feature_says_it_requires_a_commercial_licence_first` | vacuous | binds §1's first sentence |
| `no_ungranted_artifact_is_reachable_from_default` | already binds `redimnet_b5` (unresolved) | binds this row too, for a stronger reason |

The module doc of `model_licences.rs` currently says **"No row here is research-only"** and
explains why. That sentence has to be rewritten in the same change; it is load-bearing prose,
not a header.

## 4 · The `MODELS_LOCK` table

```toml
# The InsightFace `w600k_r50` face-embedding artifact (`embeddings::face`, issue #115). One
# bundle, 83 MB fp16, converted by `coremlit/conversion/face` from the OFFICIAL public
# `deepinsight/insightface` v0.7 release asset `buffalo_l.zip` (sha256 80ffe37d…), member
# `w600k_r50.onnx` (sha256 4c06341c…, IResNet-50 trained on WebFace600K).
#
# The graph is `data [1, 3, 112, 112] f32 -> embedding [1, 512] f32`, RAW (un-normalised) —
# the ONNX ends `BatchNorm -> Flatten -> Gemm -> BatchNorm` with no L2 anywhere in it, so the
# DOOR normalises. Preprocessing is RGB, NCHW, `(x - 127.5) / 127.5`, read off InsightFace's
# own `ArcFaceONNX` and then measured: feeding BGR drops the worst same-person pair through
# InsightFace's own 0.28 line. Both facts are in the recipe's README.
#
# RESEARCH-ONLY AT BOTH LAYERS, and the ONLY table here that is. The weights are
# InsightFace's non-commercial research terms and the corpus is WebFace600K. It is staged
# behind `commercial-face-arcface`, a feature that is never in `default` — see
# tests/model_licences.rs. THIS REPOSITORY IS PRIVATE for a stronger reason than
# redimnetkit's: there, no grant covers redistribution; here, a document affirmatively
# forbids it. CI fetching from our own private repository is USE, which is the line NOTICE
# already draws, and the shard therefore needs a Hugging Face read token.
#
# The revision is the ARTIFACT repo's commit SHA. Do not confuse it with the upstream SOURCE
# pins — the pack's sha256 and the converted member's — which name what the CONVERSION
# consumed and live in conversion/face/scripts/_arcface_common.py.
["FinDIT-Studio/facekit-coreml"]
kit       = "face"        # SUPERSEDED: the staged table is `kit = "arcface"` — see "What the landing changed"
include   = "w600k_r50.mlmodelc/* CHECKSUMS.sha256"
revision  = "<artifact repo commit SHA — repository not published yet>"   # 70e21269…7805e
local-dir = "Models/facekit"
```

The bundle ships `CHECKSUMS.sha256` and `MANIFEST.json`, both emitted by
`scripts/write_manifest.py`, with paths relative to the **kit root** (`./w600k_r50.mlmodelc/…`)
exactly as redimnetkit's second revision and speakerkit's are, so `shasum -c` works from the
kit root with no filter and the kit does **not** belong in `CHECKSUMLESS_KITS`.

## 5 · The `ci.yml` shard

`MODELS_LOCK`'s kits and `ci.yml`'s `model-tests` shards must be the same set
(`tests/whisper/models_lock.rs`), so the kit needs its row (`kit: arcface` as landed, not
`face` — see "What the landing changed"):

```yaml
          - kit: face
            cache: Models/facekit
            probe: Models/facekit/w600k_r50.mlmodelc
            checksum-dir: Models/facekit
            checksum-file: CHECKSUMS.sha256
            fp16-vendors: vadkit,facekit
            gates: |
              commercial-face-arcface|face_model_io face_parity_embed face_known_pairs
```

`checksum-dir` is the kit ROOT here, not the bundle, because the manifest's paths are
kit-root-relative — the same reason `identity`'s row uses `../CHECKSUMS.sha256` from inside
the bundle and this one does not need to. The shard needs the private-repo read token the
`identity` shard already documents.

## 6 · The gated tests

Three suites, `#[ignore]`d like every other model gate, reading the fixtures this branch
already commits:

| suite | what it pins |
|---|---|
| `tests/face/model_io.rs` | the load contract off the staged bundle: `data [1,3,112,112]` f32 `Fixed`, `embedding [1,512]` f32, no state, `batch_capacity() == 1`, and the artifact digest |
| `tests/face/parity_embed.rs` | committed fp32 goldens for the 18 fixture crops, cosine ≥ 0.99 per compute arm — the numbers in the recipe README's parity table |
| `tests/face/known_pairs.rs` | min same-person ≥ 0.28, max different-person < 0.20 over the 18 crops, at InsightFace's own operating point |

The parity goldens are the one piece of new DATA those need: `verify_arcface.py` measures
against a live `onnxruntime`, and a gate cannot depend on `ort` (the `face` feature pulls no
ONNX runtime). Cut them from the same run, the way `granite` and `siglip` commit
transformers-fp32 goldens.

## The order this had to land in — all four steps are done

1. **Publish** the tree in `conversion/face`'s output root to a PRIVATE
   `FinDIT-Studio/facekit-coreml`, and read back its commit SHA and the published
   `w600k_r50.mlmodelc/weights/weight.bin` SHA-256. Nothing before this step is possible;
   nothing after it is blocked.
2. Write `src/embeddings/face/arcface.rs` and its `#[cfg]`-gated `mod` declaration (§2), the
   `commercial-face-arcface` feature (§1), and `tests/face/common/mod.rs` with the pin
   `const`s. This is what creates `loader`, `gate` and `pin`.
3. Land the `MODELS_LOCK` table (§4), the `ci.yml` shard (§5) and the licence row (§3) **in
   one change** — until the revision exists the row names a bundle nothing stages, and until
   the shard exists the table has no downloader. Rewrite `model_licences.rs`'s "No row here
   is research-only" module doc and `FEATURE_MAP.md`'s `face` paragraph in the same change.
4. Add the gated tests (§6) and the curated feature-combination row
   (`face` alone is already listed; `commercial-face-arcface` needs its own).
