//! **Which model artifacts a closed-source product may ship, enforced rather
//! than remembered.**
//!
//! coremlit is MIT OR Apache-2.0, but the products built on it are not
//! necessarily open source. A model whose WEIGHTS or whose TRAINING CORPUS
//! forbids commercial use is therefore disqualifying for the shipping path,
//! while still being perfectly legal for CI to fetch and test against: this
//! repository redistributes no weight bytes at all (`NOTICE`'s "CI DOWNLOADS;
//! IT DOES NOT REDISTRIBUTE", and `MODELS_LOCK` stages everything into a
//! gitignored `Models/` for the duration of a job). Those are two different
//! permissions, and only the first one gates a feature.
//!
//! # The table is keyed by ARTIFACT + SHA-256, never by repository
//!
//! This is the whole lesson of the investigation that produced this file.
//! `fal/AuraFace-v1` is tagged `apache-2.0` on Hugging Face while four of its
//! five ONNX files are byte-identical to InsightFace artifacts distributed for
//! "non-commercial research purposes only". A repo-keyed table gets four rows
//! wrong and reads clean while doing it. So [`Artifact::file`] plus
//! [`Artifact::key`] is the identity, the repository is only where the bytes
//! were fetched from, and
//! [`identical_bytes_carry_identical_terms`] refuses the exact shape of that
//! failure: two rows over the same SHA-256 that disagree about what the bytes
//! permit.
//!
//! # The three directions
//!
//! Modelled on `CHECKSUMLESS_KITS` in `tests/whisper/models_lock.rs`, which
//! this repository already uses precisely so an exemption cannot outlive its
//! cause. Red on all three of:
//!
//!   1. a `MODELS_LOCK` artifact with no licence row — and the reverse, a row
//!      naming a repository no table stages
//!      ([`every_staged_repo_has_a_licence_row_and_every_row_names_a_staged_repo`]);
//!   2. a research-only row reachable from `default`, or gated by a feature
//!      that is not `commercial-` prefixed
//!      ([`no_research_only_artifact_is_reachable_without_a_commercial_gate`]);
//!   3. a `commercial-`prefixed feature gating an artifact whose row is
//!      actually clear
//!      ([`every_commercial_feature_gates_a_research_only_artifact`]).
//!
//! The third is the one people forget, and it is what keeps the table honest
//! as artifacts change: the day an upstream relicenses, the gate that was
//! protecting it becomes a gate protecting nothing, and it must be retired
//! rather than left standing as false reassurance.
//!
//! # What a check that cannot fire proves
//!
//! **No row in the seeded table is research-only today, and no
//! `commercial-`prefixed feature exists.** Directions 2 and 3 therefore cannot
//! fire against the real repository right now — they are tripwires for the
//! artifact that has not been added yet. A tripwire nobody has seen trip is not
//! a tripwire, so every predicate below is ALSO driven by hermetic falsifiers
//! over doctored input (`falsifiers::*`), which run everywhere, need no models
//! and no repository files, and fail if the predicate ever stops detecting the
//! thing it exists to detect. Direction 1, the SHA-256 pin cross-check and the
//! same-bytes-same-terms rule DO bind live data today.
//!
//! # What this file can and cannot see
//!
//! `MODELS_LOCK` names repositories, selectors and revisions — not files. The
//! file list only exists after a download, and these checks are hermetic. So
//! direction 1 binds at TABLE granularity ("every table is covered by at least
//! one row"), while the ROWS are per-file, which is what makes the AuraFace
//! failure mode expressible at all. The seeding rule for rows is: one row per
//! staged file this repository independently pins by SHA-256, plus the
//! `revision = "main"` whisper artifacts that nothing can pin. Files inside a
//! staged bundle that carry no independent pin (`Segmentation.mlmodelc`,
//! `PLDA.mlmodelc` and the other FluidInference bundles) are not rows, and a
//! research-only artifact could in principle hide in one; that is a real,
//! named gap, and closing it needs a per-file manifest those bundles do not
//! have.
//!
//! Hermetic: pure file reads, no network, no models, no feature needs
//! enabling. The `MODELS_LOCK`-reading checks SKIP outside the repository
//! workspace (the published tarball packages no lock file), exactly as
//! `tests/whisper/models_lock.rs` does; the falsifiers never skip.

// The workspace-root anchor, FOUND by searching upward for the `[workspace]`
// manifest rather than counted in `../` hops — see its module doc.
#[path = "support/workspace_root.rs"]
#[allow(dead_code)]
mod workspace_root;

use std::{
  collections::{BTreeMap, BTreeSet},
  path::Path,
};

// ---------------------------------------------------------------------------
// The table
// ---------------------------------------------------------------------------

/// What one licence layer permits, and the reading this repository has on it.
///
/// The payload is prose on purpose: a bare SPDX identifier answers "which
/// licence" and never "and does that let us ship it", which is the only
/// question this file is asking. Every variant is a NEWTYPE of exactly one
/// payload — the workspace house rule
/// (`no_enum_in_the_workspace_has_a_struct_shaped_or_multi_field_variant`).
#[derive(Debug, Clone, Copy)]
enum Terms {
  /// Commercial use permitted with no condition beyond retaining notices —
  /// MIT, Apache-2.0, BSD. Payload: the licence and where it is declared.
  Permissive(&'static str),
  /// Commercial use permitted, but attribution is a CONDITION of it, so
  /// shipping without the notice is infringement rather than impoliteness.
  /// Payload: the licence, and what has to be reproduced.
  Attribution(&'static str),
  /// **Disqualifying.** Forbids commercial use. Payload: the exact
  /// restriction and where it is stated.
  ResearchOnly(&'static str),
  /// Not established. Payload: the open QUESTION and where to go to answer it.
  ///
  /// Deliberately distinct from [`Terms::Permissive`]: rounding an unknown to
  /// "clear" is how a table stops being evidence. Unresolved is not
  /// disqualifying either — it is a row that no shipping claim may rest on
  /// until somebody resolves it.
  Unresolved(&'static str),
}

impl Terms {
  /// The verdict, without the prose — what two rows over identical bytes must
  /// agree on.
  const fn verdict(self) -> &'static str {
    match self {
      Self::Permissive(_) => "permissive",
      Self::Attribution(_) => "attribution-required",
      Self::ResearchOnly(_) => "research-only",
      Self::Unresolved(_) => "unresolved",
    }
  }

  /// Whether these terms forbid the shipping path outright.
  const fn forbids_commercial_use(self) -> bool {
    matches!(self, Self::ResearchOnly(_))
  }

  /// The prose payload.
  const fn detail(self) -> &'static str {
    match self {
      Self::Permissive(d) | Self::Attribution(d) | Self::ResearchOnly(d) | Self::Unresolved(d) => d,
    }
  }
}

/// How a row addresses its bytes.
enum Key {
  /// The file's SHA-256, lowercase hex.
  Sha256(&'static str),
  /// No immutable byte identity exists for this artifact. Payload: why.
  ///
  /// Legal ONLY where the row's `MODELS_LOCK` table is still on
  /// `revision = "main"` — a moving target, so there is no single set of bytes
  /// to key on. Tied to that cause in both directions by
  /// [`unpinned_rows_exist_only_where_the_lock_pins_a_moving_revision`]: the
  /// day the LOUD FOLLOW-UP in `MODELS_LOCK` lands and those tables pin a
  /// commit, this exemption goes red and has to be replaced by a hash.
  Unpinned(&'static str),
}

/// One staged file, and what its bytes permit.
struct Artifact {
  /// Path under `Models/`, exactly as a `model-tests` shard stages it.
  file: &'static str,
  /// The bytes' identity — see [`Key`].
  key: Key,
  /// `<crate-relative source>::<identifier>` where this repository ALREADY
  /// pins those bytes, or `""` for a [`Key::Unpinned`] row.
  ///
  /// The licence attaches to bytes, so a hash copied here and never checked
  /// again is a hash that goes stale the first time an artifact is
  /// re-converted. This field is what stops that: the identifier names a
  /// `const` or a `fn` in the tree, and the SHA-256 in that pin has to be the
  /// one in this row.
  pin: &'static str,
  /// The `MODELS_LOCK` table that stages the file.
  staged_by: &'static str,
  /// The cargo feature a caller must enable before the shipping path can load
  /// it. Research-only artifacts must be gated by a `commercial-` feature; see
  /// [`COMMERCIAL_PREFIX`].
  gate: &'static str,
  /// Terms on the weight bytes themselves.
  weights: Terms,
  /// Terms on the data the weights were TRAINED ON.
  ///
  /// A different question from [`Artifact::weights`], and the one nearly every
  /// model fails: Apache-2.0 weights trained on a corpus licensed for research
  /// only are still research only. `NOTICE` records the weights layer for every
  /// component and the corpus layer for none, which is why so many rows below
  /// are [`Terms::Unresolved`] here.
  corpus: Terms,
  /// Where the two verdicts above come from.
  source: &'static str,
}

impl Artifact {
  /// Which layer disqualifies the artifact, when one does.
  fn disqualifying_layer(&self) -> Option<&'static str> {
    if self.weights.forbids_commercial_use() {
      Some("weights")
    } else if self.corpus.forbids_commercial_use() {
      Some("training corpus")
    } else {
      None
    }
  }

  /// The terms of the layer named by [`Self::disqualifying_layer`].
  fn disqualifying_terms(&self) -> Option<Terms> {
    if self.weights.forbids_commercial_use() {
      Some(self.weights)
    } else if self.corpus.forbids_commercial_use() {
      Some(self.corpus)
    } else {
      None
    }
  }

  /// The row's SHA-256, or `None` when it is [`Key::Unpinned`].
  const fn sha256(&self) -> Option<&'static str> {
    match self.key {
      Key::Sha256(hex) => Some(hex),
      Key::Unpinned(_) => None,
    }
  }
}

/// The prefix that marks a feature as gating artifacts a commercial licence is
/// needed for.
///
/// Chosen over the alternatives by the owner, and it can be READ BACKWARDS —
/// `commercial-face` looks like "cleared for commercial use" to anyone who has
/// not read this file. That is why
/// [`every_commercial_feature_says_it_requires_a_commercial_licence_first`]
/// exists and why it checks the FIRST sentence: the correction has to arrive
/// before the misreading has time to settle.
const COMMERCIAL_PREFIX: &str = "commercial-";

/// The phrase every `commercial-` feature's documentation must open with,
/// normalised (see [`normalise_spelling`]).
const COMMERCIAL_DOC_PHRASE: &str = "requires a commercial license";

/// Every artifact `MODELS_LOCK` stages that this repository pins by SHA-256,
/// plus the whisper artifacts nothing can pin, and what each one permits.
///
/// Seeded from what the repository actually stages today. Every SHA-256 is
/// copied from the pin named in the same row's [`Artifact::pin`] and checked
/// against it by [`every_rows_sha256_matches_the_pin_it_names`], so the two
/// cannot drift apart.
///
/// **No row here is research-only.** Every disqualification found so far sits
/// in `Terms::Unresolved` on the CORPUS layer, because `NOTICE` documents the
/// weights layer throughout and the corpus layer nowhere. That is a finding
/// about this repository's records, not a clean bill of health.
const ARTIFACTS: &[Artifact] = &[
  // --- whisper -------------------------------------------------------------
  Artifact {
    file: "whisperkit-coreml/openai_whisper-tiny/AudioEncoder.mlmodelc/weights/weight.bin",
    key: Key::Unpinned(
      "`argmaxinc/whisperkit-coreml` is still on `revision = \"main\"` (MODELS_LOCK's LOUD \
       FOLLOW-UP), so no immutable byte identity exists to key on; the same reason puts the \
       `whisper` kit in CHECKSUMLESS_KITS.",
    ),
    pin: "",
    staged_by: "argmaxinc/whisperkit-coreml",
    gate: "whisper",
    weights: Terms::Permissive(
      "MIT. WhisperKit's CoreML conversion (argmaxinc/WhisperKit, MIT) of OpenAI Whisper (MIT).",
    ),
    corpus: Terms::Unresolved(
      "OpenAI has published no terms for the ~680 000 hours of web audio Whisper was trained on, \
       and it does not name the sources. NOTICE section 3 records the weights only. Resolve \
       against openai/whisper's model card and paper before any shipping claim rests on the \
       corpus layer.",
    ),
    source: "NOTICE section 3",
  },
  Artifact {
    file: "whisperkit-coreml/openai_whisper-tiny/TextDecoder.mlmodelc/weights/weight.bin",
    key: Key::Unpinned(
      "`argmaxinc/whisperkit-coreml` is still on `revision = \"main\"` (MODELS_LOCK's LOUD \
       FOLLOW-UP), so no immutable byte identity exists to key on; the same reason puts the \
       `whisper` kit in CHECKSUMLESS_KITS.",
    ),
    pin: "",
    staged_by: "argmaxinc/whisperkit-coreml",
    gate: "whisper",
    weights: Terms::Permissive(
      "MIT. WhisperKit's CoreML conversion (argmaxinc/WhisperKit, MIT) of OpenAI Whisper (MIT).",
    ),
    corpus: Terms::Unresolved(
      "Same undisclosed ~680 000-hour web-audio corpus as the encoder; see that row.",
    ),
    source: "NOTICE section 3",
  },
  Artifact {
    file: "tokenizers/whisper-tiny/tokenizer.json",
    key: Key::Unpinned(
      "`openai/whisper-tiny` is still on `revision = \"main\"` (MODELS_LOCK's LOUD FOLLOW-UP), so \
       no immutable byte identity exists to key on.",
    ),
    pin: "",
    staged_by: "openai/whisper-tiny",
    gate: "whisper",
    weights: Terms::Permissive("MIT — OpenAI's own tokenizer artifact for whisper-tiny."),
    corpus: Terms::Unresolved(
      "The BPE vocabulary was fit on the same undisclosed corpus as the weights, so the corpus \
       layer is open for the same reason. A vocabulary carries no weights, which narrows the \
       exposure but does not close the question.",
    ),
    source: "NOTICE section 3",
  },
  // --- granite -------------------------------------------------------------
  Artifact {
    file: "embedkit-granite/granite-97m-multilingual-r2/granite_97m_512.mlmodelc/weights/weight.bin",
    key: Key::Sha256("276bc93c49a4f37ffefdfb2e10f7d7e1ef57db9027c7ad0d3f2e4160f81a79be"),
    pin: "tests/granite/model_io.rs::ARTIFACT_SHA256",
    staged_by: "FinDIT-Studio/embedkit-coreml",
    gate: "granite",
    weights: Terms::Permissive(
      "Apache-2.0. ibm-granite/granite-embedding-97m-multilingual-r2; the staged file is a format \
       conversion with unchanged weight VALUES, so the upstream terms govern.",
    ),
    corpus: Terms::Unresolved(
      "IBM describes the Granite embedding training mixture in the model card but does not state \
       per-source licences for it, and NOTICE section 7a records the weights layer only. Resolve \
       against ibm-granite/granite-embedding-97m-multilingual-r2's data statement.",
    ),
    source: "NOTICE section 7a",
  },
  Artifact {
    file: "embedkit-granite/granite-97m-multilingual-r2/tokenizer.json",
    key: Key::Sha256("4f2842d568e2724370aec203652a42ac783c7937f8347a1a2cc7506d71f1582f"),
    pin: "src/embeddings/granite/mod.rs::TOKENIZER_SHA256_HEX",
    staged_by: "FinDIT-Studio/embedkit-coreml",
    gate: "granite",
    weights: Terms::Permissive(
      "Apache-2.0, the same terms as the model it indexes. Distributed WITH the artifact and read \
       from disk, so whoever redistributes the artifact directory redistributes it.",
    ),
    corpus: Terms::Unresolved("Same open question as the granite weights row."),
    source: "NOTICE section 7b",
  },
  // --- siglip --------------------------------------------------------------
  Artifact {
    file: "siglip2-naflex/siglip2-base-patch16-naflex-512/siglip2_vision_512.mlmodelc/weights/\
           weight.bin",
    key: Key::Sha256("31fc44e771553c5b28b7af6561b46650ce5e1e4711dfef9f471ed32d502077b6"),
    pin: "tests/siglip/model_io.rs::ARTIFACT_SHA256",
    staged_by: "FinDIT-Studio/siglip2-naflex-coreml",
    gate: "siglip",
    weights: Terms::Permissive(
      "Apache-2.0. google/siglip2-base-patch16-naflex; the artifact repo declares apache-2.0 too. \
       The graph is RESTRUCTURED (the position-embedding resize is lifted host-side), which \
       Apache-2.0 permits with the change stated — NOTICE section 8a states it.",
    ),
    corpus: Terms::Unresolved(
      "SigLIP 2 is trained on WebLI, which Google has not released and whose terms are not \
       stated. NOTICE section 8a records the weights layer only. This one cannot be resolved from \
       public material alone.",
    ),
    source: "NOTICE section 8a",
  },
  Artifact {
    file: "siglip2-naflex/siglip2-base-patch16-naflex-512/siglip2_text_64.mlmodelc/weights/\
           weight.bin",
    key: Key::Sha256("8b781500cc6a596fa3a27b16b56e3d81e675e642ecd3542722d1f185aa0a6f67"),
    pin: "tests/siglip/text_model_io.rs::ARTIFACT_SHA256",
    staged_by: "FinDIT-Studio/siglip2-naflex-coreml",
    gate: "siglip",
    weights: Terms::Permissive("Apache-2.0; the vision tower's twin, same checkpoint."),
    corpus: Terms::Unresolved("Same unreleased WebLI corpus as the vision tower; see that row."),
    source: "NOTICE section 8a",
  },
  Artifact {
    file: "siglip2-naflex/siglip2-base-patch16-naflex-512/tokenizer.json",
    key: Key::Sha256("58a1696e79c9d97937389ed116f552a15c84811d7b8023918b86f4bc5775b1b0"),
    pin: "src/embeddings/siglip/text/mod.rs::TOKENIZER_SHA256_HEX",
    staged_by: "FinDIT-Studio/siglip2-naflex-coreml",
    gate: "siglip",
    weights: Terms::Permissive(
      "Apache-2.0, the same terms as the model. The Gemma tokenizer as packaged with the SigLIP 2 \
       checkpoint; distributed WITH the artifact, not compiled into the crate.",
    ),
    corpus: Terms::Unresolved("Same unreleased WebLI corpus as the weights rows."),
    source: "NOTICE section 8b",
  },
  // --- ced -----------------------------------------------------------------
  Artifact {
    file: "ced/ced-tiny/ced_tiny.mlmodelc/weights/weight.bin",
    key: Key::Sha256("5635cd9f932583105d1bf40bd07eb54e3f715a70d8319923cd0617a1dea3db01"),
    pin: "tests/ced/model_io.rs::TINY_SHA256",
    staged_by: "FinDIT-Studio/cedkit-coreml",
    gate: "ced",
    weights: Terms::Permissive(
      "Apache-2.0. mispeech/ced-tiny (Xiaomi); the CoreML graph is restructured from unchanged \
       weight values, and NOTICE section 9 states the changes.",
    ),
    corpus: Terms::Unresolved(
      "CED is distilled on AudioSet. The AudioSet ONTOLOGY and label set are CC-BY-4.0, but the \
       segments themselves are YouTube audio Google never redistributed, and the derived-weights \
       question is not addressed by either. NOTICE section 9 records the weights layer only.",
    ),
    source: "NOTICE section 9",
  },
  // --- speaker: the FluidInference base layer ------------------------------
  Artifact {
    file: "speakerkit/wespeaker_v2.mlmodelc/weights/weight.bin",
    key: Key::Sha256("34004f6798d35cad7071e2fdc67e63faaa782f53697e1cb49bcb452cf81ae151"),
    pin: "tests/speaker/model_io.rs::int8_wespeaker_matches_fluidinference_pinned_sha256",
    staged_by: "FluidInference/speaker-diarization-coreml",
    gate: "speaker",
    weights: Terms::Unresolved(
      "The RETIRED int8 WeSpeaker embedder, kept for tests. NOTICE section 4 gives the \
       FluidInference repo as \"SDK Apache-2.0; parent pyannote model cc-by-4.0\", but the \
       WeSpeaker embedder itself it records only as \"see its model license \
       (https://github.com/wenet-e2e/wespeaker)\" — it names no licence. Resolve against the \
       wenet-e2e/wespeaker model licence before treating this as clear.",
    ),
    corpus: Terms::Unresolved(
      "WeSpeaker's published embedders are trained on VoxCeleb. VoxCeleb's own terms are the \
       thing to check, and they are the specific reason this row is not rounded to clear — a \
       corpus restricted to non-commercial research would make the DERIVED weights research-only \
       whatever the weights layer says. Not stated anywhere in this repository.",
    ),
    source: "NOTICE section 4",
  },
  // --- speaker: the FinDIT-Studio overlay, the two SHIPPING artifacts ------
  Artifact {
    file: "speakerkit/pyannote_segmentation.mlmodelc/weights/weight.bin",
    key: Key::Sha256("0266f4ad4d843ecf31ef9220ad6b80616b3ec64a4404b64f3ea0371554e236ec"),
    pin: "tests/speaker/model_io.rs::fp16_safe_segmentation_matches_pinned_sha256",
    staged_by: "FinDIT-Studio/speakerkit-coreml",
    gate: "speaker",
    weights: Terms::Permissive(
      "MIT. An issue-#15 re-conversion of pyannote/segmentation-3.0 (MIT) with fp16-survivable \
       guards; the artifact repo declares HF licence \"other\"/mixed-upstream, and NOTICE section \
       4 records that the upstream MIT terms still govern because the weight values are the \
       upstream ones.",
    ),
    corpus: Terms::Unresolved(
      "pyannote/segmentation-3.0 is trained on a mixture (AMI, DIHARD, VoxConverse and others) \
       whose members carry different terms, several of them research-only. NOTICE section 4 \
       records the weights layer only. This is the row most likely to become research-only once \
       somebody resolves it.",
    ),
    source: "NOTICE section 4",
  },
  Artifact {
    file: "speakerkit/wespeaker.mlmodelc/weights/weight.bin",
    key: Key::Sha256("680837ec172d67c3197bba93800e1623eebfd35c3b17011802f5f98b8026a0aa"),
    pin: "tests/speaker/model_io.rs::fp16_safe_wespeaker_fp32_matches_pinned_sha256",
    staged_by: "FinDIT-Studio/speakerkit-coreml",
    gate: "speaker",
    weights: Terms::Unresolved(
      "The SHIPPING fp32 WeSpeaker embedder. NOTICE section 4 records the WeSpeaker embedder as \
       \"see its model license (https://github.com/wenet-e2e/wespeaker)\" and names none; the \
       artifact repo declares HF licence \"other\"/mixed-upstream. The CC-BY-4.0 in that section \
       belongs to pyannote/speaker-diarization-community-1 (the PLDA `diaric` clusters through) \
       and to FluidInference's parent pyannote model — NOT to these embedder weights, and \
       reading it across is the mistake this row exists to stop.",
    ),
    corpus: Terms::Unresolved(
      "VoxCeleb, per the WeSpeaker toolkit's published recipes; terms not stated in this \
       repository. Same open question as the retired int8 sibling — and the same reason it \
       matters more than the weights layer.",
    ),
    source: "NOTICE section 4",
  },
  // --- clap ----------------------------------------------------------------
  Artifact {
    file: "clapkit/clap_audio.mlmodelc/weights/weight.bin",
    key: Key::Sha256("723fe6aab7c4af1c671a210a35c289c67763bc6a7532b9df155a0c3fc0c3c9d7"),
    pin: "tests/clap/model_io.rs::clap_audio_artifacts_match_pinned_sha256",
    staged_by: "FinDIT-Studio/clapkit-coreml",
    gate: "clap",
    weights: Terms::Attribution(
      "laion/clap-htsat-unfused. NOTICE section 6a records an upstream ambiguity — textclap's \
       MODELS.md treats the checkpoints as CC-BY-4.0, the current HF card declares apache-2.0 — \
       and BOTH require attribution, which is why this is attribution-required rather than \
       permissive. The LAION citation in NOTICE section 4's style must ship with any binary that \
       bundles these weights.",
    ),
    corpus: Terms::Unresolved(
      "LAION-Audio-630K is assembled from several audio-caption sources with differing terms, and \
       NOTICE section 6a records the weights layer only.",
    ),
    source: "NOTICE section 6a",
  },
  Artifact {
    file: "clapkit/clap_audio_int8.mlmodelc/weights/weight.bin",
    key: Key::Sha256("b3a37ec5550dcdd6932b314b830275ebcba013748421e1a517760b9afeabafb8"),
    pin: "tests/clap/model_io.rs::clap_audio_int8_artifacts_match_pinned_sha256",
    staged_by: "FinDIT-Studio/clapkit-coreml",
    gate: "clap",
    weights: Terms::Attribution("A palettization of the fp16 audio tower; same terms as it."),
    corpus: Terms::Unresolved("Same LAION-Audio-630K question as the fp16 audio tower."),
    source: "NOTICE section 6a",
  },
  Artifact {
    file: "clapkit/clap_text.mlmodelc/weights/weight.bin",
    key: Key::Sha256("7f4e15e9ccb0ffbc2341eec286e9d9934d3d3d8d6465dfddebed248bddc0e3dd"),
    pin: "tests/clap/text_model_io.rs::clap_text_artifacts_match_pinned_sha256",
    staged_by: "FinDIT-Studio/clapkit-coreml",
    gate: "clap",
    weights: Terms::Attribution("The audio tower's twin, same checkpoint and same terms."),
    corpus: Terms::Unresolved("Same LAION-Audio-630K question as the audio tower."),
    source: "NOTICE section 6a",
  },
  Artifact {
    file: "clapkit/clap_text_int8.mlmodelc/weights/weight.bin",
    key: Key::Sha256("f181a595cefce402335499c32ea2f9727ef334afea9c592a2eabebb4172350a0"),
    pin: "tests/clap/text_model_io.rs::clap_text_int8_artifacts_match_pinned_sha256",
    staged_by: "FinDIT-Studio/clapkit-coreml",
    gate: "clap",
    weights: Terms::Attribution("A palettization of the fp16 text tower; same terms as it."),
    corpus: Terms::Unresolved("Same LAION-Audio-630K question as the fp16 text tower."),
    source: "NOTICE section 6a",
  },
  // --- lid -----------------------------------------------------------------
  Artifact {
    file: "lid/SpeechBrainECAPAVoxLingua107.mlmodelc/weights/weight.bin",
    key: Key::Sha256("81fbb61f6706c50e924a2ee2a4fc04e6408276df948117a1c6ac7675c23aac67"),
    pin: "tests/lid/common/mod.rs::ARTIFACT_SHA256",
    staged_by: "aufklarer/SpeechBrain-ECAPA-VoxLingua107-21M-CoreML",
    gate: "lid",
    weights: Terms::Permissive(
      "Apache-2.0 at both layers of the chain: speechbrain/lang-id-voxlingua107-ecapa upstream, \
       and the aufklarer CoreML export that MODELS_LOCK stages declares apache-2.0 too.",
    ),
    corpus: Terms::Unresolved(
      "VoxLingua107 is scraped YouTube speech. The dataset's own terms are the thing to check and \
       NOTICE section 10a does not record them — it documents the weights layer and the export's \
       stated changes only.",
    ),
    source: "NOTICE section 10a",
  },
];

// ---------------------------------------------------------------------------
// The three directions, as predicates over data
// ---------------------------------------------------------------------------
//
// Pure functions returning the failures they found, so the hermetic falsifiers
// below can drive exactly the same code the real-table checks do. A predicate
// only the happy path ever reaches is not a predicate.

/// Direction 1 — every staged repository is covered, and every row covers one.
fn unmatched_coverage(tables: &[String], rows: &[Artifact]) -> Vec<String> {
  let staged: BTreeSet<&str> = tables.iter().map(String::as_str).collect();
  let claimed: BTreeSet<&str> = rows.iter().map(|r| r.staged_by).collect();
  let mut failures = Vec::new();
  for repo in staged.difference(&claimed) {
    failures.push(format!(
      "MODELS_LOCK stages {repo:?} and no licence row covers it. Every staged repository needs at \
       least one row: an artifact whose terms nobody wrote down is an artifact nobody can clear \
       for the shipping path."
    ));
  }
  for repo in claimed.difference(&staged) {
    failures.push(format!(
      "a licence row is staged_by {repo:?}, which no MODELS_LOCK table names. Either the table \
       was removed and the row is describing bytes CI no longer fetches, or the name is a typo."
    ));
  }
  failures
}

/// Direction 2 — no research-only artifact is reachable without opting in.
///
/// Two clauses, because `default = []` alone would make this vacuous forever:
/// the row's gate must not be reachable from `default`, AND it must carry the
/// [`COMMERCIAL_PREFIX`]. A research-only artifact behind a plain kit feature
/// such as `speaker` is exactly as shipped as one in `default` — every
/// downstream product enables the kit it uses.
fn research_only_reachable(rows: &[Artifact], default_closure: &BTreeSet<String>) -> Vec<String> {
  let mut failures = Vec::new();
  for row in rows {
    let Some(layer) = row.disqualifying_layer() else {
      continue;
    };
    let terms = row
      .disqualifying_terms()
      .expect("a disqualified row has terms");
    if default_closure.contains(row.gate) {
      failures.push(format!(
        "{}: research-only at the {layer} layer, but its gate {:?} is reachable from `default`, \
         so a plain `cargo add coremlit` turns it on. {}",
        row.file,
        row.gate,
        terms.detail()
      ));
    }
    if !row.gate.starts_with(COMMERCIAL_PREFIX) {
      failures.push(format!(
        "{}: research-only at the {layer} layer, but its gate {:?} does not carry the {:?} \
         prefix. A plain kit feature is not an opt-in — every product that uses the kit enables \
         it. {}",
        row.file,
        row.gate,
        COMMERCIAL_PREFIX,
        terms.detail()
      ));
    }
  }
  failures
}

/// Direction 3 — no `commercial-` feature gates only clear artifacts.
///
/// The one people forget. A gate that protects nothing is worse than no gate:
/// it reads as a live restriction, so nobody re-examines the artifacts behind
/// it, and the next artifact added there inherits reassurance it never earned.
fn commercial_features_gating_nothing_restricted(
  rows: &[Artifact],
  features: &BTreeSet<String>,
) -> Vec<String> {
  let mut failures = Vec::new();
  for feature in features.iter().filter(|f| f.starts_with(COMMERCIAL_PREFIX)) {
    let gated: Vec<&Artifact> = rows.iter().filter(|r| r.gate == feature.as_str()).collect();
    if gated.is_empty() {
      failures.push(format!(
        "feature {feature:?} carries the {COMMERCIAL_PREFIX:?} prefix but no licence row is gated \
         by it. Either it gates an artifact with no row (direction 1), or it is a gate left \
         standing after the artifact it protected went away — retire it."
      ));
      continue;
    }
    if gated.iter().all(|r| r.disqualifying_layer().is_none()) {
      let cleared: Vec<&str> = gated.iter().map(|r| r.file).collect();
      failures.push(format!(
        "feature {feature:?} carries the {COMMERCIAL_PREFIX:?} prefix, but every artifact it \
         gates is CLEAR: {}. An upstream relicensed, or the terms were re-read — either way the \
         gate now says a restriction exists that does not, so retire it and move the artifacts to \
         a plain feature.",
        cleared.join(", ")
      ));
    }
  }
  failures
}

/// The documentation rule for [`COMMERCIAL_PREFIX`] features.
fn commercial_features_without_the_phrase(
  features: &BTreeSet<String>,
  docs: &BTreeMap<String, String>,
) -> Vec<String> {
  let mut failures = Vec::new();
  for feature in features.iter().filter(|f| f.starts_with(COMMERCIAL_PREFIX)) {
    let doc = docs.get(feature).map_or("", String::as_str);
    if doc.trim().is_empty() {
      failures.push(format!(
        "feature {feature:?} has no documentation comment above it in Cargo.toml. The prefix can \
         be read as \"cleared for commercial use\"; the first sentence is what stops that."
      ));
      continue;
    }
    let first = first_sentence(doc);
    if !normalise_spelling(&first).contains(COMMERCIAL_DOC_PHRASE) {
      failures.push(format!(
        "feature {feature:?}: its first documented sentence is {first:?}, which does not say \
         {COMMERCIAL_DOC_PHRASE:?}. The name reads as an ENDORSEMENT of commercial use; the \
         sentence that corrects it has to be the first one, not a caveat further down."
      ));
    }
  }
  failures
}

/// The first sentence of a documentation block: everything up to the first
/// full stop that ends a word, with the block's line breaks flattened.
fn first_sentence(doc: &str) -> String {
  let flat = doc.split_whitespace().collect::<Vec<_>>().join(" ");
  match flat.find(". ") {
    Some(end) => flat[..=end].trim().to_string(),
    None => flat.trim_end_matches('.').trim().to_string(),
  }
}

/// Lowercase, with the British spelling folded onto the American one.
///
/// Both spellings satisfy the rule. Failing a feature for writing "license"
/// would be a trap with no safety value — the reader is warned either way.
fn normalise_spelling(text: &str) -> String {
  text.to_lowercase().replace("licence", "license")
}

// ---------------------------------------------------------------------------
// Repository readers
// ---------------------------------------------------------------------------

/// One `["repo/name"]` table of `MODELS_LOCK`, reduced to what this file needs.
struct LockTable {
  name: String,
  fields: BTreeMap<String, String>,
}

/// `MODELS_LOCK`, or `None` outside the repository workspace.
///
/// The lock is deliberately NOT packaged with the crate, so a `cargo test` run
/// from the published tarball must SKIP rather than fail `NotFound` — the same
/// contract `tests/whisper/models_lock.rs` documents.
fn lock_tables() -> Option<Vec<LockTable>> {
  let root = workspace_root::try_workspace_root()?;
  let lock = root.join("MODELS_LOCK");
  if !lock.is_file() {
    eprintln!("model_licences checks skipped: not in the repository workspace");
    return None;
  }
  let text = std::fs::read_to_string(&lock).unwrap_or_else(|e| panic!("read MODELS_LOCK: {e}"));
  Some(parse_lock(&text))
}

/// A tiny hand-rolled reader over the lock's fixed `["repo/name"]` +
/// `key = "value"` shape, mirroring the one in
/// `tests/whisper/models_lock.rs` and the sed/awk block ci.yml runs at CI
/// time. No TOML crate: the point is to read what the file literally says.
fn parse_lock(contents: &str) -> Vec<LockTable> {
  let mut tables: Vec<LockTable> = Vec::new();
  for line in contents.lines() {
    let line = line.trim();
    if line.is_empty() || line.starts_with('#') {
      continue;
    }
    if let Some(name) = line.strip_prefix("[\"").and_then(|s| s.strip_suffix("\"]")) {
      tables.push(LockTable {
        name: name.to_string(),
        fields: BTreeMap::new(),
      });
      continue;
    }
    let Some(table) = tables.last_mut() else {
      continue; // a pre-table key (`cache-epoch`), not this reader's concern
    };
    let Some((key, value)) = line.split_once('=') else {
      panic!("MODELS_LOCK: not a table header or `key = value`: {line:?}");
    };
    let value = value.trim();
    let value = value
      .strip_prefix('"')
      .and_then(|v| v.strip_suffix('"'))
      .unwrap_or_else(|| panic!("MODELS_LOCK: value for {key:?} is not quoted: {value:?}"));
    table
      .fields
      .insert(key.trim().to_string(), value.to_string());
  }
  tables
}

/// A file addressed relative to this crate's manifest directory.
fn read_rel(rel: &str) -> String {
  std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join(rel))
    .unwrap_or_else(|e| panic!("read {rel}: {e}"))
}

/// The `[features]` block of this crate's manifest, comments included.
fn features_block() -> String {
  features_block_of(&read_rel("Cargo.toml"))
}

/// The same block, read from the manifest the REPOSITORY holds rather than the
/// one the compiling package happens to sit next to.
///
/// `cargo package` re-serialises the manifest into the tarball and DROPS EVERY
/// COMMENT doing it, so a feature's documentation exists only in the
/// checked-in file. Checks that read comments must read that file; checks that
/// need only names or entries are happy with either. `None` outside the
/// repository workspace, where the comment-bearing manifest is not present at
/// all and the rule is simply unverifiable.
fn repository_features_block() -> Option<String> {
  let root = workspace_root::try_workspace_root()?;
  let manifest = root.join("coremlit/Cargo.toml");
  if !manifest.is_file() {
    eprintln!("model_licences: no comment-bearing manifest; the doc rule is skipped");
    return None;
  }
  let text = std::fs::read_to_string(&manifest)
    .unwrap_or_else(|e| panic!("read {}: {e}", manifest.display()));
  Some(features_block_of(&text))
}

/// The `[features]` block of `manifest`, comments included.
fn features_block_of(manifest: &str) -> String {
  let mut out = String::new();
  let mut inside = false;
  for line in manifest.lines() {
    if line.starts_with('[') {
      inside = line.trim() == "[features]";
      continue;
    }
    if inside {
      out.push_str(line);
      out.push('\n');
    }
  }
  out
}

/// The declared feature names.
fn feature_names(block: &str) -> BTreeSet<String> {
  let mut names = BTreeSet::new();
  for line in block.lines() {
    if line.starts_with(char::is_whitespace) {
      continue;
    }
    let trimmed = line.trim_start();
    if trimmed.is_empty() || trimmed.starts_with('#') {
      continue;
    }
    if let Some((key, _)) = line.split_once('=') {
      let key = key.trim();
      if !key.is_empty() && !key.contains(char::is_whitespace) {
        names.insert(key.to_string());
      }
    }
  }
  names
}

/// One feature's entries — the quoted contents of its `[..]` value, spread
/// over as many lines as rustfmt/taplo left it on.
fn feature_entries(block: &str, feature: &str) -> Vec<String> {
  let mut collecting = false;
  let mut buf = String::new();
  for line in block.lines() {
    if collecting {
      buf.push('\n');
      buf.push_str(line);
      if line.contains(']') {
        break;
      }
      continue;
    }
    if line.starts_with(char::is_whitespace) {
      continue;
    }
    let Some((key, rest)) = line.split_once('=') else {
      continue;
    };
    if key.trim() != feature {
      continue;
    }
    collecting = true;
    buf.push_str(rest);
    if rest.contains(']') {
      break;
    }
  }
  buf
    .split('"')
    .skip(1)
    .step_by(2)
    .map(str::to_string)
    .collect()
}

/// Every feature transitively enabled by `seed`, `seed` included.
///
/// Entries naming a dependency (`dep:x`) or a dependency's own feature (`x/y`)
/// are not this crate's features and do not extend the closure.
fn feature_closure(block: &str, seed: &str) -> BTreeSet<String> {
  let mut seen = BTreeSet::new();
  let mut queue = vec![seed.to_string()];
  while let Some(feature) = queue.pop() {
    if !seen.insert(feature.clone()) {
      continue;
    }
    for entry in feature_entries(block, &feature) {
      if !entry.starts_with("dep:") && !entry.contains('/') {
        queue.push(entry);
      }
    }
  }
  seen
}

/// The contiguous `#` comment block immediately above each feature.
///
/// A blank line ends a block, so a comment about the section above cannot be
/// mistaken for documentation of the feature below it.
fn feature_docs(block: &str) -> BTreeMap<String, String> {
  let mut docs = BTreeMap::new();
  let mut pending: Vec<&str> = Vec::new();
  for line in block.lines() {
    let trimmed = line.trim();
    if trimmed.is_empty() {
      pending.clear();
      continue;
    }
    if let Some(comment) = trimmed.strip_prefix('#') {
      pending.push(comment.trim());
      continue;
    }
    if line.starts_with(char::is_whitespace) {
      continue;
    }
    if let Some((key, _)) = line.split_once('=') {
      let key = key.trim();
      if !key.is_empty() && !key.contains(char::is_whitespace) {
        docs.insert(key.to_string(), pending.join(" "));
      }
    }
    pending.clear();
  }
  docs
}

/// The SHA-256 pins recorded at one [`Artifact::pin`] locator.
///
/// Either a path-keyed manifest (`&[("weights/weight.bin", "<hex>"), ..]`, the
/// shape every per-kit `model_io` gate uses) or bare hex literals (the shape a
/// scalar `TOKENIZER_SHA256_HEX` uses). Both are read as text: the pins live in
/// other test binaries and in feature-gated modules, so they cannot be
/// imported — but they CAN be read, which is what
/// `tests/whisper/models_lock.rs` already does to hold the fp16 roster.
enum Pins {
  /// Bundle-relative path to SHA-256.
  Manifest(BTreeMap<String, String>),
  /// Bare SHA-256 literals.
  Literals(BTreeSet<String>),
}

/// Cut markers for the end of a pinned item: the next thing that starts at
/// column zero. A pin list never contains one, so the window is exactly the
/// declaration that was anchored.
const ITEM_END: &[&str] = &[
  "\nconst ",
  "\npub const ",
  "\nfn ",
  "\npub fn ",
  "\nstatic ",
  "\n#[",
  "\n/// ",
  "\n//!",
  "\n}",
];

/// Reads `<crate-relative source>::<identifier>`.
///
/// The identifier is anchored on its DECLARATION (`const NAME:` or `fn NAME(`),
/// not on a bare mention, and the declaration must occur exactly once — an
/// ambiguous anchor is a reader that could silently read the wrong pin.
fn pins_at(locator: &str) -> Pins {
  let (rel, ident) = locator
    .split_once("::")
    .unwrap_or_else(|| panic!("pin locator {locator:?} is not `<source>::<identifier>`"));
  let text = read_rel(rel);

  let konst = format!("const {ident}:");
  let func = format!("fn {ident}(");
  let anchor = if text.matches(konst.as_str()).count() == 1 {
    konst
  } else if text.matches(func.as_str()).count() == 1 {
    func
  } else {
    panic!(
      "pin locator {locator:?}: {rel} holds {} declarations `{konst}` and {} declarations \
       `{func}`; exactly one of the two must be present exactly once, or this reader could be \
       reading the wrong pin",
      text.matches(konst.as_str()).count(),
      text.matches(func.as_str()).count()
    )
  };

  let start = text.find(&anchor).expect("anchor counted above");
  let rest = &text[start + 1..];
  let end = ITEM_END
    .iter()
    .filter_map(|marker| rest.find(marker))
    .min()
    .unwrap_or(rest.len());
  let window = &text[start..start + 1 + end];

  let quoted: Vec<&str> = window.split('"').skip(1).step_by(2).collect();
  let manifest: BTreeMap<String, String> = quoted
    .windows(2)
    .filter(|pair| is_sha256(pair[1]) && !is_sha256(pair[0]))
    .map(|pair| (pair[0].to_string(), pair[1].to_string()))
    .collect();
  if manifest.is_empty() {
    let literals: BTreeSet<String> = quoted
      .iter()
      .filter(|q| is_sha256(q))
      .map(|q| (*q).to_string())
      .collect();
    assert!(
      !literals.is_empty(),
      "pin locator {locator:?}: no SHA-256 found under `{anchor}`. The reader anchored, so the \
       declaration moved or stopped holding hashes."
    );
    Pins::Literals(literals)
  } else {
    Pins::Manifest(manifest)
  }
}

/// Whether `s` is 64 lowercase hex digits.
fn is_sha256(s: &str) -> bool {
  s.len() == 64 && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

/// The bundle-relative key a row's pin manifest is looked up by: whatever
/// follows the last `.mlmodelc/`, or the bare file name when the artifact is
/// not inside a compiled bundle.
fn bundle_relative(file: &str) -> &str {
  file
    .rsplit_once(".mlmodelc/")
    .map_or_else(|| file.rsplit('/').next().unwrap_or(file), |(_, tail)| tail)
}

// ---------------------------------------------------------------------------
// The live checks — the real table, the real lock, the real manifest
// ---------------------------------------------------------------------------

/// **Direction 1.** Every repository `MODELS_LOCK` stages is covered by a
/// licence row, and every row names a repository that is actually staged.
///
/// Both halves, because either one alone rots: coverage-only lets a row
/// outlive the table it describes, and reverse-only lets a new table arrive
/// with nobody having asked what its bytes permit.
#[test]
fn every_staged_repo_has_a_licence_row_and_every_row_names_a_staged_repo() {
  let Some(tables) = lock_tables() else {
    return;
  };
  assert!(
    tables.len() >= 8,
    "only {} MODELS_LOCK tables parsed; this reader has stopped matching the lock's shape and \
     would pass vacuously",
    tables.len()
  );
  let names: Vec<String> = tables.iter().map(|t| t.name.clone()).collect();
  let failures = unmatched_coverage(&names, ARTIFACTS);
  assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// **Direction 2.** No research-only artifact is reachable from `default`, and
/// none is gated by a feature without the [`COMMERCIAL_PREFIX`].
///
/// Vacuous against today's table — nothing is research-only — and deliberately
/// kept anyway: it is the check the first disqualifying artifact will meet.
/// `falsifiers::direction_two_*` are what prove it can still fire.
#[test]
fn no_research_only_artifact_is_reachable_without_a_commercial_gate() {
  let block = features_block();
  let closure = feature_closure(&block, "default");
  let failures = research_only_reachable(ARTIFACTS, &closure);
  assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// **Direction 3.** No `commercial-`prefixed feature gates only clear
/// artifacts.
#[test]
fn every_commercial_feature_gates_a_research_only_artifact() {
  let block = features_block();
  let features = feature_names(&block);
  let failures = commercial_features_gating_nothing_restricted(ARTIFACTS, &features);
  assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Every `commercial-` feature's first documented sentence says a commercial
/// licence is required — the correction for a prefix that can be read
/// backwards.
#[test]
fn every_commercial_feature_says_it_requires_a_commercial_licence_first() {
  let Some(block) = repository_features_block() else {
    return;
  };
  assert!(
    block.contains('#'),
    "the `[features]` block read for the doc rule carries no comments at all, so the rule would \
     pass vacuously. That is the stripped manifest `cargo package` writes, not the checked-in one."
  );
  let features = feature_names(&block);
  let docs = feature_docs(&block);
  let failures = commercial_features_without_the_phrase(&features, &docs);
  assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// No `commercial-` feature is reachable from `default`.
///
/// Stronger than direction 2's first clause and independent of any row: even
/// before a research-only artifact exists, a gate that `default` turns on is
/// not a gate. Today `default = []`, so the closure is `{"default"}` and this
/// holds trivially; it stops holding the moment somebody adds one.
#[test]
fn no_commercial_feature_is_reachable_from_default() {
  let block = features_block();
  let closure = feature_closure(&block, "default");
  let leaked: Vec<&String> = closure
    .iter()
    .filter(|f| f.starts_with(COMMERCIAL_PREFIX))
    .collect();
  assert!(
    leaked.is_empty(),
    "`default` enables {leaked:?}. A {COMMERCIAL_PREFIX:?} feature is an opt-in by construction; \
     reaching it from `default` means every `cargo add coremlit` consumer takes on a licence \
     obligation without asking for it."
  );
}

/// Every row is keyed by a well-formed SHA-256, or carries a reason it cannot
/// be.
///
/// Well-formedness is not pedantry: a truncated or upper-case hash silently
/// stops matching the pin it is supposed to equal, and a placeholder like
/// `"TODO"` would sail through a check that only compared strings.
#[test]
fn every_row_is_keyed_by_a_wellformed_sha256_or_a_reasoned_exemption() {
  for row in ARTIFACTS {
    match row.key {
      Key::Sha256(hex) => {
        assert!(
          is_sha256(hex),
          "{}: key {hex:?} is not 64 lowercase hex digits",
          row.file
        );
        assert!(
          !row.pin.is_empty(),
          "{}: a hashed row must name the pin its hash is copied from, or the hash goes stale the \
           first time the artifact is re-converted",
          row.file
        );
      }
      Key::Unpinned(reason) => {
        assert!(
          !reason.trim().is_empty(),
          "{}: an unpinned row with no reason is an exemption nobody can retire",
          row.file
        );
        assert!(
          row.pin.is_empty(),
          "{}: an unpinned row names the pin {:?}. If those bytes are pinned, key on them.",
          row.file,
          row.pin
        );
      }
    }
  }
}

/// A row may be [`Key::Unpinned`] only while its table is on
/// `revision = "main"`, and every row on a commit-pinned table must be hashed.
///
/// The staleness half, in the `CHECKSUMLESS_KITS` style: the exemption is tied
/// to its cause in both directions, so `MODELS_LOCK`'s LOUD FOLLOW-UP landing —
/// whisper's two tables moving from `main` to an immutable commit — turns this
/// red and forces the hashes in, instead of leaving three rows describing bytes
/// nobody can identify.
#[test]
fn unpinned_rows_exist_only_where_the_lock_pins_a_moving_revision() {
  let Some(tables) = lock_tables() else {
    return;
  };
  let revisions: BTreeMap<&str, &str> = tables
    .iter()
    .map(|t| {
      (
        t.name.as_str(),
        t.fields.get("revision").map_or("", String::as_str),
      )
    })
    .collect();
  let mut moving = 0usize;
  for row in ARTIFACTS {
    let revision = revisions.get(row.staged_by).copied().unwrap_or_else(|| {
      panic!(
        "{}: staged_by {:?} names no MODELS_LOCK table",
        row.file, row.staged_by
      )
    });
    if revision == "main" {
      moving += 1;
    }
    match row.key {
      Key::Sha256(_) => assert_ne!(
        revision, "main",
        "{}: keyed by SHA-256, but MODELS_LOCK's {:?} is still on `revision = \"main\"`. The \
         bytes CI fetches can change without the lock changing, so the hash is a claim about one \
         download rather than about the artifact.",
        row.file, row.staged_by
      ),
      Key::Unpinned(_) => assert_eq!(
        revision, "main",
        "{}: exempt from hashing, but MODELS_LOCK's {:?} pins an immutable revision {revision:?}. \
         The reason for the exemption is gone — key this row on the bytes at that revision.",
        row.file, row.staged_by
      ),
    }
  }
  assert!(
    moving > 0,
    "no row sits on a `revision = \"main\"` table, so this check no longer sees the case it \
     exists for. Delete it, or the exemption it guards."
  );
}

/// Every row's SHA-256 equals the pin it names.
///
/// The hash is the KEY; a key nothing verifies is a comment. This is what
/// makes it a key: re-convert an artifact, re-pin its `model_io` gate, and this
/// goes red until somebody re-reads the licence for the new bytes.
#[test]
fn every_rows_sha256_matches_the_pin_it_names() {
  let mut checked = 0usize;
  for row in ARTIFACTS {
    let Some(expected) = row.sha256() else {
      continue;
    };
    match pins_at(row.pin) {
      Pins::Manifest(manifest) => {
        let key = bundle_relative(row.file);
        let pinned = manifest.get(key).unwrap_or_else(|| {
          panic!(
            "{}: pin {:?} holds no entry for {key:?} (it holds {:?}). The row names a file its \
             own pin does not cover.",
            row.file,
            row.pin,
            manifest.keys().collect::<Vec<_>>()
          )
        });
        assert_eq!(
          pinned, expected,
          "{}: the licence row keys on {expected} but {} pins {pinned} for {key}. These are \
           different bytes, and the licence attaches to bytes.",
          row.file, row.pin
        );
      }
      Pins::Literals(literals) => assert!(
        literals.contains(expected),
        "{}: the licence row keys on {expected}, which {} does not pin (it pins {:?}).",
        row.file,
        row.pin,
        literals
      ),
    }
    checked += 1;
  }
  assert!(
    checked >= 10,
    "only {checked} rows cross-checked against a pin; the table has shrunk or the readers have \
     stopped matching, and this check would pass vacuously"
  );
}

/// **The AuraFace rule.** Two rows over the same SHA-256 agree on what those
/// bytes permit.
///
/// `fal/AuraFace-v1` is tagged `apache-2.0` while four of its five ONNX files
/// are byte-identical to InsightFace artifacts distributed for non-commercial
/// research only. Both statements cannot be true of the same bytes: one of them
/// is the repository's claim about a file it did not train. Keying on the hash
/// is what makes the contradiction visible instead of letting the second
/// repository's tag quietly overwrite the first's restriction.
#[test]
fn identical_bytes_carry_identical_terms() {
  let failures = contradictory_terms(ARTIFACTS);
  assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Rows that key on the same bytes and disagree about them.
fn contradictory_terms(rows: &[Artifact]) -> Vec<String> {
  let mut by_hash: BTreeMap<&str, Vec<&Artifact>> = BTreeMap::new();
  for row in rows {
    if let Some(hex) = row.sha256() {
      by_hash.entry(hex).or_default().push(row);
    }
  }
  let mut failures = Vec::new();
  for (hex, group) in by_hash {
    let first = group[0];
    for other in &group[1..] {
      if first.weights.verdict() != other.weights.verdict()
        || first.corpus.verdict() != other.corpus.verdict()
      {
        failures.push(format!(
          "sha256 {hex} is claimed by {} (weights {}, corpus {}) and by {} (weights {}, corpus \
           {}). Identical bytes cannot carry different terms — one row is repeating a repository \
           tag rather than the licence of the artifact it re-hosts.",
          first.file,
          first.weights.verdict(),
          first.corpus.verdict(),
          other.file,
          other.weights.verdict(),
          other.corpus.verdict(),
        ));
      }
    }
  }
  failures
}

/// Every row's file lives under the directory its `MODELS_LOCK` table stages,
/// and — where the table names explicit `files` — is one of them.
///
/// A row attached to a path its own table does not stage is terms recorded
/// against the wrong bytes, which is the same class of error as keying by
/// repository.
#[test]
fn every_row_lives_under_the_directory_its_table_stages() {
  let Some(tables) = lock_tables() else {
    return;
  };
  let by_name: BTreeMap<&str, &LockTable> = tables.iter().map(|t| (t.name.as_str(), t)).collect();
  for row in ARTIFACTS {
    let table = by_name.get(row.staged_by).unwrap_or_else(|| {
      panic!(
        "{}: staged_by {:?} names no MODELS_LOCK table",
        row.file, row.staged_by
      )
    });
    let local_dir = table
      .fields
      .get("local-dir")
      .unwrap_or_else(|| panic!("MODELS_LOCK table {:?} has no `local-dir`", row.staged_by));
    let vendor_path = local_dir
      .strip_prefix("Models/")
      .unwrap_or_else(|| panic!("`local-dir` {local_dir:?} does not start with `Models/`"));
    let prefix = format!("{vendor_path}/");
    let tail = row.file.strip_prefix(&prefix).unwrap_or_else(|| {
      panic!(
        "{}: staged_by {:?} downloads into {local_dir:?}, so the row's path must start with \
         {prefix:?}",
        row.file, row.staged_by
      )
    });
    if let Some(files) = table.fields.get("files") {
      assert!(
        files.split_whitespace().any(|f| f == tail),
        "{}: table {:?} stages the explicit file list {files:?}, which does not include {tail:?}",
        row.file,
        row.staged_by
      );
    }
  }
}

/// Every verdict carries prose, and every unresolved one names what is open.
///
/// An empty payload turns the table back into a bare SPDX list, which is the
/// thing it was built not to be — and an `Unresolved` with nothing to follow
/// is indistinguishable from nobody having looked.
#[test]
fn every_verdict_carries_its_reasoning() {
  for row in ARTIFACTS {
    for (layer, terms) in [("weights", row.weights), ("corpus", row.corpus)] {
      assert!(
        terms.detail().trim().len() > 40,
        "{}: the {layer} verdict {:?} carries no reasoning ({:?})",
        row.file,
        terms.verdict(),
        terms.detail()
      );
    }
    assert!(
      !row.source.trim().is_empty(),
      "{}: no source recorded for its verdicts",
      row.file
    );
  }
}

/// No file is listed twice; a second row would be unreachable and could
/// silently disagree with the first.
#[test]
fn no_file_is_listed_twice() {
  let files: BTreeSet<&str> = ARTIFACTS.iter().map(|r| r.file).collect();
  assert_eq!(
    files.len(),
    ARTIFACTS.len(),
    "the licence table lists a file twice"
  );
}

/// The state of the table, asserted rather than remembered.
///
/// If this goes red because a row became research-only, that is the point: the
/// module doc above says no row is, directions 2 and 3 are described as
/// tripwires, and both statements have to be revisited together.
#[test]
fn no_row_is_research_only_today_and_the_doc_says_so() {
  let restricted: Vec<&str> = ARTIFACTS
    .iter()
    .filter(|r| r.disqualifying_layer().is_some())
    .map(|r| r.file)
    .collect();
  assert!(
    restricted.is_empty(),
    "these rows are now research-only: {restricted:?}. Directions 2 and 3 are live from here on: \
     move them behind a `{COMMERCIAL_PREFIX}` feature and rewrite this file's module doc, which \
     currently tells the reader nothing is restricted."
  );
  let unresolved = ARTIFACTS
    .iter()
    .filter(|r| matches!(r.corpus, Terms::Unresolved(_)))
    .count();
  assert_eq!(
    unresolved,
    ARTIFACTS.len(),
    "the module doc says the corpus layer is unresolved for EVERY row because NOTICE documents \
     none; that is no longer true, so say what changed."
  );
}

// ---------------------------------------------------------------------------
// Falsifiers — the predicates against input built to trip them
// ---------------------------------------------------------------------------

/// A check nobody has watched fail is not a check.
///
/// Every predicate above is driven here against doctored data: the shape it
/// must FLAG, and the shape it must not. These run everywhere — no models, no
/// repository files, no features — so directions 2 and 3 stay demonstrably
/// live even while the real table has nothing for them to catch.
mod falsifiers {
  use std::collections::{BTreeMap, BTreeSet};

  use super::{
    Artifact, Key, Terms, commercial_features_gating_nothing_restricted,
    commercial_features_without_the_phrase, contradictory_terms, feature_closure, feature_docs,
    feature_names, first_sentence, research_only_reachable, unmatched_coverage,
  };

  /// A row with everything but the fields a given test is about.
  const fn row(
    file: &'static str,
    staged_by: &'static str,
    gate: &'static str,
    weights: Terms,
    corpus: Terms,
  ) -> Artifact {
    Artifact {
      file,
      key: Key::Sha256("0000000000000000000000000000000000000000000000000000000000000000"),
      pin: "a falsifier row, never resolved against the tree",
      staged_by,
      gate,
      weights,
      corpus,
      source: "a falsifier, not a real record",
    }
  }

  const CLEAR: Terms = Terms::Permissive("clear, for a falsifier");
  const RESTRICTED: Terms =
    Terms::ResearchOnly("non-commercial research purposes only, for a falsifier");

  fn staged(names: &[&str]) -> Vec<String> {
    names.iter().map(|n| (*n).to_string()).collect()
  }

  fn features(names: &[&str]) -> BTreeSet<String> {
    names.iter().map(|n| (*n).to_string()).collect()
  }

  // --- direction 1 ---------------------------------------------------------

  #[test]
  fn direction_one_passes_when_every_table_and_row_line_up() {
    let rows = [row("a/w.bin", "vendor/one", "kit", CLEAR, CLEAR)];
    assert!(unmatched_coverage(&staged(&["vendor/one"]), &rows).is_empty());
  }

  #[test]
  fn direction_one_reds_when_a_staged_repo_has_no_row() {
    let rows = [row("a/w.bin", "vendor/one", "kit", CLEAR, CLEAR)];
    let failures = unmatched_coverage(&staged(&["vendor/one", "vendor/two"]), &rows);
    assert_eq!(failures.len(), 1, "{failures:?}");
    assert!(failures[0].contains("vendor/two"), "{failures:?}");
  }

  #[test]
  fn direction_one_reds_when_a_row_names_no_staged_repo() {
    let rows = [
      row("a/w.bin", "vendor/one", "kit", CLEAR, CLEAR),
      row("b/w.bin", "vendor/gone", "kit", CLEAR, CLEAR),
    ];
    let failures = unmatched_coverage(&staged(&["vendor/one"]), &rows);
    assert_eq!(failures.len(), 1, "{failures:?}");
    assert!(failures[0].contains("vendor/gone"), "{failures:?}");
  }

  // --- direction 2 ---------------------------------------------------------

  #[test]
  fn direction_two_passes_when_a_restricted_row_sits_behind_an_opt_in_commercial_gate() {
    let rows = [row(
      "a/w.bin",
      "vendor/one",
      "commercial-face",
      CLEAR,
      RESTRICTED,
    )];
    let closure = features(&["default"]);
    assert!(research_only_reachable(&rows, &closure).is_empty());
  }

  #[test]
  fn direction_two_reds_when_a_research_only_row_is_reachable_from_default() {
    let rows = [row(
      "a/w.bin",
      "vendor/one",
      "commercial-face",
      CLEAR,
      RESTRICTED,
    )];
    let closure = features(&["default", "commercial-face"]);
    let failures = research_only_reachable(&rows, &closure);
    assert_eq!(failures.len(), 1, "{failures:?}");
    assert!(
      failures[0].contains("reachable from `default`"),
      "{failures:?}"
    );
    assert!(failures[0].contains("training corpus"), "{failures:?}");
  }

  #[test]
  fn direction_two_reds_when_a_research_only_row_is_gated_by_a_plain_feature() {
    let rows = [row("a/w.bin", "vendor/one", "speaker", RESTRICTED, CLEAR)];
    let failures = research_only_reachable(&rows, &features(&["default"]));
    assert_eq!(failures.len(), 1, "{failures:?}");
    assert!(failures[0].contains("does not carry"), "{failures:?}");
    assert!(failures[0].contains("weights layer"), "{failures:?}");
  }

  #[test]
  fn direction_two_names_the_weights_layer_when_that_is_what_disqualifies() {
    let rows = [row("a/w.bin", "vendor/one", "kit", RESTRICTED, CLEAR)];
    let failures = research_only_reachable(&rows, &features(&["default"]));
    assert!(failures[0].contains("weights layer"), "{failures:?}");
  }

  #[test]
  fn direction_two_names_the_corpus_layer_when_that_is_what_disqualifies() {
    let rows = [row("a/w.bin", "vendor/one", "kit", CLEAR, RESTRICTED)];
    let failures = research_only_reachable(&rows, &features(&["default"]));
    assert!(
      failures[0].contains("training corpus layer"),
      "{failures:?}"
    );
  }

  #[test]
  fn direction_two_ignores_unresolved_rows_rather_than_treating_them_as_restricted() {
    let rows = [row(
      "a/w.bin",
      "vendor/one",
      "kit",
      Terms::Unresolved("open question, for a falsifier"),
      CLEAR,
    )];
    assert!(research_only_reachable(&rows, &features(&["default"])).is_empty());
  }

  // --- direction 3 ---------------------------------------------------------

  #[test]
  fn direction_three_passes_when_a_commercial_gate_covers_a_restricted_row() {
    let rows = [row(
      "a/w.bin",
      "vendor/one",
      "commercial-face",
      CLEAR,
      RESTRICTED,
    )];
    let declared = features(&["default", "commercial-face"]);
    assert!(commercial_features_gating_nothing_restricted(&rows, &declared).is_empty());
  }

  #[test]
  fn direction_three_reds_when_a_commercial_feature_gates_only_clear_artifacts() {
    let rows = [row(
      "a/w.bin",
      "vendor/one",
      "commercial-face",
      CLEAR,
      CLEAR,
    )];
    let declared = features(&["default", "commercial-face"]);
    let failures = commercial_features_gating_nothing_restricted(&rows, &declared);
    assert_eq!(failures.len(), 1, "{failures:?}");
    assert!(
      failures[0].contains("every artifact it gates is CLEAR"),
      "{failures:?}"
    );
  }

  #[test]
  fn direction_three_reds_when_a_commercial_feature_gates_nothing_at_all() {
    let rows = [row("a/w.bin", "vendor/one", "speaker", CLEAR, CLEAR)];
    let declared = features(&["default", "commercial-face"]);
    let failures = commercial_features_gating_nothing_restricted(&rows, &declared);
    assert_eq!(failures.len(), 1, "{failures:?}");
    assert!(
      failures[0].contains("no licence row is gated by it"),
      "{failures:?}"
    );
  }

  #[test]
  fn direction_three_leaves_plain_features_alone() {
    let rows = [row("a/w.bin", "vendor/one", "speaker", CLEAR, CLEAR)];
    let declared = features(&["default", "speaker", "whisper"]);
    assert!(commercial_features_gating_nothing_restricted(&rows, &declared).is_empty());
  }

  // --- the documentation rule ---------------------------------------------

  fn docs(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
    pairs
      .iter()
      .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
      .collect()
  }

  #[test]
  fn the_doc_rule_passes_on_a_first_sentence_that_says_it() {
    let declared = features(&["commercial-face"]);
    let written = docs(&[(
      "commercial-face",
      "Requires a commercial licence from the weights' author. Adds the face embedder.",
    )]);
    assert!(commercial_features_without_the_phrase(&declared, &written).is_empty());
  }

  #[test]
  fn the_doc_rule_accepts_either_spelling() {
    let declared = features(&["commercial-face"]);
    let written = docs(&[("commercial-face", "Requires a commercial license. Adds it.")]);
    assert!(commercial_features_without_the_phrase(&declared, &written).is_empty());
  }

  #[test]
  fn the_doc_rule_reds_when_the_phrase_arrives_after_the_first_sentence() {
    let declared = features(&["commercial-face"]);
    let written = docs(&[(
      "commercial-face",
      "Adds the face embedder. Requires a commercial licence.",
    )]);
    let failures = commercial_features_without_the_phrase(&declared, &written);
    assert_eq!(failures.len(), 1, "{failures:?}");
    assert!(
      failures[0].contains("has to be the first one"),
      "{failures:?}"
    );
  }

  #[test]
  fn the_doc_rule_reds_on_an_undocumented_commercial_feature() {
    let declared = features(&["commercial-face"]);
    let failures = commercial_features_without_the_phrase(&declared, &docs(&[]));
    assert_eq!(failures.len(), 1, "{failures:?}");
    assert!(
      failures[0].contains("no documentation comment"),
      "{failures:?}"
    );
  }

  #[test]
  fn the_doc_rule_leaves_plain_features_alone() {
    let declared = features(&["speaker"]);
    let written = docs(&[("speaker", "The CoreML segmentation and embedding backends.")]);
    assert!(commercial_features_without_the_phrase(&declared, &written).is_empty());
  }

  #[test]
  fn a_single_sentence_doc_is_read_whole() {
    assert_eq!(
      first_sentence("Requires a commercial licence."),
      "Requires a commercial licence"
    );
    assert_eq!(
      first_sentence("Requires a commercial licence. And more."),
      "Requires a commercial licence."
    );
  }

  // --- the AuraFace rule ---------------------------------------------------

  /// The exact shape that produced this file: one repository tags bytes
  /// `apache-2.0`, another distributes the same bytes for research only.
  #[test]
  fn the_auraface_collision_reds() {
    const SHA: &str = "aaaa000000000000000000000000000000000000000000000000000000000000";
    let rows = [
      Artifact {
        key: Key::Sha256(SHA),
        ..row(
          "auraface/glintr100.onnx",
          "fal/AuraFace-v1",
          "kit",
          CLEAR,
          CLEAR,
        )
      },
      Artifact {
        key: Key::Sha256(SHA),
        ..row(
          "insightface/glintr100.onnx",
          "insightface/buffalo_l",
          "kit",
          RESTRICTED,
          RESTRICTED,
        )
      },
    ];
    let failures = contradictory_terms(&rows);
    assert_eq!(failures.len(), 1, "{failures:?}");
    assert!(failures[0].contains(SHA), "{failures:?}");
  }

  #[test]
  fn agreeing_rows_over_the_same_bytes_pass() {
    const SHA: &str = "bbbb000000000000000000000000000000000000000000000000000000000000";
    let rows = [
      Artifact {
        key: Key::Sha256(SHA),
        ..row("one/w.bin", "vendor/one", "kit", CLEAR, CLEAR)
      },
      Artifact {
        key: Key::Sha256(SHA),
        ..row("two/w.bin", "vendor/two", "kit", CLEAR, CLEAR)
      },
    ];
    assert!(contradictory_terms(&rows).is_empty());
  }

  // --- the manifest readers ------------------------------------------------

  const DOCTORED_FEATURES: &str = "\
default = [\"speaker\"]
speaker = [\"dep:diaric\"]
# Requires a commercial licence from the weights' author.
# Adds the face embedder.
commercial-face = [\"dep:facelib\", \"speaker\"]

# A comment about the section, not about the feature after the blank line.

lid = [\"dep:rustfft\"]
";

  #[test]
  fn the_feature_reader_finds_every_declared_name() {
    assert_eq!(
      feature_names(DOCTORED_FEATURES),
      features(&["default", "speaker", "commercial-face", "lid"])
    );
  }

  #[test]
  fn the_closure_follows_this_crates_features_and_not_dependency_features() {
    assert_eq!(
      feature_closure(DOCTORED_FEATURES, "default"),
      features(&["default", "speaker"])
    );
    assert_eq!(
      feature_closure(DOCTORED_FEATURES, "commercial-face"),
      features(&["commercial-face", "speaker"])
    );
  }

  #[test]
  fn a_doc_block_stops_at_the_blank_line_above_it() {
    let written = feature_docs(DOCTORED_FEATURES);
    assert_eq!(
      written.get("commercial-face").map(String::as_str),
      Some("Requires a commercial licence from the weights' author. Adds the face embedder.")
    );
    assert_eq!(written.get("lid").map(String::as_str), Some(""));
  }

  /// The same manifest with `default` pulling the commercial gate in — the
  /// mutation direction 2's first clause exists for, read through the real
  /// manifest reader rather than a hand-built set.
  const LEAKY_FEATURES: &str = "\
default = [\"commercial-face\"]
speaker = [\"dep:diaric\"]
# Requires a commercial licence from the weights' author.
commercial-face = [\"dep:facelib\", \"speaker\"]
";

  #[test]
  fn a_default_that_pulls_in_a_commercial_gate_is_visible_to_the_reader() {
    let closure = feature_closure(LEAKY_FEATURES, "default");
    assert_eq!(
      closure,
      features(&["default", "commercial-face", "speaker"])
    );
  }

  #[test]
  fn the_leaky_manifest_reds_on_direction_two() {
    let closure = feature_closure(LEAKY_FEATURES, "default");
    let rows = [row(
      "a/w.bin",
      "vendor/one",
      "commercial-face",
      CLEAR,
      RESTRICTED,
    )];
    let failures = research_only_reachable(&rows, &closure);
    assert_eq!(failures.len(), 1, "{failures:?}");
    assert!(
      failures[0].contains("reachable from `default`"),
      "{failures:?}"
    );
  }
}
