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
//! **And the CHECKS have to be keyed the same way.** A table keyed by artifact
//! whose coverage check compares repository NAMES is the same defect one level
//! up: one row over a repository made every other file that repository stages
//! invisible, and the check passed while `openai/whisper-tiny` staged three
//! files and this table carried one. Every predicate below therefore
//! reconciles against a REPOSITORY FACT — the lock's own selectors, the
//! `#[cfg(feature = ...)]` in the source tree, the manifest's feature graph,
//! the per-file SHA-256 pins — and never against a field of the row it is
//! checking. The row's own `gate`, `pin` and `loader` are claims, and each one
//! has a check whose job is to disbelieve it.
//!
//! # The three directions
//!
//! Modelled on `CHECKSUMLESS_KITS` in `tests/whisper/models_lock.rs`, which
//! this repository already uses precisely so an exemption cannot outlive its
//! cause. Red on all three of:
//!
//!   1. a file `MODELS_LOCK` stages with no licence row — and the reverse, a
//!      row naming a file no table stages
//!      ([`every_staged_file_has_a_licence_row_and_every_row_names_a_staged_file`]);
//!   2. an artifact this crate WIRES into a configuration its terms cannot
//!      carry. TWO clauses over two different sets of rows, because "the terms
//!      forbid it" and "nobody knows what the terms are" are not the same
//!      claim — and each clause reads the MANIFEST channel and the TEST
//!      channel, because a suite that runs against the bytes wires them
//!      without ever naming the module that manifests them:
//!      - the STRONG clause, research-only rows only — the loader's `#[cfg]`
//!        feature must be `commercial-` prefixed and must sit in no
//!        non-commercial feature's closure
//!        ([`no_research_only_artifact_is_wired_without_a_commercial_gate`]),
//!        and no target the manifest declares may compile a reference to the
//!        artifact unless its own `required-features` closure is a commercial
//!        opt-in
//!        ([`no_research_only_artifact_is_tested_without_a_commercial_gate`]);
//!      - the WIDE clause, research-only AND unresolved rows — whatever gates
//!        it, `default` must not enable that gate
//!        ([`no_ungranted_artifact_is_wired_into_default`]), and no target a
//!        bare `cargo test` already builds may reference it
//!        ([`no_ungranted_artifact_is_tested_under_default`]);
//!   3. a `commercial-`prefixed feature that gates only artifacts granted at
//!      both layers, or that no `#[cfg(feature = ...)]` in the tree names at
//!      all ([`every_commercial_feature_gates_an_artifact_with_no_shipping_grant`]).
//!
//! The third is the one people forget, and it is what keeps the table honest
//! as artifacts change: the day an upstream relicenses, the gate that was
//! protecting it becomes a gate protecting nothing, and it must be retired
//! rather than left standing as false reassurance.
//!
//! # What direction 2 guarantees, in the word that means it: WIRED
//!
//! **"Wired" is what THIS CRATE does with an artifact: MANIFESTED — a module
//! the tree compiles that names the bytes — STAGED by a `MODELS_LOCK` table,
//! and TESTED by gate suites that run against them.** So the guarantee
//! direction 2 makes is that **no research-only artifact is *wired* —
//! manifested, staged or tested — by a feature reachable from `default`**.
//!
//! **Three words, three derivations, one sentence each.** This section named
//! all three while the predicates below derived only the first, and a claim
//! over three channels checked in one is a claim that outruns its check:
//!
//!   - **MANIFESTED** is derived by [`loader_gates`] joined with
//!     [`feature_closures`] — the `#[cfg(feature = ...)]` the tree puts on the
//!     module that names the bytes, against every non-commercial feature whose
//!     closure reaches that gate ([`research_only_wired`],
//!     [`ungranted_wired_into_default`]).
//!   - **STAGED** is derived by the equality
//!     [`rows_whose_loader_is_not_their_kit`] enforces — the lock row that
//!     stages the bytes declares a `kit`, and that `kit` must BE the loader
//!     module whose `#[cfg]` the clauses above judge, so the gate they clear
//!     is the gate over the artifact the lock actually downloads.
//!   - **TESTED** is derived by [`research_only_tested`] and
//!     [`ungranted_tested_under_default`] — for every `[[test]]`, `[[bench]]`
//!     and `[[example]]` the manifest declares, the files the compiler reaches
//!     from its `path` with no `commercial-` feature on, searched for the
//!     artifact's own names, against the `required-features` that target
//!     claims.
//!
//! **The third one is a TRIPWIRE, and the sentence has to say so.** What it
//! searches is NAMES — the loader module's file and identifier, the lock's
//! `local-dir` and the staged directory's own name — written in the target's
//! source, in source it `include!`s, and in the text of a fixture it embeds
//! with `include_str!`/`include_bytes!`. That set is closed under the two
//! splicings whose spelling is exact, and under nothing else: a module a macro
//! generates, code a build script emits, an environment variable read at run
//! time, a path composed at run time out of separately harmless parts. It
//! fails closed — an unresolvable file, an unreadable `#[cfg]`, a `cfg_attr`
//! that could move or gate a module are all refusals — so what it can produce
//! is a false RED, never a false green. It is not a semantic proof that an
//! ordinary-feature suite cannot reach the bytes.
//!
//! **That proof is STAGING's, above.** The bytes are not in this repository,
//! and the only CI shard that stages `Models/facekit` — in ci.yml and in
//! coverage.yml alike — is the kit's own, the kit
//! [`rows_whose_loader_is_not_their_kit`] requires to BE the
//! `commercial-`gated loader module. A suite on a plain feature has nothing
//! on disk to open. TESTED is what catches a target that
//! reaches for them anyway — the day someone adds a second stager, or points
//! an ordinary suite at a directory a developer already has.
//!
//! None of the three reads a byte, and none of them claims to.
//!
//! **The residual, stated beside the guarantee rather than left implied.** It
//! is issue #138 §8's, in the words these checks can honour: this register
//! does not, and cannot, govern bytes a caller loads through a generic door by
//! their own path. `coremlit::Model::load` is public, and so is every door
//! built on it — `FaceEmbedder::load(<the caller's path>,
//! FaceModel::new("data", "embedding", 512), …)` compiles under plain `face`
//! and loads whatever is at that path. **The licence of THOSE bytes is the
//! caller's**, between them and whoever published them. Nothing but a digest
//! would separate a caller's own commercially licensed ArcFace-shaped model
//! from InsightFace's, so the cure for an over-broad claim is a precise claim
//! and not a byte policy in a loader: coremlit's duty over its OWN registered
//! artifact is discharged by the `commercial-` feature's name, by its first
//! documented sentence, and by the fact that this repository hands nobody
//! those bytes.
//!
//! # Two axes, and why `Unresolved` needed the second
//!
//! [`Terms::forbids_commercial_use`] answers "do the terms forbid it", and
//! only [`Terms::ResearchOnly`] does. That is the right predicate for the
//! `commercial-` prefix and the WRONG one for what `default` may ship — which
//! is how [`Terms::Unresolved`] came to be invisible to directions 2 and 3
//! while its own doc said it is "a row that no shipping claim may rest on".
//! The sentence was true and unenforced: a row shaped exactly like
//! `redimnet/redimnet_b5.mlmodelc` under `default = ["identity"]` left every
//! check in this file green.
//!
//! So there is a second axis. [`Terms::permits_a_shipping_claim`] asks whether
//! a grant exists for a claim to rest on at all: `Permissive` and
//! `Attribution` yes, `ResearchOnly` and `Unresolved` no — for opposite
//! reasons, which [`withheld_because`] keeps in different words so no failure
//! message asserts a prohibition this repository has not found. Both axes are
//! exhaustive `match`es rather than `matches!(…)`, which is the actual root
//! cause rather than a style note: a fall-through default classified
//! `Unresolved` instead of an author doing it, and a fifth variant now cannot
//! be classified by accident.
//!
//! **What this deliberately does NOT do is extend the `commercial-` prefix
//! rule to unresolved rows**, and that is a decision with evidence rather than
//! an omission. Two things forbid it. The prefix's own documentation rule
//! ([`every_commercial_feature_says_it_requires_a_commercial_licence_first`])
//! demands a first sentence saying a commercial licence is REQUIRED, which
//! over an unresolved row asserts exactly the thing the row says nobody has
//! established. And the scope is not one artifact: NINETEEN rows here carry an
//! unresolved CORPUS layer — every `whisper`, `siglip`, `clap` and `ced` row
//! and four `speaker` ones — because `NOTICE` documents the weights layer
//! throughout and the corpus layer nowhere. That count is not an estimate:
//! [`the_tables_verdict_census_is_what_this_file_says_it_is`] pins it, so an
//! argument resting on it cannot go quietly out of date. A prefix rule keyed on
//! "unresolved" would rename most of this crate's public feature surface on
//! the strength of records this repository has not finished writing.
//! `default`-reachability is the rule the evidence supports, and it leaves
//! `identity` a plain feature.
//!
//! # What these checks bind, and why the falsifiers stay anyway
//!
//! **This section used to say that no row was research-only and that no
//! `commercial-`prefixed feature existed.** `facekit/w600k_r50.mlmodelc` ended
//! both: InsightFace's zoo publishes those weights "for non-commercial
//! research purposes only" and WebFace600K is a signed research-only
//! agreement, so it is research-only at BOTH layers, and it rides
//! `commercial-face-arcface`. Directions 2 and 3 now bind an artifact instead
//! of standing as tripwires — the STRONG clause has one row in scope, and
//! direction 3 has one feature to keep honest.
//!
//! One row is one row, so the hermetic falsifiers stay and are still the
//! reason to believe these predicates: `falsifiers::*` drives every one of
//! them over doctored input, runs everywhere with no models and no repository
//! files, and fails if a predicate stops detecting the thing it exists to
//! detect. What changed is that a green run is now evidence about a real
//! artifact as well as about the mechanism.
//!
//! Direction 2's WIDE clause has always had rows in scope, and now has
//! twenty-one: the nineteen with an unresolved corpus layer,
//! `redimnet/redimnet_b5.mlmodelc` whose weights layer is unresolved, and the
//! face row, which is in scope for the stronger reason. What makes it pass is
//! not an empty set but two live facts — `default = []`, and a
//! `#[cfg(feature = ...)]` on every one of those twenty-one loaders. Remove
//! the `identity` gate from `src/audio/mod.rs`, or put a kit feature into
//! `default`, and it reds against the real table with no doctoring at all.
//!
//! **And that last claim is worth exactly as much as the manifest reader
//! behind it.** It was first checked by writing one mutation — `default =
//! ["identity"]`, in the one formatting it happened to be typed in — and the
//! reader it was checked against was hand-rolled: it skipped indented lines,
//! split on the first `=`, and pulled DOUBLE-quoted runs out of the value. Six
//! spellings Cargo obeys defeated all three steps (an indented key, a literal
//! `'…'` string, a quoted key, a `#` comment carrying `]` inside a multi-line
//! array, a `features.default` dotted key, a `[ features ]` header), and under
//! every one of them `default`'s closure came back as `{"default"}` and the
//! clause stayed GREEN on a manifest that ships the bytes. The reader is now
//! [`declared_features`], which is the real `toml` parser and fails closed;
//! `falsifiers::the_reader_sees_default_under_every_valid_spelling` and
//! `falsifiers::direction_two_reds_from_default_under_every_valid_spelling`
//! carry all six spellings and report every one that regresses in a single
//! run, and
//! `falsifiers::an_undecodable_manifest_panics_rather_than_reading_as_empty`
//! pins the fail-closed half. So the claim above now reads: it reds against
//! the real table for every spelling of `default` the manifest can be written
//! in — which is what it always meant, and not what it had been measured
//! against.
//!
//! What "cannot fire" does NOT mean is "reads nothing". Directions 2 and 3
//! bind live data today even though nothing can trip them: the gate every row
//! runs on is read out of the tree's `#[cfg(feature = ...)]` by
//! [`loader_gates`], the closures out of the manifest by [`feature_closures`],
//! and the set of features any `#[cfg]` in `src/` actually names by
//! [`cfg_features_in_source`]. Break the loader gating and
//! [`every_rows_gate_matches_the_cfg_that_guards_its_loader`] goes red now, on
//! today's clean table. Direction 1, the SHA-256 pin cross-check, the
//! pin-ownership rule and the same-bytes-same-terms rule bind live data too.
//!
//! # What this file can and cannot see
//!
//! `MODELS_LOCK` selects in two shapes, and direction 1 treats them
//! differently because they carry different amounts of truth:
//!
//!   - `files = "a b c"` NAMES every file the table stages, so the check is an
//!     exact bijection — every listed file needs a row, every row must be one
//!     of the listed files. This is the shape that was passing vacuously.
//!   - `include = "<glob>"` names a PATTERN; the file list exists only after a
//!     download, and these checks are hermetic. A row on such a table is
//!     reconciled against the pattern (rows must be selected by it) and
//!     against the repository's own per-file SHA-256 manifests, which is what
//!     makes a row keyed on `weights/weight.bin` cover the whole `.mlmodelc`
//!     instead of the one file it keys on.
//!
//! The residue is real and named rather than implied. A `.mlmodelc` this
//! repository pins no manifest for cannot be keyed on bytes — the six
//! FluidInference bundles nothing else publishes are in that position — so
//! each is a BUNDLE row carrying [`Key::Unmanifested`] and the reason. They
//! are rows, with terms, gated, and covered by directions 2 and 3; what they
//! lack is a byte identity, and
//! [`unpinned_rows_exist_only_where_the_lock_pins_a_moving_revision`] ties
//! that exemption to its cause in both directions. Files a glob stages that
//! are not model artifacts at all (`CHECKSUMS.sha256`, the `.mlpackage`
//! sources whose `weights/weight.bin` the upstream's own checksum file lists
//! as byte-identical to the compiled bundle's) carry no rows and are the one
//! gap this file still cannot close hermetically.
//!
//! What stops "every bundle a glob stages has a row" from resting on nobody
//! having forgotten one is a SECOND enumeration, written for an unrelated
//! reason: `tests/fp16_guards.rs` pins guard sites per bundle, and
//! [`every_fp16_pinned_bundle_under_a_staged_vendor_has_a_licence_row`] refuses
//! any path in that roster which sits under a staged vendor directory and has
//! no row here. It is partial — a bundle with no guard sites appears in neither
//! enumeration — and it is a repository fact rather than a restatement of this
//! table, which is the only kind of check worth having.
//!
//! # EVIDENCE
//!
//! Every verdict below that rests on an upstream statement rather than on
//! `NOTICE` cites it here, pinned. A licence read once and not pinned is a
//! licence somebody has to read again.
//!
//!   - **openai/whisper-tiny** — HF repo, revision
//!     `169d4a4341b33bc18d8881c4b69c2e104e1cc0af`, declares `apache-2.0`. NOT
//!     MIT: the `openai/whisper` CODE repository is MIT, which is what
//!     `NOTICE` section 3 records for this chain.
//!   - **argmaxinc/whisperkit-coreml** — HF repo, revision
//!     `0f63a7800b00dd0226abd051b906c246e1907482`, declares `mit`.
//!   - **ibm-granite/granite-embedding-97m-multilingual-r2** — HF repo,
//!     revision `835ad14087e140460703cf0fae09f97d469d65c2`, declares
//!     `apache-2.0`; its model card's Data Collection section states "All
//!     training data is sourced under permissive, commercial-friendly
//!     licenses, making Granite Embedding R2 suitable for unrestricted
//!     enterprise deployment", over four named source classes and a stated
//!     data-clearance process.
//!   - **wenet-e2e/wespeaker** — `docs/pretrained.md`, section "Model
//!     License", at commit `c28dfb71f557a7eee05be164edce2577bf8708f8`: "The
//!     pretrained model in WeNet follows the license of it's corresponding
//!     dataset. For example, the pretrained model on VoxCeleb follows
//!     `Creative Commons Attribution 4.0 International License.`". There is no
//!     `docs/model_license.md` in that repository. The rule is stated once and
//!     worked through for VoxCeleb ONLY: the same page ships a same-named
//!     `cnceleb_resnet34_LM`, and CN-Celeb's terms are stated nowhere.
//!   - **VoxCeleb** — `https://mm.kaist.ac.kr/datasets/voxceleb/` (the URL
//!     WeSpeaker cites), retrieved 2026-09-01: "The VoxCeleb dataset is
//!     available to download for research purposes under a Creative Commons
//!     Attribution 4.0 International License. The copyright remains with the
//!     original owners of the video." The canonical Oxford VGG page
//!     (`https://www.robots.ox.ac.uk/~vgg/data/voxceleb/`) states no licence
//!     at all as of the same date, which is why the research-purposes wording
//!     is carried as a restriction rather than smoothed away.
//!   - **VoxLingua107** — `https://cs.taltech.ee/staff/tanel.alumae/data/voxlingua107/`,
//!     retrieved 2026-09-01, section "License and copyright": "The
//!     VoxLingua107 dataset is distributed under the Creative Commons
//!     Attribution 4.0 International License. The copyright remains with the
//!     original owners of the video." Corroborated by the Wayback snapshot
//!     `web.archive.org/web/20250624193952/https://bark.phon.ioc.ee/voxlingua107/`;
//!     the `bark.phon.ioc.ee` host that `NOTICE` and the literature cite is
//!     unreachable (connection reset) as of 2026-09-01.
//!   - **speechbrain/lang-id-voxlingua107-ecapa** — HF repo, revision
//!     `0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9`, declares `apache-2.0`.
//!   - **aufklarer/SpeechBrain-ECAPA-VoxLingua107-21M-CoreML** — HF repo,
//!     revision `2aa4d715a79e410d5f9aa32bd7a4fc9225bf9eb0` (the revision
//!     `MODELS_LOCK` pins), declares `apache-2.0`.
//!   - **AudioSet** — `https://research.google.com/audioset/download.html`,
//!     retrieved 2026-09-01: "The dataset is made available by Google Inc.
//!     under a Creative Commons Attribution 4.0 International (CC BY 4.0)
//!     license, while the ontology is available under a Creative Commons
//!     Attribution-ShareAlike 4.0 International (CC BY-SA 4.0) license." Note
//!     the direction: DATASET CC-BY-4.0, ONTOLOGY CC-BY-SA-4.0. Neither
//!     licenses the audio, which is YouTube media Google never redistributed.
//!   - **LAION-AI/CLAP** — `README.md` at commit
//!     `f14f288e5c9d2c7b7177b63512d0ba84f3ebf322`: "Due to copyright reasons,
//!     we cannot release the dataset we train this model on", and "Because
//!     most of the dataset has copyright restriction, unfortunatly we cannot
//!     directly share other preprocessed datasets."
//!   - **pyannote/segmentation-3.0** — HF repo, revision
//!     `e66f3d3b9eb0873085418a7b813d3b369bf160bb`, declares `mit` via the HF
//!     API; the card body is GATED (accessing the files requires accepting
//!     conditions), so its training-data list — AISHELL, AliMeeting, AMI,
//!     AVA-AVD, DIHARD, Ego4D, MSDWild, REPERE, VoxConverse, no terms stated
//!     for any — is recorded from the public model page on 2026-09-01 rather
//!     than pinned to that revision.
//!   - **InsightFace model zoo** — `deepinsight/insightface`, the
//!     `model_zoo/README.md` and the Python package's own model-zoo page, at
//!     revision `ffa12d315041c0505b077c7ff057ca914bb8dc7e` (the commit
//!     `conversion/face` pins for `face_align.py` and `ArcFaceONNX`): "**ALL**
//!     models are available for non-commercial research purposes only."
//!     `buffalo_l` — whose `w600k_r50` member this repository converts — is one
//!     of its packaged models. No commercial licence is offered anywhere in
//!     that repository or on its release pages; issue #115's census looked and
//!     found none to buy.
//!   - **WebFace260M / WebFace600K** — `https://www.face-benchmark.org/`, the
//!     corpus `w600k_r50` is trained on (`600K` is its 600 000-identity
//!     subset), released under a signed licence agreement confining it to
//!     non-commercial academic research. This is the layer that would
//!     disqualify the shipping path even if a commercial grant over the
//!     weights appeared — which is exactly why this table asks the two
//!     questions separately, and it is why issue #115's census ended with no
//!     shippable candidate rather than with a purchase order.
//!   - **InsightFace's recognition demo** — `web-demos/src_recognition/main.py`
//!     at commit `f8aa2c17e18044a86bbfa04be40e00cd2ff40a4f` (sha256
//!     `24a94180…9509`). Not a licence: it is where the `0.28` / `0.20`
//!     operating point `tests/face/known_pairs.rs` gates on comes from, pinned
//!     here so a threshold taken from upstream cannot later be mistaken for one
//!     this repository fitted to its own fixtures.
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

/// One licence layer's terms, in the form two rows over identical bytes can
/// actually be compared on.
///
/// Prose alone cannot be compared, and a four-way verdict class cannot either:
/// MIT and Apache-2.0 are both `Permissive` while imposing different notice
/// obligations, and two `ResearchOnly` artifacts can forbid materially
/// different things. So the canonical identifier and the obligations are
/// FIELDS — [`contradictory_terms`] reads them — and the prose is the reading
/// laid on top of them rather than the thing being compared.
#[derive(Debug, Clone, Copy)]
struct Statement {
  /// The canonical licence identifier: the SPDX id where one governs, or `""`
  /// where the layer is [`Terms::Unresolved`] and no identifier has been
  /// established. Compared VERBATIM across rows, so it is written in SPDX
  /// spelling (`Apache-2.0`, `CC-BY-4.0`) and nothing else.
  licence: &'static str,
  /// What the identifier alone does not carry: the obligations and
  /// prohibitions that travel with these bytes. Compared as a SET across rows,
  /// so two rows over one SHA-256 that record different restrictions are a
  /// contradiction even when they agree on the identifier.
  restrictions: &'static [&'static str],
  /// The reading this repository has, and where it comes from.
  detail: &'static str,
}

/// The obligation a permissive licence imposes on a binary that ships the
/// bytes, and the only one: reproduce the notice.
const RETAIN_NOTICE: &[&str] = &["retain-copyright-and-licence-notice"];

/// CC-BY-4.0's condition. Commercial use is permitted, but crediting the
/// author is a CONDITION of the grant, so shipping without the credit is
/// infringement rather than impoliteness.
const CREDIT_AUTHOR: &[&str] = &[
  "retain-copyright-and-licence-notice",
  "credit-the-author-in-the-product",
];

/// CC-BY-4.0 as VoxCeleb's own distributor states it — the grant, plus the two
/// things the identifier does not carry. The download page says the dataset is
/// "available to download for research purposes under a Creative Commons
/// Attribution 4.0 International License", and that "the copyright remains with
/// the original owners of the video". CC-BY-4.0 permits commercial use; the
/// research-purposes wording and the retained third-party copyright are the
/// tension a shipping decision has to be taken with its eyes open, so they are
/// recorded as restrictions rather than smoothed into the identifier.
const CREDIT_AUTHOR_VOXCELEB: &[&str] = &[
  "retain-copyright-and-licence-notice",
  "credit-the-author-in-the-product",
  "upstream-states-for-research-purposes-on-the-download-page",
  "third-party-copyright-retained-in-the-source-videos",
];

/// CC-BY-4.0 over a scraped-video corpus: the grant, the retained third-party
/// copyright, and the take-down policy the distributor operates — under which
/// the corpus a model was trained on is not guaranteed to stay the corpus that
/// is distributed.
const CREDIT_AUTHOR_SCRAPED: &[&str] = &[
  "retain-copyright-and-licence-notice",
  "credit-the-author-in-the-product",
  "third-party-copyright-retained-in-the-source-videos",
  "upstream-operates-a-notice-and-take-down-policy",
];

/// The identifier for a layer governed by a MIXTURE that the upstream asserts
/// is uniformly permissive without itemising it.
///
/// Not an SPDX id, because no single licence governs — and deliberately not
/// rounded to one, because the row would then claim more than the upstream
/// said. Compared verbatim like any other identifier, so a second row over the
/// same bytes claiming a real SPDX id is a contradiction, which is correct: a
/// vendor assertion and a licence grant are not the same evidence.
const PERMISSIVE_MIXTURE: &str = "permissive-mixture (vendor-asserted, not itemised)";

/// [`PERMISSIVE_MIXTURE`]'s obligations: the ordinary notice, plus the fact
/// that the permission rests on the vendor's word rather than on a licence
/// anybody can read.
const RETAIN_NOTICE_VENDOR_ASSERTED: &[&str] = &[
  "retain-copyright-and-licence-notice",
  "permission-rests-on-a-vendor-assertion-rather-than-a-per-source-licence-list",
];

/// The only legal payload for [`Terms::Unresolved`]: nothing is established,
/// so no obligation may be recorded — recording one would be an answer, and
/// the point of the variant is that there is not one yet.
const NOTHING_ESTABLISHED: &[&str] = &[];

/// Terms that confine use to non-commercial research and grant no
/// redistribution.
///
/// Not an SPDX identifier and deliberately not rounded to one: InsightFace's
/// model zoo and the WebFace260M agreement are two separate documents that
/// happen to impose the same two obligations, and neither is a licence
/// anybody can look up by name. Compared as a SET like every other
/// restrictions list, so a second row over these bytes recording a weaker
/// obligation is a contradiction rather than a rounding.
const RESEARCH_ONLY_NO_REDISTRIBUTION: &[&str] = &[
  "non-commercial-research-use-only",
  "no-commercial-use",
  "no-redistribution-of-the-weights",
];

/// What one licence layer permits, and the reading this repository has on it.
///
/// The verdict is the CLASS; [`Statement`] carries the identifier and the
/// obligations, because the class alone is too coarse to compare two rows on.
/// Every variant is a NEWTYPE of exactly one payload — the workspace house
/// rule (`no_enum_in_the_workspace_has_a_struct_shaped_or_multi_field_variant`).
#[derive(Debug, Clone, Copy)]
enum Terms {
  /// Commercial use permitted with no condition beyond retaining notices —
  /// MIT, Apache-2.0, BSD.
  Permissive(Statement),
  /// Commercial use permitted, but attribution is a CONDITION of it, so
  /// shipping without the notice is infringement rather than impoliteness.
  Attribution(Statement),
  /// **Disqualifying.** Forbids commercial use.
  ResearchOnly(Statement),
  /// Not established. The prose names the open QUESTION and where to go to
  /// answer it.
  ///
  /// Deliberately distinct from [`Terms::Permissive`]: rounding an unknown to
  /// "clear" is how a table stops being evidence. Unresolved is not
  /// disqualifying either — it is a row that no shipping claim may rest on
  /// until somebody resolves it.
  ///
  /// That last sentence is a CHECK, not a promise:
  /// [`Terms::permits_a_shipping_claim`] is false here, and
  /// [`no_ungranted_artifact_is_wired_into_default`] refuses to let such a
  /// row be wired into the one configuration this crate chooses for its
  /// consumers. It was prose alone once, and a row shaped exactly like
  /// `redimnet/redimnet_b5.mlmodelc` then sat under `default = ["identity"]`
  /// with every check in this file green.
  Unresolved(Statement),
}

impl Terms {
  /// A permissive layer: SPDX id, obligations, and the reading.
  const fn permissive(
    licence: &'static str,
    restrictions: &'static [&'static str],
    detail: &'static str,
  ) -> Self {
    Self::Permissive(Statement {
      licence,
      restrictions,
      detail,
    })
  }

  /// A layer whose commercial grant is conditional on attribution.
  const fn attribution(
    licence: &'static str,
    restrictions: &'static [&'static str],
    detail: &'static str,
  ) -> Self {
    Self::Attribution(Statement {
      licence,
      restrictions,
      detail,
    })
  }

  /// A layer that forbids the shipping path.
  const fn research_only(
    licence: &'static str,
    restrictions: &'static [&'static str],
    detail: &'static str,
  ) -> Self {
    Self::ResearchOnly(Statement {
      licence,
      restrictions,
      detail,
    })
  }

  /// A layer nobody has established. No identifier, no obligations — see
  /// [`NOTHING_ESTABLISHED`].
  const fn unresolved(detail: &'static str) -> Self {
    Self::Unresolved(Statement {
      licence: "",
      restrictions: NOTHING_ESTABLISHED,
      detail,
    })
  }

  /// The verdict class.
  ///
  /// NOT what two rows over identical bytes are compared on — see
  /// [`Terms::effective`], and the finding that four coarse strings let
  /// "MIT" and "Apache-2.0" over one SHA-256 read as agreement.
  const fn verdict(self) -> &'static str {
    match self {
      Self::Permissive(_) => "permissive",
      Self::Attribution(_) => "attribution-required",
      Self::ResearchOnly(_) => "research-only",
      Self::Unresolved(_) => "unresolved",
    }
  }

  /// **Axis one.** Whether these terms forbid the shipping path outright.
  ///
  /// The predicate the `commercial-` prefix hangs on, and it stays narrow on
  /// purpose: [`Terms::Unresolved`] is NOT a prohibition, and a feature whose
  /// documented first sentence must say a commercial licence is required
  /// cannot honestly gate an artifact for which nobody has found one. What
  /// `default` may ship is a different question — see
  /// [`Self::permits_a_shipping_claim`].
  ///
  /// Written as an exhaustive `match` rather than `matches!(…)`, which is not
  /// style. A fall-through default is what classified `Unresolved` here
  /// instead of an author, and made it invisible to directions 2 and 3 while
  /// its own doc said no shipping claim may rest on it. A fifth variant now
  /// cannot be classified by accident: the compiler asks.
  const fn forbids_commercial_use(self) -> bool {
    match self {
      Self::ResearchOnly(_) => true,
      Self::Permissive(_) | Self::Attribution(_) | Self::Unresolved(_) => false,
    }
  }

  /// **Axis two.** Whether there is a grant for a shipping claim to rest on.
  ///
  /// Not the negation of [`Self::forbids_commercial_use`], and the difference
  /// is the whole point: `ResearchOnly` says the terms are known and they
  /// forbid it, `Unresolved` says nobody knows what the terms are. Both leave
  /// a shipping claim with nothing to stand on; only one of them is a finding
  /// of prohibition, and [`withheld_because`] keeps the two apart in the words
  /// a failure message uses.
  ///
  /// Exhaustive for the same reason as the axis above.
  const fn permits_a_shipping_claim(self) -> bool {
    match self {
      Self::Permissive(_) | Self::Attribution(_) => true,
      Self::ResearchOnly(_) | Self::Unresolved(_) => false,
    }
  }

  /// The structured payload.
  const fn statement(self) -> Statement {
    match self {
      Self::Permissive(s) | Self::Attribution(s) | Self::ResearchOnly(s) | Self::Unresolved(s) => s,
    }
  }

  /// The prose payload.
  const fn detail(self) -> &'static str {
    self.statement().detail
  }

  /// The canonical licence identifier.
  const fn licence(self) -> &'static str {
    self.statement().licence
  }

  /// The obligations, as a set.
  fn restrictions(self) -> BTreeSet<&'static str> {
    self.statement().restrictions.iter().copied().collect()
  }

  /// **What two rows over identical bytes must agree on**: the class, the
  /// canonical identifier, and the obligation set.
  ///
  /// Comparing the class alone is what let identical bytes pass while one row
  /// called them MIT and the other Apache-2.0, and what let two research-only
  /// rows with different redistribution restrictions read as agreement.
  fn effective(self) -> (&'static str, &'static str, BTreeSet<&'static str>) {
    (self.verdict(), self.licence(), self.restrictions())
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
  /// The lock pins an immutable commit, so the bytes ARE determined — but this
  /// repository holds no per-file SHA-256 manifest for them, so the row has no
  /// hash to key on. Payload: why, and what would close it.
  ///
  /// A different exemption from [`Key::Unpinned`] and it must stay different:
  /// unpinned means "nobody can name these bytes", unmanifested means "these
  /// bytes are named by the lock and this repository never wrote the hash
  /// down". Legal ONLY on a table whose selector is a GLOB — an explicit
  /// `files` list names every file it stages, so a row on one can always be
  /// reconciled without a manifest.
  Unmanifested(&'static str),
}

/// One staged file, and what its bytes permit.
struct Artifact {
  /// Path under `Models/`, exactly as a `model-tests` shard stages it.
  file: &'static str,
  /// The bytes' identity — see [`Key`].
  key: Key,
  /// `<crate-relative source>::<identifier>` where this repository ALREADY
  /// pins those bytes, or `""` for a row that has no hash to pin.
  ///
  /// The licence attaches to bytes, so a hash copied here and never checked
  /// again is a hash that goes stale the first time an artifact is
  /// re-converted. This field is what stops that: the identifier names a
  /// `const` or a `fn` in the tree, and the SHA-256 in that pin has to be the
  /// one in this row. [`every_pin_locator_belongs_to_the_kit_and_bundle_it_is_read_for`]
  /// is what stops the locator from being ANY pin that happens to hold a
  /// `weights/weight.bin` key.
  pin: &'static str,
  /// The `MODELS_LOCK` table that stages the file.
  staged_by: &'static str,
  /// `<crate-relative source>::<module>` — the module declaration whose
  /// `#[cfg(feature = ...)]` decides whether the shipping path can load this
  /// artifact at all.
  ///
  /// This is the field [`Artifact::gate`] is CHECKED AGAINST. The gate a row
  /// claims is worth nothing on its own: the question direction 2 asks is
  /// "which cargo features make this artifact loadable", and only the tree
  /// answers it. [`loader_gates`] reads the answer, and
  /// [`every_rows_gate_matches_the_cfg_that_guards_its_loader`] refuses a row
  /// whose claim and whose tree disagree.
  loader: &'static str,
  /// The cargo feature a caller must enable before the shipping path can load
  /// it — the row's CLAIM, reconciled against [`Artifact::loader`]. Research-only
  /// artifacts must be gated by a `commercial-` feature; see
  /// [`COMMERCIAL_PREFIX`].
  gate: &'static str,
  /// Terms on the weight bytes themselves.
  weights: Terms,
  /// Terms on the data the weights were TRAINED ON.
  ///
  /// A different question from [`Artifact::weights`], and the one nearly every
  /// model fails: Apache-2.0 weights trained on a corpus licensed for research
  /// only are still research only. The two layers have two different sources,
  /// and a layer stays [`Terms::Unresolved`] only while its own source is
  /// silent — `NOTICE` documenting one layer says nothing about the other.
  corpus: Terms,
  /// Where the two verdicts above come from — `NOTICE` for a layer this
  /// repository already recorded, and a PINNED upstream revision for a layer
  /// resolved from the upstream's own statement.
  source: &'static str,
}

impl Artifact {
  /// The first layer for which `holds`, and its terms.
  ///
  /// Weights before corpus, so a message points at the document a reader can
  /// go and re-read first. An artifact is only as shippable as its least
  /// permissive layer, so BOTH are asked and either one answering is enough.
  fn layer_where(&self, holds: impl Fn(Terms) -> bool) -> Option<(&'static str, Terms)> {
    if holds(self.weights) {
      Some(("weights", self.weights))
    } else if holds(self.corpus) {
      Some(("training corpus", self.corpus))
    } else {
      None
    }
  }

  /// Which layer disqualifies the artifact, when one does — the layer whose
  /// terms are established and FORBID commercial use.
  fn disqualifying_layer(&self) -> Option<&'static str> {
    self
      .layer_where(Terms::forbids_commercial_use)
      .map(|(layer, _)| layer)
  }

  /// Which layer leaves a shipping claim with nothing to rest on, when one
  /// does, and its terms — research-only OR unresolved.
  ///
  /// A STRICTLY WIDER question than [`Self::disqualifying_layer`], and the one
  /// `default`-reachability is checked on. Asking the narrow question there is
  /// what let an unresolved row sit in the default feature set with every
  /// check green.
  fn ungranted_layer(&self) -> Option<(&'static str, Terms)> {
    self.layer_where(|terms| !terms.permits_a_shipping_claim())
  }

  /// The row's SHA-256, or `None` when it has no hash to key on.
  const fn sha256(&self) -> Option<&'static str> {
    match self.key {
      Key::Sha256(hex) => Some(hex),
      Key::Unpinned(_) | Key::Unmanifested(_) => None,
    }
  }

  /// The `.mlmodelc` bundle this row's file sits in, path included, or `None`
  /// when the artifact is a loose file.
  fn bundle(&self) -> Option<&'static str> {
    if self.file.ends_with(".mlmodelc") {
      return Some(self.file);
    }
    self
      .file
      .split_once(".mlmodelc/")
      .map(|(head, _)| &self.file[..head.len() + ".mlmodelc".len()])
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

/// The openings a `commercial-` feature's first documented sentence may take,
/// normalised (see [`normalise_spelling`]).
///
/// **Begins-with, not contains.** A substring test passes
/// "This feature no longer requires a commercial license", which is the exact
/// reading the rule exists to prevent, and it passes
/// "Cleared for commercial use! This feature requires a commercial license"
/// because a `. `-only sentence splitter never sees the first sentence end.
/// The warning has to BE the opening, not appear somewhere inside it.
const COMMERCIAL_DOC_OPENINGS: &[&str] = &[
  "requires a commercial license",
  "this feature requires a commercial license",
  "enabling this feature requires a commercial license",
  "using this feature requires a commercial license",
];

/// Words that invert or suspend whatever sentence they appear in.
///
/// Checked across the WHOLE first sentence, matched as words rather than
/// substrings ("cannot" must not be found inside "notice"). A first sentence
/// that opens with the warning and then takes it back has not warned anybody.
const NEGATIONS: &[&str] = &[
  "no", "not", "never", "neither", "nor", "none", "without", "unless", "cannot", "cant", "dont",
  "doesnt", "isnt", "wont", "except",
];

/// Every artifact `MODELS_LOCK` stages that this repository pins by SHA-256,
/// plus the whisper artifacts nothing can pin, and what each one permits.
///
/// Seeded from what the repository actually stages today. Every SHA-256 is
/// copied from the pin named in the same row's [`Artifact::pin`] and checked
/// against it by [`every_rows_sha256_matches_the_pin_it_names`], so the two
/// cannot drift apart.
///
/// **Exactly one row here is research-only, and it is research-only at BOTH
/// layers**: `facekit/w600k_r50.mlmodelc`, InsightFace's `w600k_r50` on
/// WebFace600K. It is the only row in the table for which a document
/// affirmatively FORBIDS the shipping path rather than merely failing to
/// permit it, which is why it is the only one behind a `commercial-` feature.
///
/// Two other shapes are worth keeping straight beside it, because all three
/// are findings about this repository's records rather than a clean bill of
/// health:
///
///   - `redimnet/redimnet_b5.mlmodelc` is the register's only unresolved
///     WEIGHTS layer. Not an oversight here and not a gap in `NOTICE`:
///     `IDRnD/redimnet` genuinely grants nothing over the released
///     checkpoints, its MIT covering "the Software", so there is no document
///     to record.
///   - every OTHER unresolved layer is a CORPUS layer, because `NOTICE`
///     documents the weights layer throughout and the corpus layer nowhere.
const ARTIFACTS: &[Artifact] = &[
  // --- whisper -------------------------------------------------------------
  Artifact {
    file: "whisperkit-coreml/openai_whisper-tiny/MelSpectrogram.mlmodelc",
    key: Key::Unpinned(
      "`argmaxinc/whisperkit-coreml` is still on `revision = \"main\"` (MODELS_LOCK's LOUD \
       FOLLOW-UP), so no immutable byte identity exists to key on; the same reason puts the \
       `whisper` kit in CHECKSUMLESS_KITS. The row names the BUNDLE rather than a file inside it \
       because with no byte identity there is no precision to be had from naming one.",
    ),
    pin: "",
    staged_by: "argmaxinc/whisperkit-coreml",
    loader: "src/audio/mod.rs::whisper",
    gate: "whisper",
    weights: Terms::permissive(
      "MIT",
      RETAIN_NOTICE,
      "argmaxinc/whisperkit-coreml declares MIT on the artifact repository itself, and it is \
       WhisperKit's CoreML conversion (argmaxinc/WhisperKit, MIT) of OpenAI Whisper. Note the \
       chain is NOT MIT end to end: the openai/whisper CODE repository is MIT, but the \
       openai/whisper-tiny MODEL repository these weights convert declares apache-2.0. Both are \
       permissive; the tokenizer rows below record the model repository's own terms rather than \
       reading MIT across. Revisions pinned in the module doc's EVIDENCE section.",
    ),
    corpus: Terms::unresolved(
      "The mel front-end of the same conversion. Its filterbank is derived from the checkpoint's \
       own preprocessing constants, so it inherits the encoder row's open corpus question; \
       whether a filterbank carries anything of the corpus at all is a second question nobody \
       here has answered, and it does not close the first.",
    ),
    source: "NOTICE section 3; openai/whisper-tiny and argmaxinc/whisperkit-coreml (EVIDENCE, \
             module doc)",
  },
  Artifact {
    file: "whisperkit-coreml/openai_whisper-tiny/AudioEncoder.mlmodelc",
    key: Key::Unpinned(
      "`argmaxinc/whisperkit-coreml` is still on `revision = \"main\"` (MODELS_LOCK's LOUD \
       FOLLOW-UP), so no immutable byte identity exists to key on; the same reason puts the \
       `whisper` kit in CHECKSUMLESS_KITS. The row names the BUNDLE rather than a file inside it \
       because with no byte identity there is no precision to be had from naming one.",
    ),
    pin: "",
    staged_by: "argmaxinc/whisperkit-coreml",
    loader: "src/audio/mod.rs::whisper",
    gate: "whisper",
    weights: Terms::permissive(
      "MIT",
      RETAIN_NOTICE,
      "argmaxinc/whisperkit-coreml declares MIT on the artifact repository itself, and it is \
       WhisperKit's CoreML conversion (argmaxinc/WhisperKit, MIT) of OpenAI Whisper. Note the \
       chain is NOT MIT end to end: the openai/whisper CODE repository is MIT, but the \
       openai/whisper-tiny MODEL repository these weights convert declares apache-2.0. Both are \
       permissive; the tokenizer rows below record the model repository's own terms rather than \
       reading MIT across. Revisions pinned in the module doc's EVIDENCE section.",
    ),
    corpus: Terms::unresolved(
      "OpenAI has published no terms for the ~680 000 hours of web audio Whisper was trained on, \
       and it does not name the sources. Its model card's Training Data section says only that \
       the models are trained on audio \"collected from the internet\". NOTICE section 3 records \
       the weights only, and the model card revision pinned in the EVIDENCE section states \
       nothing further.",
    ),
    source: "NOTICE section 3; openai/whisper-tiny and argmaxinc/whisperkit-coreml (EVIDENCE, \
             module doc)",
  },
  Artifact {
    file: "whisperkit-coreml/openai_whisper-tiny/TextDecoder.mlmodelc",
    key: Key::Unpinned(
      "`argmaxinc/whisperkit-coreml` is still on `revision = \"main\"` (MODELS_LOCK's LOUD \
       FOLLOW-UP), so no immutable byte identity exists to key on; the same reason puts the \
       `whisper` kit in CHECKSUMLESS_KITS. The row names the BUNDLE rather than a file inside it \
       because with no byte identity there is no precision to be had from naming one.",
    ),
    pin: "",
    staged_by: "argmaxinc/whisperkit-coreml",
    loader: "src/audio/mod.rs::whisper",
    gate: "whisper",
    weights: Terms::permissive(
      "MIT",
      RETAIN_NOTICE,
      "argmaxinc/whisperkit-coreml declares MIT on the artifact repository itself, and it is \
       WhisperKit's CoreML conversion (argmaxinc/WhisperKit, MIT) of OpenAI Whisper. Note the \
       chain is NOT MIT end to end: the openai/whisper CODE repository is MIT, but the \
       openai/whisper-tiny MODEL repository these weights convert declares apache-2.0. Both are \
       permissive; the tokenizer rows below record the model repository's own terms rather than \
       reading MIT across. Revisions pinned in the module doc's EVIDENCE section.",
    ),
    corpus: Terms::unresolved(
      "Same undisclosed ~680 000-hour web-audio corpus as the encoder; see that row.",
    ),
    source: "NOTICE section 3; openai/whisper-tiny and argmaxinc/whisperkit-coreml (EVIDENCE, \
             module doc)",
  },
  Artifact {
    file: "tokenizers/whisper-tiny/tokenizer.json",
    key: Key::Unpinned(
      "`openai/whisper-tiny` is still on `revision = \"main\"` (MODELS_LOCK's LOUD FOLLOW-UP), so \
       no immutable byte identity exists to key on.",
    ),
    pin: "",
    staged_by: "openai/whisper-tiny",
    loader: "src/audio/mod.rs::whisper",
    gate: "whisper",
    weights: Terms::permissive(
      "Apache-2.0",
      RETAIN_NOTICE,
      "OpenAI's own tokenizer artifact for whisper-tiny. The openai/whisper-tiny repository declares \
       apache-2.0 — NOT the MIT of the openai/whisper code repository, which is what NOTICE \
       section 3 records for this chain. Both are permissive, so nothing about the shipping path \
       changes; the identifier does, and it is the identifier two rows over one SHA-256 are \
       compared on. Revision pinned in the module doc's EVIDENCE section.",
    ),
    corpus: Terms::unresolved(
      "The BPE vocabulary was fit on the same undisclosed corpus as the weights, so the corpus \
       layer is open for the same reason. A vocabulary carries no weights, which narrows the \
       exposure but does not close the question.",
    ),
    source: "NOTICE section 3",
  },
  Artifact {
    file: "tokenizers/whisper-tiny/tokenizer_config.json",
    key: Key::Unpinned(
      "`openai/whisper-tiny` is still on `revision = \"main\"` (MODELS_LOCK's LOUD FOLLOW-UP), so \
       no immutable byte identity exists to key on.",
    ),
    pin: "",
    staged_by: "openai/whisper-tiny",
    loader: "src/audio/mod.rs::whisper",
    gate: "whisper",
    weights: Terms::permissive(
      "Apache-2.0",
      RETAIN_NOTICE,
      "The tokenizer's configuration sidecar, staged by the same `files` list and under the same \
       declared apache-2.0 as the vocabulary it configures; see that row for why this chain is \
       not MIT.",
    ),
    corpus: Terms::unresolved(
      "Configuration derived from the same undisclosed corpus as the vocabulary; open for the \
       same reason and with the same narrowed exposure. See the `tokenizer.json` row.",
    ),
    source: "NOTICE section 3",
  },
  Artifact {
    file: "tokenizers/whisper-tiny/config.json",
    key: Key::Unpinned(
      "`openai/whisper-tiny` is still on `revision = \"main\"` (MODELS_LOCK's LOUD FOLLOW-UP), so \
       no immutable byte identity exists to key on.",
    ),
    pin: "",
    staged_by: "openai/whisper-tiny",
    loader: "src/audio/mod.rs::whisper",
    gate: "whisper",
    weights: Terms::permissive(
      "Apache-2.0",
      RETAIN_NOTICE,
      "The whisper-tiny model configuration, staged by the same `files` list and under the same \
       declared apache-2.0; see the `tokenizer.json` row for why this chain is not MIT. It \
       carries architecture hyperparameters, no weight values.",
    ),
    corpus: Terms::unresolved(
      "Architecture configuration states nothing about the corpus, and the corpus layer is open \
       for the whole whisper-tiny checkpoint. See the `tokenizer.json` row.",
    ),
    source: "NOTICE section 3",
  },
  // --- granite -------------------------------------------------------------
  Artifact {
    file: "embedkit-granite/granite-97m-multilingual-r2/granite_97m_512.mlmodelc/weights/weight.bin",
    key: Key::Sha256("276bc93c49a4f37ffefdfb2e10f7d7e1ef57db9027c7ad0d3f2e4160f81a79be"),
    pin: "tests/granite/model_io.rs::ARTIFACT_SHA256",
    staged_by: "FinDIT-Studio/embedkit-coreml",
    loader: "src/embeddings/mod.rs::granite",
    gate: "granite",
    weights: Terms::permissive(
      "Apache-2.0",
      RETAIN_NOTICE,
      "ibm-granite/granite-embedding-97m-multilingual-r2; the staged file is a format conversion \
       with unchanged weight VALUES, so the upstream terms govern.",
    ),
    corpus: Terms::permissive(
      PERMISSIVE_MIXTURE,
      RETAIN_NOTICE_VENDOR_ASSERTED,
      "IBM STATES the corpus terms, which is what this row used to miss: the model card's Data \
       Collection section opens \"All training data is sourced under permissive, \
       commercial-friendly licenses, making Granite Embedding R2 suitable for unrestricted \
       enterprise deployment\", and records a data-clearance process behind it. That is a \
       vendor ASSERTION over a mixture rather than a per-source licence list, and two of the \
       four named sources (web-scraped title-body pairs, IBM-internal data) are not \
       independently checkable — so the assertion is the restriction. Evidence pinned in the \
       module doc.",
    ),
    source: "NOTICE section 7a",
  },
  Artifact {
    file: "embedkit-granite/granite-97m-multilingual-r2/tokenizer.json",
    key: Key::Sha256("4f2842d568e2724370aec203652a42ac783c7937f8347a1a2cc7506d71f1582f"),
    pin: "src/embeddings/granite/mod.rs::TOKENIZER_SHA256_HEX",
    staged_by: "FinDIT-Studio/embedkit-coreml",
    loader: "src/embeddings/mod.rs::granite",
    gate: "granite",
    weights: Terms::permissive(
      "Apache-2.0",
      RETAIN_NOTICE,
      "The same terms as the model it indexes. Distributed WITH the artifact and read from disk, \
       so whoever redistributes the artifact directory redistributes it.",
    ),
    corpus: Terms::permissive(
      PERMISSIVE_MIXTURE,
      RETAIN_NOTICE_VENDOR_ASSERTED,
      "Fit on the same mixture as the weights, under the same vendor assertion; see that row.",
    ),
    source: "NOTICE section 7b",
  },
  // --- siglip --------------------------------------------------------------
  Artifact {
    file: "siglip2-naflex/siglip2-base-patch16-naflex-512/siglip2_vision_512.mlmodelc/weights/\
           weight.bin",
    key: Key::Sha256("31fc44e771553c5b28b7af6561b46650ce5e1e4711dfef9f471ed32d502077b6"),
    pin: "tests/siglip/model_io.rs::ARTIFACT_SHA256",
    staged_by: "FinDIT-Studio/siglip2-naflex-coreml",
    loader: "src/embeddings/mod.rs::siglip",
    gate: "siglip",
    weights: Terms::permissive(
      "Apache-2.0",
      RETAIN_NOTICE,
      "google/siglip2-base-patch16-naflex; the artifact repo declares apache-2.0 too. The graph \
       is RESTRUCTURED (the position-embedding resize is lifted host-side), which Apache-2.0 \
       permits with the change stated — NOTICE section 8a states it.",
    ),
    corpus: Terms::unresolved(
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
    loader: "src/embeddings/mod.rs::siglip",
    gate: "siglip",
    weights: Terms::permissive(
      "Apache-2.0",
      RETAIN_NOTICE,
      "The vision tower's twin, same checkpoint and same declared terms at both layers of the \
       conversion chain.",
    ),
    corpus: Terms::unresolved("Same unreleased WebLI corpus as the vision tower; see that row."),
    source: "NOTICE section 8a",
  },
  Artifact {
    file: "siglip2-naflex/siglip2-base-patch16-naflex-512/pos_embed_16x16x768.f32le.bin",
    key: Key::Sha256("3ba1ba032ad8d97e0a1afebf4513615fbfedb56f646c14dcdb83d3c228c12860"),
    pin: "tests/siglip/model_io.rs::SIDECAR_SHA256",
    staged_by: "FinDIT-Studio/siglip2-naflex-coreml",
    loader: "src/embeddings/mod.rs::siglip",
    gate: "siglip",
    weights: Terms::permissive(
      "Apache-2.0",
      RETAIN_NOTICE,
      "The base position grid — the checkpoint's `position_embedding.weight` reshaped 16x16x768, \
       little-endian f32. These are WEIGHT VALUES lifted out of the graph, not metadata, so they \
       carry the checkpoint's own terms and need their own row.",
    ),
    corpus: Terms::unresolved(
      "Same unreleased WebLI corpus as the towers these values were trained alongside; see the \
       vision row.",
    ),
    source: "NOTICE section 8a",
  },
  Artifact {
    file: "siglip2-naflex/siglip2-base-patch16-naflex-512/tokenizer.json",
    key: Key::Sha256("58a1696e79c9d97937389ed116f552a15c84811d7b8023918b86f4bc5775b1b0"),
    pin: "src/embeddings/siglip/text/mod.rs::TOKENIZER_SHA256_HEX",
    staged_by: "FinDIT-Studio/siglip2-naflex-coreml",
    loader: "src/embeddings/mod.rs::siglip",
    gate: "siglip",
    weights: Terms::permissive(
      "Apache-2.0",
      RETAIN_NOTICE,
      "The same terms as the model. The Gemma tokenizer as packaged with the SigLIP 2 checkpoint; \
       distributed WITH the artifact, not compiled into the crate.",
    ),
    corpus: Terms::unresolved("Same unreleased WebLI corpus as the weights rows."),
    source: "NOTICE section 8b",
  },
  // --- ced -----------------------------------------------------------------
  Artifact {
    file: "ced/ced-tiny/ced_tiny.mlmodelc/weights/weight.bin",
    key: Key::Sha256("5635cd9f932583105d1bf40bd07eb54e3f715a70d8319923cd0617a1dea3db01"),
    pin: "tests/ced/model_io.rs::TINY_SHA256",
    staged_by: "FinDIT-Studio/cedkit-coreml",
    loader: "src/audio/mod.rs::ced",
    gate: "ced",
    weights: Terms::permissive(
      "Apache-2.0",
      RETAIN_NOTICE,
      "mispeech/ced-tiny (Xiaomi); the CoreML graph is restructured from unchanged weight values, \
       and NOTICE section 9 states the changes.",
    ),
    corpus: Terms::unresolved(
      "CED is distilled on AudioSet, and the row this replaces had AudioSet's two licences the \
       WRONG WAY ROUND. Google's own download page states the DATASET (the labelled segment \
       CSVs) is CC-BY-4.0 and the ONTOLOGY is CC-BY-SA-4.0 — share-alike, a materially different \
       obligation. Neither covers the AUDIO: the segments are YouTube media Google never \
       redistributed, and neither licence addresses whether anything reaches the DISTILLED \
       weights. That last question is why this layer stays unresolved rather than becoming an \
       attribution row. Evidence pinned in the module doc.",
    ),
    source: "NOTICE section 9",
  },
  // --- speaker: the FluidInference base layer ------------------------------
  //
  // Seven bundles arrive from here and nothing else publishes them
  // (MODELS_LOCK's "layer 1 of 2" table). Only `wespeaker_v2.mlmodelc` carries
  // a per-file SHA-256 manifest in this repository, so the other six are
  // BUNDLE rows keyed `Key::Unmanifested` — present, described and gated, with
  // the missing manifest recorded as the reason they cannot be keyed on bytes.
  // Leaving them out entirely is what the repo-keyed coverage check used to
  // permit: one row over this table made the other six invisible.
  Artifact {
    file: "speakerkit/wespeaker_v2.mlmodelc/weights/weight.bin",
    key: Key::Sha256("34004f6798d35cad7071e2fdc67e63faaa782f53697e1cb49bcb452cf81ae151"),
    pin: "tests/speaker/model_io.rs::int8_wespeaker_matches_fluidinference_pinned_sha256",
    staged_by: "FluidInference/speaker-diarization-coreml",
    loader: "src/audio/mod.rs::speaker",
    gate: "speaker",
    weights: Terms::attribution(
      "CC-BY-4.0",
      CREDIT_AUTHOR_VOXCELEB,
      "The RETIRED int8 WeSpeaker embedder, kept for tests. NOTICE section 4 names no licence for \
       the WeSpeaker component and POINTS AT the toolkit's model licence — so the row is resolved \
       by going there and reading it. WeSpeaker's rule is that a pretrained model follows the \
       licence of its corpus, and it gives exactly one worked instance: VoxCeleb models follow \
       CC BY 4.0. PROVENANCE, because the toolkit publishes a same-named CNCeleb ResNet34_LM \
       whose terms it states nowhere: this repository's own parity oracle is \
       `wespeaker_resnet34_lm.onnx`, the English pyannote/FluidAudio diarization lineage, which \
       is the VoxCeleb `voxceleb_resnet34_LM` — the corpus-prefixed upstream name is what this \
       repository does not pin, and that is the residue on this row. Evidence in the module doc.",
    ),
    corpus: Terms::attribution(
      "CC-BY-4.0",
      CREDIT_AUTHOR_VOXCELEB,
      "VoxCeleb. WeSpeaker's own model-licence document places its VoxCeleb-trained pretrained \
       models under CC BY 4.0, which is a grant rather than a research-only restriction — so the \
       corpus layer does NOT disqualify the shipping path, but attribution is a condition of it. \
       Evidence pinned in the module doc's EVIDENCE section.",
    ),
    source: "NOTICE section 4; wenet-e2e/wespeaker model licence (EVIDENCE, module doc)",
  },
  Artifact {
    file: "speakerkit/wespeaker_int8.mlmodelc",
    key: Key::Unmanifested(
      "Byte-identical to `wespeaker_v2.mlmodelc` — `wespeaker_v2_and_wespeaker_int8_are_byte_\
       identical` in tests/speaker/model_io.rs walks both trees and compares every file — but \
       this repository writes no SHA-256 down under the int8 path itself. Keying this row on \
       wespeaker_v2's manifest would be a lookup by BUNDLE-RELATIVE NAME across two different \
       bundles, which is the repo-keyed mistake this table exists to refuse.",
    ),
    pin: "",
    staged_by: "FluidInference/speaker-diarization-coreml",
    loader: "src/audio/mod.rs::speaker",
    gate: "speaker",
    weights: Terms::attribution(
      "CC-BY-4.0",
      CREDIT_AUTHOR_VOXCELEB,
      "The same bytes as `wespeaker_v2.mlmodelc` under a second name, so necessarily the same \
       terms; see that row for the WeSpeaker model-licence chain and the provenance residue. \
       MODELS_LOCK's overlay table deliberately keeps this bundle FluidInference's rather than \
       taking the FinDIT re-palettization.",
    ),
    corpus: Terms::attribution(
      "CC-BY-4.0",
      CREDIT_AUTHOR_VOXCELEB,
      "VoxCeleb, on WeSpeaker's stated CC BY 4.0 terms for its VoxCeleb-trained models — the same \
       bytes and the same corpus as `wespeaker_v2.mlmodelc`; see that row.",
    ),
    source: "NOTICE section 4; wenet-e2e/wespeaker model licence (EVIDENCE, module doc)",
  },
  Artifact {
    file: "speakerkit/Segmentation.mlmodelc",
    key: Key::Unmanifested(
      "FluidInference's repo ships no CHECKSUMS.sha256 and this repository pins no per-file \
       manifest for this bundle: it is not a shipping candidate (tests/speaker/model_io.rs's \
       DECISION picks `pyannote_segmentation.mlmodelc`), so only tests/fp16_guards.rs touches \
       it, and that pins guard SITES rather than bytes. A per-file manifest here would close it.",
    ),
    pin: "",
    staged_by: "FluidInference/speaker-diarization-coreml",
    loader: "src/audio/mod.rs::speaker",
    gate: "speaker",
    weights: Terms::attribution(
      "CC-BY-4.0",
      CREDIT_AUTHOR,
      "The newer \"community-1\" conversion set (tests/speaker/model_io.rs, spec-vs-reality \
       delta 1), so the parent is pyannote/speaker-diarization-community-1 — the model NOTICE \
       section 4 records as CC-BY-4.0 and REQUIRING attribution, not the MIT segmentation-3.0 \
       that `pyannote_segmentation.mlmodelc` derives from. Two different parents behind two \
       similarly named bundles is exactly why this row exists.",
    ),
    corpus: Terms::unresolved(
      "The pyannote training mixture (AMI, DIHARD, VoxConverse and others), whose members carry \
       different terms and several of which are research-only. NOTICE section 4 records the \
       weights layer only, and the community-1 mixture is not published per-source.",
    ),
    source: "NOTICE section 4",
  },
  Artifact {
    file: "speakerkit/Embedding.mlmodelc",
    key: Key::Unmanifested(
      "The split-pipeline embedding backend, NOT targeted per spec section 2.4 and never loaded \
       by the shipping path; no per-file manifest is pinned for it anywhere in this repository. \
       A manifest here would close it.",
    ),
    pin: "",
    staged_by: "FluidInference/speaker-diarization-coreml",
    loader: "src/audio/mod.rs::speaker",
    gate: "speaker",
    weights: Terms::attribution(
      "CC-BY-4.0",
      CREDIT_AUTHOR_VOXCELEB,
      "The WeSpeaker embedder split into a frontend/backend pair, published in the same \
       conversion set, so the same CC BY 4.0 chain and the same provenance residue as \
       `wespeaker_v2.mlmodelc`; see that row.",
    ),
    corpus: Terms::attribution(
      "CC-BY-4.0",
      CREDIT_AUTHOR_VOXCELEB,
      "VoxCeleb, on WeSpeaker's stated CC BY 4.0 terms for its VoxCeleb-trained models; see the \
       `wespeaker_v2.mlmodelc` row for the evidence.",
    ),
    source: "NOTICE section 4; wenet-e2e/wespeaker model licence (EVIDENCE, module doc)",
  },
  Artifact {
    file: "speakerkit/FBank.mlmodelc",
    key: Key::Unmanifested(
      "The split-pipeline filterbank frontend, NOT targeted per spec section 2.4 and never loaded \
       by the shipping path; no per-file manifest is pinned for it anywhere in this repository. \
       A manifest here would close it.",
    ),
    pin: "",
    staged_by: "FluidInference/speaker-diarization-coreml",
    loader: "src/audio/mod.rs::speaker",
    gate: "speaker",
    weights: Terms::attribution(
      "CC-BY-4.0",
      CREDIT_AUTHOR_VOXCELEB,
      "The frontend half of the same split WeSpeaker pipeline as `Embedding.mlmodelc`, so the \
       same CC BY 4.0 chain governs it. Whether a mel frontend carries protectable weight values \
       at all is a separate question this repository has not answered; recording the stricter \
       answer costs an attribution line and risks nothing.",
    ),
    corpus: Terms::attribution(
      "CC-BY-4.0",
      CREDIT_AUTHOR_VOXCELEB,
      "VoxCeleb, on WeSpeaker's stated CC BY 4.0 terms for its VoxCeleb-trained models; see the \
       `wespeaker_v2.mlmodelc` row for the evidence.",
    ),
    source: "NOTICE section 4; wenet-e2e/wespeaker model licence (EVIDENCE, module doc)",
  },
  Artifact {
    file: "speakerkit/PLDA.mlmodelc",
    key: Key::Unmanifested(
      "The community-1 PLDA projection, deliberately UNLOADED — clustering stays in `diaric`, \
       which projects in f64 on the host (spec section 3 non-goal) — so nothing in this \
       repository pins its bytes; tests/fp16_guards.rs pins its guard sites and nothing else \
       reads it.",
    ),
    pin: "",
    staged_by: "FluidInference/speaker-diarization-coreml",
    loader: "src/audio/mod.rs::speaker",
    gate: "speaker",
    weights: Terms::attribution(
      "CC-BY-4.0",
      CREDIT_AUTHOR,
      "This is the artifact the CC-BY-4.0 in NOTICE section 4 actually belongs to: \
       pyannote/speaker-diarization-community-1, the PLDA `diaric` clusters through. Attribution \
       is a CONDITION of the commercial grant, which is why the section carries the citation \
       block a shipping product has to reproduce.",
    ),
    corpus: Terms::unresolved(
      "The pyannote community-1 training mixture, not published per-source; NOTICE section 4 \
       records the weights layer only. Same open question as the `Segmentation.mlmodelc` row.",
    ),
    source: "NOTICE section 4",
  },
  Artifact {
    file: "speakerkit/PldaRho.mlmodelc",
    key: Key::Unmanifested(
      "The rho companion to `PLDA.mlmodelc`, unloaded for the same reason and pinned by nothing \
       but tests/fp16_guards.rs's guard sites.",
    ),
    pin: "",
    staged_by: "FluidInference/speaker-diarization-coreml",
    loader: "src/audio/mod.rs::speaker",
    gate: "speaker",
    weights: Terms::attribution(
      "CC-BY-4.0",
      CREDIT_AUTHOR,
      "The second half of the same community-1 PLDA projection as `PLDA.mlmodelc` — same parent, \
       same CC-BY-4.0 grant, same attribution condition. See that row.",
    ),
    corpus: Terms::unresolved(
      "The pyannote community-1 training mixture, not published per-source; see the \
       `PLDA.mlmodelc` row.",
    ),
    source: "NOTICE section 4",
  },
  // --- speaker: the FinDIT-Studio overlay, the two SHIPPING artifacts ------
  Artifact {
    file: "speakerkit/pyannote_segmentation.mlmodelc/weights/weight.bin",
    key: Key::Sha256("0266f4ad4d843ecf31ef9220ad6b80616b3ec64a4404b64f3ea0371554e236ec"),
    pin: "tests/speaker/model_io.rs::fp16_safe_segmentation_matches_pinned_sha256",
    staged_by: "FinDIT-Studio/speakerkit-coreml",
    loader: "src/audio/mod.rs::speaker",
    gate: "speaker",
    weights: Terms::permissive(
      "MIT",
      RETAIN_NOTICE,
      "An issue-#15 re-conversion of pyannote/segmentation-3.0 (MIT) with fp16-survivable guards; \
       the artifact repo declares HF licence \"other\"/mixed-upstream, and NOTICE section 4 \
       records that the upstream MIT terms still govern because the weight values are the \
       upstream ones. The upstream repository is GATED — obtaining it requires accepting access \
       conditions — which is a condition on getting the bytes, not on the MIT grant over them, \
       and this repository fetches the re-conversion rather than the gated original.",
    ),
    corpus: Terms::unresolved(
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
    loader: "src/audio/mod.rs::speaker",
    gate: "speaker",
    weights: Terms::attribution(
      "CC-BY-4.0",
      CREDIT_AUTHOR_VOXCELEB,
      "The SHIPPING fp32 WeSpeaker embedder. It lands on CC-BY-4.0 by a DIFFERENT ROUTE from the \
       PLDA rows, and the distinction is the point: NOTICE section 4's CC-BY-4.0 belongs to \
       pyannote/speaker-diarization-community-1 and to FluidInference's parent pyannote model, \
       NOT to these embedder weights, and reading it across is still the mistake this row exists \
       to stop. What resolves this row is the document NOTICE section 4 POINTS AT — WeSpeaker's \
       own model licence, CC BY 4.0 for its VoxCeleb models. Same identifier, unrelated grant, \
       and the provenance residue on the `wespeaker_v2.mlmodelc` row applies here too.",
    ),
    corpus: Terms::attribution(
      "CC-BY-4.0",
      CREDIT_AUTHOR_VOXCELEB,
      "VoxCeleb, per the WeSpeaker toolkit's published recipes, and WeSpeaker's own model-licence \
       document places its VoxCeleb-trained models under CC BY 4.0. That is the CORPUS-layer \
       source — a different document from NOTICE section 4's weights-layer record, which is why \
       resolving one says nothing about the other. Evidence pinned in the module doc's EVIDENCE \
       section.",
    ),
    source: "NOTICE section 4; wenet-e2e/wespeaker model licence (EVIDENCE, module doc)",
  },
  // --- clap ----------------------------------------------------------------
  Artifact {
    file: "clapkit/clap_audio.mlmodelc/weights/weight.bin",
    key: Key::Sha256("723fe6aab7c4af1c671a210a35c289c67763bc6a7532b9df155a0c3fc0c3c9d7"),
    pin: "tests/clap/model_io.rs::clap_audio_artifacts_match_pinned_sha256",
    staged_by: "FinDIT-Studio/clapkit-coreml",
    loader: "src/embeddings/mod.rs::clap",
    gate: "clap",
    weights: Terms::attribution(
      "CC-BY-4.0",
      CREDIT_AUTHOR,
      "laion/clap-htsat-unfused. NOTICE section 6a records an upstream ambiguity — textclap's \
       MODELS.md treats the checkpoints as CC-BY-4.0, the HF card declares apache-2.0 — and BOTH \
       require attribution, so the STRICTER of the two governs here: CC-BY-4.0's credit is a \
       CONDITION of the grant where Apache-2.0's notice is not. The LAION citation in NOTICE \
       section 4's style must ship with any binary that bundles these weights.",
    ),
    corpus: Terms::unresolved(
      "LAION-Audio-630K. The upstream states the corpus terms and they are NEGATIVE: LAION-AI/CLAP's \
       README says \"Due to copyright reasons, we cannot release the dataset we train this model \
       on\" and that most of it \"has copyright restriction\" — only source links and captions \
       were published. What that reaches is the open question: the upstream does not say whether \
       the restriction travels to the DERIVED weights, and this is a stated restriction rather \
       than the silence the row used to record. Evidence pinned in the module doc.",
    ),
    source: "NOTICE section 6a; LAION-AI/CLAP README (EVIDENCE, module doc)",
  },
  Artifact {
    file: "clapkit/clap_audio_int8.mlmodelc/weights/weight.bin",
    key: Key::Sha256("b3a37ec5550dcdd6932b314b830275ebcba013748421e1a517760b9afeabafb8"),
    pin: "tests/clap/model_io.rs::clap_audio_int8_artifacts_match_pinned_sha256",
    staged_by: "FinDIT-Studio/clapkit-coreml",
    loader: "src/embeddings/mod.rs::clap",
    gate: "clap",
    weights: Terms::attribution(
      "CC-BY-4.0",
      CREDIT_AUTHOR,
      "A palettization of the fp16 audio tower — different bytes, same checkpoint, same stricter \
       CC-BY-4.0 reading; see the fp16 audio row.",
    ),
    corpus: Terms::unresolved(
      "Same stated LAION-Audio-630K copyright restriction as the fp16 audio tower, and the same \
       open question about whether it reaches derived weights; see that row.",
    ),
    source: "NOTICE section 6a; LAION-AI/CLAP README (EVIDENCE, module doc)",
  },
  Artifact {
    file: "clapkit/clap_text.mlmodelc/weights/weight.bin",
    key: Key::Sha256("7f4e15e9ccb0ffbc2341eec286e9d9934d3d3d8d6465dfddebed248bddc0e3dd"),
    pin: "tests/clap/text_model_io.rs::clap_text_artifacts_match_pinned_sha256",
    staged_by: "FinDIT-Studio/clapkit-coreml",
    loader: "src/embeddings/mod.rs::clap",
    gate: "clap",
    weights: Terms::attribution(
      "CC-BY-4.0",
      CREDIT_AUTHOR,
      "The audio tower's twin, same checkpoint and the same stricter CC-BY-4.0 reading of the \
       upstream ambiguity; see the fp16 audio row.",
    ),
    corpus: Terms::unresolved(
      "Same stated LAION-Audio-630K copyright restriction as the audio tower; see that row.",
    ),
    source: "NOTICE section 6a; LAION-AI/CLAP README (EVIDENCE, module doc)",
  },
  Artifact {
    file: "clapkit/clap_text_int8.mlmodelc/weights/weight.bin",
    key: Key::Sha256("f181a595cefce402335499c32ea2f9727ef334afea9c592a2eabebb4172350a0"),
    pin: "tests/clap/text_model_io.rs::clap_text_int8_artifacts_match_pinned_sha256",
    staged_by: "FinDIT-Studio/clapkit-coreml",
    loader: "src/embeddings/mod.rs::clap",
    gate: "clap",
    weights: Terms::attribution(
      "CC-BY-4.0",
      CREDIT_AUTHOR,
      "A palettization of the fp16 text tower — different bytes, same checkpoint, same stricter \
       CC-BY-4.0 reading; see the fp16 audio row.",
    ),
    corpus: Terms::unresolved(
      "Same stated LAION-Audio-630K copyright restriction as the fp16 text tower; see that row.",
    ),
    source: "NOTICE section 6a; LAION-AI/CLAP README (EVIDENCE, module doc)",
  },
  // --- lid -----------------------------------------------------------------
  Artifact {
    file: "lid/SpeechBrainECAPAVoxLingua107.mlmodelc/weights/weight.bin",
    key: Key::Sha256("81fbb61f6706c50e924a2ee2a4fc04e6408276df948117a1c6ac7675c23aac67"),
    pin: "tests/lid/common/mod.rs::ARTIFACT_SHA256",
    staged_by: "aufklarer/SpeechBrain-ECAPA-VoxLingua107-21M-CoreML",
    loader: "src/audio/mod.rs::lid",
    gate: "lid",
    weights: Terms::permissive(
      "Apache-2.0",
      RETAIN_NOTICE,
      "Apache-2.0 at both layers of the chain, each confirmed against the repository's own \
       declaration rather than inferred: speechbrain/lang-id-voxlingua107-ecapa upstream, and the \
       aufklarer CoreML export MODELS_LOCK stages. Revisions pinned in the module doc's EVIDENCE \
       section — the export's is the same commit MODELS_LOCK pins.",
    ),
    corpus: Terms::attribution(
      "CC-BY-4.0",
      CREDIT_AUTHOR_SCRAPED,
      "VoxLingua107, and its own distributor STATES the terms: \"The VoxLingua107 dataset is \
       distributed under the Creative Commons Attribution 4.0 International License. The \
       copyright remains with the original owners of the video.\" So the corpus layer is a \
       GRANT with an attribution condition, not an unknown — and the retained third-party \
       copyright and the published take-down policy are the two things the identifier alone does \
       not carry, which is why they are restrictions here. Evidence pinned in the module doc.",
    ),
    source: "NOTICE section 10a; VoxLingua107 distribution page (EVIDENCE, module doc)",
  },
  // --- identity ------------------------------------------------------------
  Artifact {
    file: "redimnet/redimnet_b5.mlmodelc/weights/weight.bin",
    key: Key::Sha256("1735fc68f4cdf10ad8bb56135da3bd8c0c83f6c3549ee8514f0346046f90a79b"),
    pin: "tests/identity/common/mod.rs::ARTIFACT_SHA256",
    staged_by: "FinDIT-Studio/redimnetkit-coreml",
    loader: "src/audio/mod.rs::identity",
    gate: "identity",
    weights: Terms::unresolved(
      "NO WRITTEN GRANT COVERS THESE BYTES, and that is a step DOWN in artifact-level clarity \
       from the incumbent rather than a step across. `IDRnD/redimnet` ships MIT, but the grant \
       is written over \"the Software\" — the model source — and neither that repository nor \
       `PalabraAI/redimnet2` extends it to the released `.pt` assets in writing. Compare the \
       row this sits beside: WeSpeaker's own model-licence document places its \
       VoxCeleb-trained pretrained models under CC-BY-4.0, an explicit weights grant with \
       attribution as a CONDITION, which is why `speakerkit/wespeaker.mlmodelc` is an \
       attribution row and this one is not. The corpus layer below is the binding constraint \
       and it is unchanged, so this does not disqualify the shipping path; what it does is \
       remove a written permission we previously had, and the register should show that as \
       `unresolved` rather than borrow the source licence's identifier for weight bytes it \
       does not name. Re-tagging an upstream CODE licence onto a weights artifact is the \
       exact conflation this campaign has already paid for once — `aufklarer/\
       ReDimNet2-B6-CoreML` declares `license: mit` over VoxBlink2-trained weights whose \
       corpus is CC-BY-NC-SA-4.0. It is also why the artifact repository MODELS_LOCK names is \
       PRIVATE: fetching our own conversion for our own CI is use, and publishing it openly \
       would have been redistribution under no grant.",
    ),
    corpus: Terms::attribution(
      "CC-BY-4.0",
      CREDIT_AUTHOR_VOXCELEB,
      "VoxCeleb2-dev, and NO NEW EXPOSURE: this is the same corpus lineage the incumbent \
       WeSpeaker embedder already carries, so the decision it needs has already been taken. \
       The `-vox2-` lineage is the only one usable here — the same upstream release publishes \
       `M-vb2+vox2+cnc-ft_mix.pt` and `S-vb2-ptn.pt` trained on VoxBlink2, whose distributor \
       states the CC-BY-NC-SA-4.0 term propagates to the trained model (\"The license of the \
       model is also CC BY-NC-SA 4.0, no commercial application is allowed\"). The conversion \
       recipe refuses any asset whose name is not `-vox2-` \
       (`conversion/redimnet/scripts/_redimnet_common.py::verify_asset_name`), so the \
       distinction is enforced at the point the bytes are loaded rather than remembered here.",
    ),
    source: "conversion/redimnet/README.md and LICENCE_ROW.md; IDRnD/redimnet LICENSE (MIT, over \
             \"the Software\"); wenet-e2e/wespeaker model licence; voxblink2.github.io",
  },
  // --- face (commercial-face-arcface) --------------------------------------
  //
  // THE ONLY ROW IN THIS TABLE THAT IS RESEARCH-ONLY, and it is research-only
  // at both layers. Everything the three directions were built for meets an
  // artifact here for the first time.
  Artifact {
    file: "facekit/w600k_r50.mlmodelc/weights/weight.bin",
    // MEASURED on the PUBLISHED bundle, not on the conversion run: the
    // `.mlmodelc` is produced by `xcrun coremlcompiler`, which the Python
    // toolchain does not pin, and the ReDimNet recipe measured which files
    // that affects — `model.mil`, `weights/weight.bin` and `metadata.json`
    // re-derive byte for byte, while both `coremldata.bin`s differ between two
    // compiles of the SAME .mlpackage. So this key is on the deterministic
    // half AND it is the published run's bytes, which is what a pin may name.
    key: Key::Sha256("aa08d7826a70f9bc237ea0532a5eec12cb83b8375148a1b0650f104cbb2ff492"),
    pin: "tests/face/arcface/mod.rs::ARTIFACT_SHA256",
    staged_by: "FinDIT-Studio/facekit-coreml",
    loader: "src/embeddings/face/mod.rs::arcface",
    gate: "commercial-face-arcface",
    weights: Terms::research_only(
      "InsightFace model licence (non-commercial research)",
      RESEARCH_ONLY_NO_REDISTRIBUTION,
      "InsightFace's model zoo states \"ALL models are available for non-commercial research \
       purposes only\", and `buffalo_l` — whose `w600k_r50` member this bundle is a conversion \
       of — is one of its packaged models. No commercial licence is offered for these weights: \
       issue #115's census could not find one to buy, and the owner's decision was to use them \
       for CI and development on the standing basis that this repository redistributes nothing. \
       A conversion does not lift the restriction — re-encoding a graph produces a derivative of \
       the weights, not a new work, so this bundle carries their terms exactly. This is the \
       register's FIRST research-only weights layer; every earlier restricted layer was a corpus \
       one, and every earlier weights layer was a grant or an open question.",
    ),
    corpus: Terms::research_only(
      "WebFace260M/WebFace600K licence agreement",
      RESEARCH_ONLY_NO_REDISTRIBUTION,
      "`w600k_r50` is trained on WebFace600K, the 600K-identity subset of WebFace260M, which is \
       released under a signed licence agreement confining it to non-commercial academic \
       research. This layer would disqualify the shipping path on its own even if a commercial \
       grant over the weights appeared, which is exactly why this register asks the two \
       questions separately — and it is the reason issue #115's census ended with no shippable \
       candidate at all rather than with a purchase order.",
    ),
    source: "conversion/face/README.md and LICENCE_ROW.md; InsightFace model zoo \
             (\"non-commercial research purposes only\"); WebFace260M licence agreement \
             (EVIDENCE, module doc)",
  },
];

// ---------------------------------------------------------------------------
// The three directions, as predicates over data
// ---------------------------------------------------------------------------
//
// Pure functions returning the failures they found, so the hermetic falsifiers
// below can drive exactly the same code the real-table checks do. A predicate
// only the happy path ever reaches is not a predicate.

/// **Direction 1, as it must be asked.** Every FILE the repository can name
/// under a staged table is covered by a row, and every row names a file its
/// own table actually stages.
///
/// The previous shape of this check compared MODELS_LOCK's repository NAMES
/// against `staged_by`, which meant one row over a table made every other file
/// that table stages invisible — the AuraFace lesson repeated at the mechanism
/// level, on a table that is keyed by artifact PRECISELY because repo-keying
/// gets individual files wrong. It passed while `openai/whisper-tiny` staged
/// three files and the table carried one.
///
/// So the reconciliation runs at file granularity, in both directions, against
/// what the lock literally says:
///
///   - a `files = "a b c"` table names every file it stages, so the check is
///     an exact bijection — every listed file covered, every row one of them;
///   - an `include = "<glob>"` table's file list only exists after a download,
///     so the enumeration comes from the repository's OWN per-file SHA-256
///     manifests: a row inside a `.mlmodelc` must name one, which is what
///     makes it cover the whole bundle instead of the single file it keys on.
fn unmatched_coverage(tables: &[StagedTable], rows: &[Covered<'_>]) -> Vec<String> {
  let staged: BTreeSet<&str> = tables.iter().map(|t| t.name.as_str()).collect();
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

  for table in tables {
    let mine: Vec<&Covered<'_>> = rows.iter().filter(|r| r.staged_by == table.name).collect();

    // Reverse: a row must name a path its own table's SELECTOR picks up.
    for row in &mine {
      let Some(tail) = table.table_relative(row.file) else {
        failures.push(format!(
          "{}: table {:?} stages into {:?}, so the row's path must start with {}/",
          row.file, table.name, table.vendor_dir, table.vendor_dir
        ));
        continue;
      };
      if !table.selects(tail) {
        failures.push(format!(
          "{}: table {:?} does not stage {tail:?}. Its selector is {}. A row attached to a path \
           its own table never downloads is terms recorded against bytes that are not there.",
          row.file,
          table.name,
          table.selector_description()
        ));
      }
    }

    // Forward: every file the table NAMES must be covered by some row.
    if let Selection::Files(listed) = &table.selection {
      for file in listed {
        if !mine.iter().any(|r| r.covers(file)) {
          failures.push(format!(
            "MODELS_LOCK table {:?} stages {file:?} and no licence row covers it. The table names \
             every file it stages, so this is not a granularity the check has to guess at: one \
             row over the table is not coverage of the table, which is the whole reason the \
             licence table is keyed by artifact rather than by repository.",
            table.name
          ));
        }
      }
    }
  }
  failures
}

/// A MODELS_LOCK table reduced to what direction 1 needs: where it downloads
/// to, and what it selects.
struct StagedTable {
  name: String,
  /// `local-dir` with the leading `Models/` removed — the prefix every row on
  /// this table must carry.
  vendor_dir: String,
  selection: Selection,
}

/// What one table stages, as the lock itself states it.
enum Selection {
  /// `files = "a b c"` — an exact, complete list of table-relative paths. The
  /// repository can enumerate this without downloading anything.
  Files(Vec<String>),
  /// `include = "<patterns>"` — a space-separated glob list. The file list
  /// exists only after a download, so a row on such a table is reconciled
  /// against the patterns and against the repository's own per-file manifests.
  Include(Vec<String>),
}

impl StagedTable {
  /// `file` with this table's vendor directory stripped, or `None` when the
  /// row does not live under it at all.
  fn table_relative<'a>(&self, file: &'a str) -> Option<&'a str> {
    file.strip_prefix(&format!("{}/", self.vendor_dir))
  }

  /// Whether this table's selector picks up `tail`.
  ///
  /// A row may name a `.mlmodelc` BUNDLE rather than a file inside it (that is
  /// what a [`Key::Unmanifested`] row does), and every directory pattern in
  /// this lock ends `/*`, so a bundle is selected when the pattern with that
  /// suffix removed matches the bundle itself.
  fn selects(&self, tail: &str) -> bool {
    match &self.selection {
      Selection::Files(listed) => listed
        .iter()
        .any(|f| f == tail || f.starts_with(&format!("{tail}/"))),
      Selection::Include(patterns) => patterns.iter().any(|p| {
        glob_matches(p, tail)
          || p
            .strip_suffix("/*")
            .is_some_and(|dir| glob_matches(dir, tail))
      }),
    }
  }

  /// The selector, for a failure message.
  fn selector_description(&self) -> String {
    match &self.selection {
      Selection::Files(listed) => format!("files = {:?}", listed.join(" ")),
      Selection::Include(patterns) => format!("include = {:?}", patterns.join(" ")),
    }
  }
}

/// A row plus the table-relative file set it demonstrably covers.
///
/// Coverage is what makes direction 1 a FILE-level check: a row keyed on one
/// file inside a `.mlmodelc` covers the whole bundle only because the pin it
/// names is a per-file manifest of it, and the coverage set is read from that
/// manifest rather than assumed.
struct Covered<'a> {
  file: &'a str,
  staged_by: &'a str,
  /// Table-relative paths. An entry ending `.mlmodelc` stands for everything
  /// under that bundle, and is present only when the row demonstrably covers
  /// the whole bundle — it named it, or its pin is a per-file manifest of it.
  covered: BTreeSet<String>,
}

impl Covered<'_> {
  /// Whether this row accounts for the table-relative path `file`.
  fn covers(&self, file: &str) -> bool {
    self
      .covered
      .iter()
      .any(|c| c == file || (c.ends_with(".mlmodelc") && file.starts_with(&format!("{c}/"))))
  }
}

/// `fnmatch` with `*` crossing `/`, which is what `huggingface_hub` applies to
/// `--include` patterns and therefore what `MODELS_LOCK`'s selectors mean.
///
/// `?` is not used by any pattern in the lock and is not implemented: a
/// silently-wrong match here would be a coverage hole, so an unsupported
/// metacharacter is refused rather than treated as a literal.
fn glob_matches(pattern: &str, text: &str) -> bool {
  assert!(
    !pattern.contains('?') && !pattern.contains('['),
    "glob pattern {pattern:?} uses a metacharacter this matcher does not implement"
  );
  let parts: Vec<&str> = pattern.split('*').collect();
  if parts.len() == 1 {
    return pattern == text;
  }
  let Some(mut rest) = text.strip_prefix(parts[0]) else {
    return false;
  };
  let last = parts.len() - 1;
  for (i, part) in parts.iter().enumerate().skip(1) {
    if i == last {
      return rest.len() >= part.len() && rest.ends_with(part);
    }
    if part.is_empty() {
      continue;
    }
    match rest.find(part) {
      Some(at) => rest = &rest[at + part.len()..],
      None => return false,
    }
  }
  true
}

/// **Direction 2's strong clause in the MANIFEST channel, driven by the
/// feature graph rather than by the row.** No research-only artifact is
/// manifested by any feature closure that is not itself a commercial opt-in.
///
/// One of `wired`'s three derivations and only one, which is the correction
/// this doc needed: what it reads is the `#[cfg(feature = ...)]` the tree puts
/// on the module that manifests the artifact, plus cargo's feature graph — so
/// `derived` comes from the tree via [`loader_gates`] and `closures` from the
/// manifest via [`feature_closure`], and neither comes from the table. The
/// row's `gate` string is a CLAIM and is not consulted. The staged channel is
/// [`rows_whose_loader_is_not_their_kit`]'s and the tested channel is
/// [`research_only_tested`]'s; what NO predicate here can see is a path a
/// caller passes to a public door (the module doc's residual, issue #138 §8),
/// because those bytes are the caller's.
///
/// Reading the claim is what let two shapes through. `default = []` with
/// `speaker = ["commercial-face"]` passed, because only `default`'s closure was
/// consulted and the claimed gate carried the prefix — while enabling the
/// ordinary `speaker` feature wired the restricted artifact in. Hence: EVERY
/// non-commercial feature's closure, not just `default`'s.
fn research_only_wired(
  rows: &[Artifact],
  derived: &BTreeMap<&str, BTreeSet<String>>,
  closures: &BTreeMap<String, BTreeSet<String>>,
) -> Vec<String> {
  let mut failures = Vec::new();
  for row in rows {
    let Some((layer, terms)) = row.layer_where(Terms::forbids_commercial_use) else {
      continue;
    };
    let empty = BTreeSet::new();
    let gates = derived.get(row.file).unwrap_or(&empty);
    if gates.is_empty() {
      failures.push(format!(
        "{}: research-only at the {layer} layer, and the tree puts NO `#[cfg(feature = ...)]` on \
         the module that loads it — it compiles unconditionally, so this crate wires it in with \
         no gate to opt in to. {}",
        row.file,
        terms.detail()
      ));
      continue;
    }
    for gate in gates {
      if !gate.starts_with(COMMERCIAL_PREFIX) {
        failures.push(format!(
          "{}: research-only at the {layer} layer, but the tree gates its loader on {gate:?}, \
           which does not carry the {COMMERCIAL_PREFIX:?} prefix. A plain kit feature is not an \
           opt-in — every product that uses the kit wires this artifact in. {}",
          row.file,
          terms.detail()
        ));
      }
      for (feature, closure) in closures {
        if feature.starts_with(COMMERCIAL_PREFIX) || !closure.contains(gate) {
          continue;
        }
        let via = if feature == "default" {
          "a plain `cargo add coremlit` turns it on".to_string()
        } else {
          format!("enabling the ordinary feature {feature:?} turns it on")
        };
        failures.push(format!(
          "{}: research-only at the {layer} layer behind {gate:?}, but {gate:?} is in the feature \
           closure of {feature:?}, which is not a commercial opt-in, so enabling {feature:?} \
           WIRES this artifact in — {via}. {}",
          row.file,
          terms.detail()
        ));
      }
    }
  }
  failures
}

/// Why a layer withholds the shipping claim, in the words a failure message
/// needs — and the two reasons kept APART, because they are not the same
/// finding.
///
/// [`Terms::ResearchOnly`] is an ANSWER: somebody read the terms and they
/// forbid the shipping path. [`Terms::Unresolved`] is the ABSENCE of one:
/// nobody has established anything, so the bytes may well be perfectly
/// shippable — what does not exist is a document saying so. Collapsing the two
/// would make every message here assert a prohibition this repository has not
/// found, which is the register's own over-claim defect pointed backwards.
const fn withheld_because(terms: Terms) -> &'static str {
  match terms {
    Terms::ResearchOnly(_) => {
      "The terms are ESTABLISHED and they forbid commercial use, so shipping these bytes is \
       infringement."
    }
    Terms::Unresolved(_) => {
      "NOTHING is established over these bytes. That is not a prohibition — they may well be \
       shippable — but there is no grant for a shipping claim to rest on, and a configuration \
       the consumer never chose is this crate answering the open question on their behalf."
    }
    Terms::Permissive(_) | Terms::Attribution(_) => {
      "These terms DO permit a shipping claim, so a failure quoting this sentence is a defect in \
       the predicate rather than a finding about the artifact."
    }
  }
}

/// **Direction 2's wide clause.** Nothing whose terms leave a shipping claim
/// with nothing to rest on is WIRED into `default`.
///
/// Wider than [`research_only_wired`] in the rows it covers — research-only
/// AND unresolved — and deliberately weaker in what it demands of them. The
/// strong clause insists on a `commercial-` gate that no ordinary feature
/// pulls in; this one insists only that the consumer had to ask. That
/// asymmetry is the vocabulary decision recorded in this file's module doc:
/// `default` is the single configuration coremlit chooses on a consumer's
/// behalf, and choosing an artifact nobody has found a grant for is the thing
/// [`Terms::Unresolved`]'s own doc already said may not happen.
///
/// It reads the same two live facts the strong clause does — the tree's
/// `#[cfg(feature = ...)]` and the manifest's feature graph — and never the
/// row's claimed `gate`. It is the MANIFEST channel too, widened in its rows
/// rather than in its channels: the staged channel is
/// [`rows_whose_loader_is_not_their_kit`]'s for every row alike, and this
/// clause's tested channel is [`ungranted_tested_under_default`]'s. Like the
/// strong clause it is blind to a path a caller hands a public door, which is
/// the residual the module doc states.
fn ungranted_wired_into_default(
  rows: &[Artifact],
  derived: &BTreeMap<&str, BTreeSet<String>>,
  default_closure: &BTreeSet<String>,
) -> Vec<String> {
  let mut failures = Vec::new();
  for row in rows {
    let Some((layer, terms)) = row.ungranted_layer() else {
      continue;
    };
    let empty = BTreeSet::new();
    let gates = derived.get(row.file).unwrap_or(&empty);
    if gates.is_empty() {
      failures.push(format!(
        "{}: {} at the {layer} layer, and the tree puts NO `#[cfg(feature = ...)]` on the module \
         that loads it — it is wired into EVERY configuration, `default` included, so there is \
         nothing a consumer could decline. {} {}",
        row.file,
        terms.verdict(),
        withheld_because(terms),
        terms.detail()
      ));
      continue;
    }
    for gate in gates {
      if !default_closure.contains(gate) {
        continue;
      }
      failures.push(format!(
        "{}: {} at the {layer} layer behind {gate:?}, and `default` enables {gate:?} — a plain \
         `cargo add coremlit` wires it in, so this crate took the decision instead of the \
         consumer. {} {}",
        row.file,
        terms.verdict(),
        withheld_because(terms),
        terms.detail()
      ));
    }
  }
  failures
}

/// The names a target has to write for the compiler to reach one row's
/// artifact.
///
/// Every one is DERIVED: `local_dir` and `kit_dir` from the `MODELS_LOCK`
/// table that stages the row, `module` from the loader locator whose module
/// name that same table's `kit` is required to equal
/// ([`rows_whose_loader_is_not_their_kit`]). None of them is the row's prose,
/// and none is a name this file writes down for the artifact it happens to
/// carry today.
struct ArtifactNames {
  /// The lock's `local-dir`, e.g. `Models/facekit` — the directory the staging
  /// shard downloads the bundle into, and the path a suite that opened it by
  /// hand would have to spell.
  local_dir: String,
  /// `local-dir`'s own last component, e.g. `facekit`.
  ///
  /// Not redundant with `local_dir`, and the tree is why: the library spells
  /// the joined form (`arcface::STAGED_PATH` is `"Models/facekit/…"`) while
  /// the fixture module spells the split one
  /// (`workspace_root::models_root().join("facekit")`). A reader that knew
  /// only the joined spelling would walk past the split one, which is the
  /// spelling a suite that actually loads the bundle uses.
  kit_dir: String,
  /// The module that manifests the artifact, e.g. `arcface` — the same module
  /// name [`loader_gates`] reads the `#[cfg]` off.
  module: String,
}

/// One target the manifest DECLARES — `[[test]]`, `[[bench]]` or `[[example]]`
/// — reduced to the claim it makes and the sources that claim covers.
///
/// `[[bench]]` and `[[example]]` are in for a reason found in the manifest
/// rather than assumed: `whisper_rtf_gate` is a `[[test]]` whose `path` is
/// `benches/whisper/rtf_gate.rs`. The array a target is declared in is not
/// where the boundary lies, so a rule that read only `[[test]]` would be
/// keyed on the array name instead of on what the target compiles.
struct CompiledTarget {
  /// The manifest array it was declared in — `test`, `bench` or `example`.
  kind: String,
  /// The target's `name`, as the manifest gives it.
  name: String,
  /// The entry file, manifest-relative, as the manifest gives it.
  path: String,
  /// The target's `required-features`, verbatim — the CLAIM this predicate
  /// reconciles against the sources below.
  required_features: Vec<String>,
  /// Every file this target compiles when NO `commercial-` feature is on.
  sources: Vec<OrdinarySource>,
}

/// One compiled file, reduced to what the test channel asks of it.
///
/// Attributes are not here, and that is deliberate: an attribute is not code.
/// `#[doc]` — which is what every `//!` and `///` becomes — would otherwise
/// make the PROSE of `tests/face/arcface/mod.rs` ("staged under
/// `Models/facekit/`") indistinguishable from a load, and the register would
/// be reading its own documentation back as a finding. The three things read
/// out of an attribute are the `#[cfg(feature = ...)]` that decides whether
/// the construct is here at all, the `#[path = "…"]` that says where a
/// module's file is, and the `#[cfg_attr(…)]` that could quietly be either of
/// those two and is refused for it ([`AttributeRun`]).
struct OrdinarySource {
  /// Manifest-relative path to the file.
  path: String,
  /// Every string literal in the code an ordinary feature set compiles.
  literals: Vec<String>,
  /// Every identifier in that same code.
  idents: BTreeSet<String>,
  /// Every file this one embeds with `include_str!` or `include_bytes!`.
  embedded: Vec<EmbeddedFile>,
}

/// One file an `include_str!` or `include_bytes!` compiles INTO the target.
///
/// Its bytes are in the binary, so a fixture that spells the staged directory
/// is a reference to the artifact exactly as a string literal in the source
/// is. `include!` is not here: that one splices Rust, so it becomes an
/// [`OrdinarySource`] of its own.
struct EmbeddedFile {
  /// Manifest-relative path to the embedded file.
  path: String,
  /// Its text, or `None` when the bytes are not UTF-8.
  ///
  /// A non-UTF-8 `include_bytes!` target contributes no names — there is no
  /// text to search — and it is recorded rather than dropped so that the scan
  /// says what it saw and did not read.
  text: Option<String>,
}

impl CompiledTarget {
  /// Every way this target's ordinary-feature sources name `artifact`, each
  /// as the sentence a failure message needs.
  ///
  /// Four channels, because there are four ways a suite reaches a staged
  /// bundle and only the first of them is a Rust path:
  ///
  ///   1. a source file the module NAMES — `…/arcface/mod.rs` or
  ///      `…/arcface.rs`, which are the two files `mod arcface;` resolves to;
  ///      the first is what `tests/face/arcface/mod.rs` is, where the shared
  ///      pins and the fixture loaders that open the staged bundle live;
  ///   2. the module's own identifier, which is how `arcface::MODEL` and
  ///      `use …::face::{…, arcface}` both read once the lexer is done with
  ///      them;
  ///   3. a string literal holding the lock's `local-dir`;
  ///   4. a string literal holding the directory's own name, which is the
  ///      spelling `models_root().join("facekit")` uses.
  ///
  /// The first two run over every file the target compiles, `include!`d files
  /// among them; the last two run over the source's own literals AND over the
  /// text of every file it embeds with `include_str!`/`include_bytes!`, whose
  /// bytes are in the binary just as a literal's are.
  fn references_to(&self, artifact: &ArtifactNames) -> Vec<String> {
    let mut found = Vec::new();
    for source in &self.sources {
      if source
        .path
        .split('/')
        .any(|part| part.strip_suffix(".rs").unwrap_or(part) == artifact.module)
      {
        found.push(format!(
          "{}, a source file named after the {:?} module — the one its lock table's `kit` names",
          source.path, artifact.module
        ));
      }
      if source.idents.contains(&artifact.module) {
        found.push(format!(
          "{}: the identifier {:?}",
          source.path, artifact.module
        ));
      }
      for literal in &source.literals {
        if literal.contains(&artifact.local_dir) {
          found.push(format!(
            "{}: the string {literal:?}, which holds the lock's `local-dir` {:?}",
            source.path, artifact.local_dir
          ));
        } else if literal.contains(&artifact.kit_dir) {
          found.push(format!(
            "{}: the string {literal:?}, which holds {:?} — the directory the lock stages it into",
            source.path, artifact.kit_dir
          ));
        }
      }
      for embedded in &source.embedded {
        let Some(text) = &embedded.text else {
          continue;
        };
        if text.contains(&artifact.local_dir) {
          found.push(format!(
            "{}: {}, embedded in the binary, whose text holds the lock's `local-dir` {:?}",
            source.path, embedded.path, artifact.local_dir
          ));
        } else if text.contains(&artifact.kit_dir) {
          found.push(format!(
            "{}: {}, embedded in the binary, whose text holds {:?} — the directory the lock \
             stages it into",
            source.path, embedded.path, artifact.kit_dir
          ));
        }
      }
    }
    found.dedup();
    found
  }

  /// Every feature this target's `required-features` transitively enables.
  ///
  /// A name the manifest does not declare contributes only itself: cargo
  /// refuses to build such a target at all, and guessing a closure for it
  /// would be this reader inventing the graph it is supposed to be reading.
  fn feature_closure(&self, closures: &BTreeMap<String, BTreeSet<String>>) -> BTreeSet<String> {
    let mut closure = BTreeSet::new();
    for feature in &self.required_features {
      match closures.get(feature) {
        Some(reached) => closure.extend(reached.iter().cloned()),
        None => {
          closure.insert(feature.clone());
        }
      }
    }
    closure
  }
}

/// **Direction 2's TEST channel.** No target this crate declares compiles a
/// reference to a research-only artifact unless its own `required-features`
/// closure is a commercial opt-in.
///
/// The channel [`research_only_wired`] cannot see. That clause reads the
/// `#[cfg]` on the module that MANIFESTS the artifact, which says nothing
/// about a suite that never mentions that module: a `[[test]]` target with
/// `required-features = ["face"]` whose sources open `Models/facekit` through
/// the generic `FaceEmbedder` door tests the research-only bytes under an
/// ordinary feature while both of direction 2's other clauses stay green. The
/// module doc says `wired` means manifested, staged OR TESTED; this is the
/// third word, derived rather than asserted.
///
/// What it reconciles is a CLAIM against a FACT, the same shape as every other
/// predicate here. The claim is the target's `required-features`; the fact is
/// the set of files the compiler reaches from its `path` with no `commercial-`
/// feature enabled ([`ordinary_sources`]), and the names of the artifact in
/// them ([`CompiledTarget::references_to`]). A target that names the artifact
/// and asks for nothing commercial is a suite that runs against restricted
/// bytes on an ordinary feature.
///
/// **What it checks, in the words that are exactly true.** It is a
/// FAIL-CLOSED TRIPWIRE over the NAMES the ordinary build compiles, not a
/// proof about what a suite can reach. The names are four and they are all
/// derived ([`ArtifactNames`]): the loader module's file, its identifier, the
/// lock's `local-dir`, and the staged directory's own last component. The
/// places searched are the target's own source, source it `include!`s, and the
/// text of a fixture it embeds with `include_str!`/`include_bytes!`
/// ([`ordinary_sources`]) — the set closed under the two splicings whose
/// spelling rustc resolves the same way every time. Every way it can fail to
/// enumerate that set is a panic rather than an empty answer, so its error
/// direction is a false RED.
///
/// **And what it therefore does NOT check, stated rather than chased.** A
/// module or a load that a macro generates, code a build script emits, an
/// environment variable read at run time, a path composed at run time out of
/// separately harmless parts, and — as before — a caller who hands
/// `FaceEmbedder::load` a path of their own. None of those is a name on disk.
/// The guarantee that an ordinary-feature suite cannot RUN against the
/// restricted bytes is not this clause's to make and never was: it is
/// STAGING's, where the only shard that stages the directory is the kit's own,
/// and that kit must BE the `commercial-`gated loader module
/// ([`rows_whose_loader_is_not_their_kit`]). This clause is the tripwire over
/// a target that reaches for them anyway.
fn research_only_tested(
  rows: &[Artifact],
  names: &BTreeMap<&str, ArtifactNames>,
  targets: &[CompiledTarget],
  closures: &BTreeMap<String, BTreeSet<String>>,
) -> Vec<String> {
  let mut failures = Vec::new();
  for row in rows {
    let Some((layer, terms)) = row.layer_where(Terms::forbids_commercial_use) else {
      continue;
    };
    let Some(artifact) = names.get(row.file) else {
      failures.push(format!(
        "{}: research-only at the {layer} layer, and this reader could not derive the names a \
         target would have to write to reach it — so the TEST channel of `wired` is unchecked \
         for the one row that most needs it. {}",
        row.file,
        terms.detail()
      ));
      continue;
    };
    for target in targets {
      let found = target.references_to(artifact);
      if found.is_empty() {
        continue;
      }
      let closure = target.feature_closure(closures);
      if closure.iter().any(|f| f.starts_with(COMMERCIAL_PREFIX)) {
        continue;
      }
      failures.push(format!(
        "{}: research-only at the {layer} layer, and the [[{}]] target {:?} ({}) compiles it \
         under `required-features = {:?}` — whose closure {closure:?} carries no \
         {COMMERCIAL_PREFIX:?} feature. It names the artifact at {}. A suite that runs against \
         the bytes WIRES them as surely as the module that manifests them, and an ordinary \
         feature is not an opt-in. {}",
        row.file,
        target.kind,
        target.name,
        target.path,
        target.required_features,
        found.join("; and at "),
        terms.detail()
      ));
    }
  }
  failures
}

/// **Direction 2's wide clause, in the test channel.** No target a plain
/// `cargo add coremlit` already builds compiles a reference to an artifact
/// whose terms leave a shipping claim with nothing to rest on.
///
/// The widening [`ungranted_wired_into_default`] makes over
/// [`research_only_wired`], made again in the channel [`research_only_tested`]
/// opened — wider in the rows it covers (research-only AND unresolved) and
/// weaker in what it demands of them. The strong clause insists on a
/// `commercial-` feature; this one insists only that the target asked for
/// SOMETHING `default` does not already give it. A target whose
/// `required-features` closure sits entirely inside `default`'s is one a bare
/// `cargo test` builds, so an artifact it references is one this crate chose
/// to exercise on the consumer's behalf.
///
/// **It cannot fire today, and the reason is a live fact rather than an
/// absence**: `default = []`, and every target this manifest declares requires
/// a kit feature, so no target's closure sits inside `default`'s. Put a kit
/// feature into `default`, or drop a suite's `required-features`, and it reds
/// on the real manifest. Until then the falsifiers below are the whole reason
/// to believe it, which is this file's standing arrangement for a clause with
/// nothing yet in scope.
fn ungranted_tested_under_default(
  rows: &[Artifact],
  names: &BTreeMap<&str, ArtifactNames>,
  targets: &[CompiledTarget],
  closures: &BTreeMap<String, BTreeSet<String>>,
  default_closure: &BTreeSet<String>,
) -> Vec<String> {
  let mut failures = Vec::new();
  for row in rows {
    let Some((layer, terms)) = row.ungranted_layer() else {
      continue;
    };
    let Some(artifact) = names.get(row.file) else {
      failures.push(format!(
        "{}: {} at the {layer} layer, and this reader could not derive the names a target would \
         have to write to reach it — so the TEST channel of `wired` is unchecked for it. {} {}",
        row.file,
        terms.verdict(),
        withheld_because(terms),
        terms.detail()
      ));
      continue;
    };
    for target in targets {
      let found = target.references_to(artifact);
      if found.is_empty() {
        continue;
      }
      let closure = target.feature_closure(closures);
      if closure.iter().any(|f| !default_closure.contains(f)) {
        continue;
      }
      failures.push(format!(
        "{}: {} at the {layer} layer, and the [[{}]] target {:?} ({}) compiles it under \
         `required-features = {:?}` — whose closure {closure:?} is already inside `default`'s, \
         so a bare `cargo test` runs it against these bytes and there is nothing a consumer \
         could decline. It names the artifact at {}. {} {}",
        row.file,
        terms.verdict(),
        target.kind,
        target.name,
        target.path,
        target.required_features,
        found.join("; and at "),
        withheld_because(terms),
        terms.detail()
      ));
    }
  }
  failures
}

/// **Direction 2's STAGING channel.** Every row's loader module is the module
/// named by its own `MODELS_LOCK` table's `kit`.
///
/// This is the whole derivation of the staging word in `wired`, and it is a
/// CHAIN rather than a string comparison. The lock row is what stages the
/// bytes; its `kit` names the module that manifests them; that module's
/// `#[cfg(feature = ...)]` is the gate [`loader_gates`] reads; and that gate
/// is what [`research_only_wired`] and [`ungranted_wired_into_default`] judge.
/// Break the equality and the chain runs to a DIFFERENT module's `#[cfg]`, so
/// direction 2 would clear a gate that has nothing to do with the artifact the
/// lock stages — which is why the failure text below says so rather than
/// leaving a reader to reconstruct it.
///
/// `Artifact::loader` is still written down by hand, so on its own it could
/// point at any module in the tree; tying it to the kit the LOCK declares
/// means the row cannot borrow an unrelated module's `#[cfg]`. Lock, tree and
/// manifest then have to agree before a gate is believed.
fn rows_whose_loader_is_not_their_kit(
  rows: &[Artifact],
  kits: &BTreeMap<&str, &str>,
) -> Vec<String> {
  let mut failures = Vec::new();
  for row in rows {
    let kit = kits.get(row.staged_by).unwrap_or_else(|| {
      panic!(
        "{}: staged_by {:?} names no MODELS_LOCK table",
        row.file, row.staged_by
      )
    });
    let (_, module) = row.loader.split_once("::").unwrap_or_else(|| {
      panic!(
        "{}: loader {:?} is not `<source>::<module>`",
        row.file, row.loader
      )
    });
    if module == *kit {
      continue;
    }
    failures.push(format!(
      "{}: its lock table declares kit {kit:?} but the row's loader is the {module:?} module. \
       That equality is how the STAGING channel of direction 2's `wired` is derived: the lock \
       row stages the bytes, its `kit` names the module that manifests them, and THAT module's \
       `#[cfg(feature = ...)]` is the gate `research_only_wired` and \
       `ungranted_wired_into_default` judge. A row that reads another kit's `#[cfg]` reads \
       another kit's gate — the artifact would be staged behind one feature and cleared behind \
       another.",
      row.file
    ));
  }
  failures
}

/// **Direction 3.** No `commercial-` feature gates only artifacts that are
/// GRANTED at both layers — and no `commercial-` feature gates nothing at all
/// in the SOURCE.
///
/// The one people forget. A gate that protects nothing is worse than no gate:
/// it reads as a live restriction, so nobody re-examines the artifacts behind
/// it, and the next artifact added there inherits reassurance it never earned.
///
/// Three ways to be that, and the check refuses all three. The third is the one
/// a row-driven version could not see: a feature declared in `[features]` that
/// no `#[cfg(feature = ...)]` in the tree names compiles nothing differently
/// whether it is on or off. It is a NAME, not a gate, and a restricted row
/// naming it is behind no gate at all.
///
/// **This direction runs backwards, so `unresolved` needs its own wording.**
/// The other two ask "is this artifact protected"; this one asks "does this
/// protection still have a cause", and answers RED when it does not. An
/// unresolved row therefore must not red it — the row is not clear, so a gate
/// over it is not standing over nothing. But the cause it stands on is not the
/// research-only one and the message must not say it is: research-only means a
/// document forbids the shipping path and the gate is retired when that
/// document changes; unresolved means no document grants it and the gate is
/// holding an open QUESTION, retired when somebody answers it. Calling an
/// unresolved row "restricted" would assert the very prohibition the row says
/// nobody has established, so the failure text names both causes and says
/// which is which.
fn commercial_features_gating_nothing_restricted(
  rows: &[Artifact],
  derived: &BTreeMap<&str, BTreeSet<String>>,
  features: &BTreeSet<String>,
  cfg_in_source: &BTreeSet<String>,
) -> Vec<String> {
  let mut failures = Vec::new();
  for feature in features.iter().filter(|f| f.starts_with(COMMERCIAL_PREFIX)) {
    if !cfg_in_source.contains(feature) {
      failures.push(format!(
        "feature {feature:?} carries the {COMMERCIAL_PREFIX:?} prefix but NO \
         `#[cfg(feature = ...)]` in the source tree names it, so enabling it compiles nothing \
         differently and disabling it withholds nothing. It is a name, not a gate, and any row \
         that claims it is behind no gate at all."
      ));
      continue;
    }
    let gated: Vec<&Artifact> = rows
      .iter()
      .filter(|r| {
        derived
          .get(r.file)
          .is_some_and(|g| g.contains(feature.as_str()))
      })
      .collect();
    if gated.is_empty() {
      failures.push(format!(
        "feature {feature:?} carries the {COMMERCIAL_PREFIX:?} prefix but no licence row is gated \
         by it. Either it gates an artifact with no row (direction 1), or it is a gate left \
         standing after the artifact it protected went away — retire it."
      ));
      continue;
    }
    if gated.iter().all(|r| r.ungranted_layer().is_none()) {
      let granted: Vec<&str> = gated.iter().map(|r| r.file).collect();
      failures.push(format!(
        "feature {feature:?} carries the {COMMERCIAL_PREFIX:?} prefix, but every artifact it \
         gates is GRANTED at both layers: {}. A {COMMERCIAL_PREFIX:?} gate stands on one of two \
         causes and this one has neither — a RESEARCH-ONLY row, where a document forbids the \
         shipping path, or an UNRESOLVED row, where no document grants it and the gate holds an \
         open question rather than a prohibition. An upstream relicensed, the terms were re-read, \
         or the question was answered; either way the gate now says a restriction exists that \
         does not, so retire it and move the artifacts to a plain feature.",
        granted.join(", ")
      ));
    }
  }
  failures
}

/// The documentation rule for [`COMMERCIAL_PREFIX`] features.
///
/// BEGINS WITH an affirmative warning, and carries no negation. A substring
/// test over a `. `-split first sentence passed both
/// "This feature no longer requires a commercial license" and
/// "Cleared for commercial use! This feature requires a commercial license" —
/// the first inverts the warning, the second buries it behind the exact
/// misreading the prefix invites.
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
    let normalised = normalise_spelling(&first);
    if !COMMERCIAL_DOC_OPENINGS
      .iter()
      .any(|opening| normalised.starts_with(opening))
    {
      failures.push(format!(
        "feature {feature:?}: its first documented sentence is {first:?}, which does not BEGIN \
         with any of {COMMERCIAL_DOC_OPENINGS:?}. The name reads as an ENDORSEMENT of commercial \
         use; the sentence that corrects it has to be the first one and has to open with the \
         correction — a warning that arrives after a clause, or inside one, arrives after the \
         misreading has settled."
      ));
      continue;
    }
    if let Some(word) = negation_in(&normalised) {
      failures.push(format!(
        "feature {feature:?}: its first documented sentence is {first:?}, which opens with the \
         warning and then carries the negation {word:?}. A sentence that takes the warning back \
         has not warned anybody; put the qualification in a later sentence."
      ));
    }
  }
  failures
}

/// The first negating word in `text`, matched as a WORD.
///
/// Word-level, because a substring search finds "not" inside "notice" and
/// would fail a correctly-worded feature.
fn negation_in(text: &str) -> Option<&'static str> {
  let words: BTreeSet<String> = text
    .split_whitespace()
    .map(|w| {
      w.chars()
        .filter(|c| c.is_alphanumeric())
        .collect::<String>()
    })
    .collect();
  NEGATIONS.iter().copied().find(|n| words.contains(*n))
}

/// The first sentence of a documentation block: everything up to and including
/// the first terminator that ends a word, with the block's line breaks
/// flattened.
///
/// `.`, `!` and `?` all terminate. Recognising only `. ` is what let
/// "Cleared for commercial use! This feature requires a commercial license"
/// read as ONE sentence containing the warning.
fn first_sentence(doc: &str) -> String {
  let flat = doc.split_whitespace().collect::<Vec<_>>().join(" ");
  let bytes = flat.as_bytes();
  for (i, b) in bytes.iter().enumerate() {
    if !matches!(b, b'.' | b'!' | b'?') {
      continue;
    }
    if i + 1 == bytes.len() || bytes[i + 1] == b' ' {
      return flat[..=i].trim().to_string();
    }
  }
  flat.trim().to_string()
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
//
// EVERY READER BELOW INFERS SOMETHING FROM A FILE, AND THE FAILURE MODE THAT
// MATTERS IS THE ONE WHERE A MIS-READ MAKES A CHECK PASS.
//
// Two rounds of review found the same defect twice, one layer apart: a
// hand-rolled approximation of a grammar read valid input wrongly, and the
// wrong reading was the reassuring one. First the manifest reader, which could
// not see six spellings of `default` that Cargo obeys and reported every one of
// them EMPTY; then the loader-gate reader, which scanned for the substring
// `feature = "` in attributes and comments alike and derived a REQUIREMENT from
// a negation, an `any(..)` alternative and a sentence. So the roster, and what
// each does now:
//
// | reader | reads | grammar | if it mis-reads |
// |---|---|---|---|
// | `declared_features` and its callers | `Cargo.toml` | the `toml` crate | panics; an undecodable manifest is not an empty one |
// | `gates_of_module` / `required_features` | a loader's `#[cfg]` | `syn`, one predicate per item | `Err`; only the positive form derives a gate |
// | `cfg_features_in` | every `#[cfg]`/`cfg!` under `src/` | `proc-macro2` tokens | a missed site reds direction 3; prose and strings can no longer add one |
// | `fp16_pinned_bundles` | `tests/fp16_guards.rs` rosters | `proc-macro2` tokens, anchored on the `path` field | a missed entry would silently shrink direction 1's second enumeration, so it is read structurally |
// | `parse_lock` | `MODELS_LOCK` | hand-rolled, mirroring ci.yml's sed/awk | panics on anything that is not a header, a comment or `key = "value"`; `staged_tables` panics again on a table missing `local-dir` or its selector |
// | `pins_at` | a `const`/`fn` holding SHA-256s | hand-rolled over quoted runs | panics on an ambiguous anchor or an empty result, and `every_rows_sha256_matches_the_pin_it_names` panics on a key the pin does not hold |
// | `feature_docs` | `[features]` COMMENTS | hand-rolled, line-wise | a key it cannot see arrives with NO documentation and is reported undocumented — red. "Never green" was this table's claim and it was wrong by one cell: the `#` was stripped BEFORE the indentation was checked, and a whitespace-led non-comment line did not clear the pending block, so a comment indented inside a multi-line array documented the NEXT key and the doc rule went green on it. An indented line now ends the block before anything else (`a_comment_inside_a_multi_line_array_documents_nothing`). Comments are the one thing a TOML parser drops, so this has no alternative |
// | `compiled_targets` | `Cargo.toml`'s `[[test]]`/`[[bench]]`/`[[example]]` | the `toml` crate | panics on a target with no `name`, no `path`, or a `required-features` that is not an array of strings; a target it dropped is a suite direction 2 would clear unread |
// | `scan_source` / `scan_tokens` / `attribute_run` | a declared target's compiled sources | `proc-macro2` tokens, with `syn` over each attribute run | panics on source it cannot parse or tokenise; a `#[cfg]` `required_features` will not read leaves the item IN the ordinary source set, so an unreadable gate reds rather than hides a reference |
// | `module_file` | a `mod` declaration's file | rustc's own rule over `#[path]` and the two candidates | panics when it resolves to neither candidate or to both — the two cases where guessing would enumerate a source set the compiler never has |
// | `first_sentence`, `negation_in`, `normalise_spelling` | a doc comment's PROSE | word- and sentence-level | prose is text; these infer no structure |
//
// The rule the table encodes: a reader may be hand-rolled only where every
// mis-read exits through a panic or a red. Where a mis-read could produce a
// PLAUSIBLE-BUT-WRONG value that a check then believes, it uses a real parser.
// Adding a reader here means placing it in that table, not just writing it.

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
///
/// # Refuses a repeated table header
///
/// Every consumer downstream is keyed by the table's NAME, and they disagree
/// about which of two same-named tables wins: [`kits_of`] collects into a
/// `BTreeMap` and keeps the LAST, while [`artifact_names`] and direction 1's
/// coverage `find` the FIRST. Two `["FinDIT-Studio/facekit-coreml"]` tables
/// ordered `kit = "face"` then `kit = "arcface"` would therefore show
/// [`rows_whose_loader_is_not_their_kit`] only the second one, and a plain
/// staging kit would sit behind the commercial one unseen. The repository's
/// own downloader would not agree with either reading. So a repeat is refused
/// here, before any map is built, and every consumer below operates on a
/// uniquely keyed set by construction. Today's lock has none: the two
/// speakerkit tables and the two whisper tables each name a DIFFERENT
/// repository.
fn parse_lock(contents: &str) -> Vec<LockTable> {
  let mut tables: Vec<LockTable> = Vec::new();
  let mut headers: BTreeMap<String, usize> = BTreeMap::new();
  for (index, line) in contents.lines().enumerate() {
    let numbered = index + 1;
    let line = line.trim();
    if line.is_empty() || line.starts_with('#') {
      continue;
    }
    if let Some(name) = line.strip_prefix("[\"").and_then(|s| s.strip_suffix("\"]")) {
      if let Some(first) = headers.insert(name.to_string(), numbered) {
        panic!(
          "MODELS_LOCK: the table {name:?} is declared twice, at line {first} and again at line \
           {numbered}. This reader's consumers are keyed by the table name and disagree about \
           which one wins — `kits_of` keeps the last, `artifact_names` and direction 1's \
           coverage take the first — so a second table over the same repository would stage a \
           kit that half the checks above cannot see."
        );
      }
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

/// This crate's manifest, verbatim.
fn manifest_text() -> String {
  read_rel("Cargo.toml")
}

/// The same manifest, read from the file the REPOSITORY holds rather than the
/// one the compiling package happens to sit next to.
///
/// `cargo package` re-serialises the manifest into the tarball and DROPS EVERY
/// COMMENT doing it, so a feature's documentation exists only in the
/// checked-in file. Checks that read comments must read that file; checks that
/// need only names or entries are happy with either. `None` outside the
/// repository workspace, where the comment-bearing manifest is not present at
/// all and the rule is simply unverifiable.
fn repository_manifest_text() -> Option<String> {
  let root = workspace_root::try_workspace_root()?;
  let manifest = root.join("coremlit/Cargo.toml");
  if !manifest.is_file() {
    eprintln!("model_licences: no comment-bearing manifest; the doc rule is skipped");
    return None;
  }
  Some(
    std::fs::read_to_string(&manifest)
      .unwrap_or_else(|e| panic!("read {}: {e}", manifest.display())),
  )
}

/// The `[features]` table of `manifest`, decoded by the REAL TOML parser.
///
/// # Why this is not hand-rolled any more
///
/// It was, and the reader was a hole. It skipped every line beginning with
/// whitespace, split on the first `=`, and pulled the DOUBLE-quoted runs out of
/// the value. TOML permits all of the following, and Cargo obeys every one:
///
/// ```text
///   ␣␣default = ["identity"]     an indented key — skipped outright
///   default = ['identity']       a literal string — no `"` to split on
///   "default" = ["identity"]     a quoted key — never equal to `default`
///   default = [ # note ]         a `#` comment carrying `]` — value ends early
///     "identity",
///   ]
///   [ features ]                 a non-canonical header — block came back empty
///   features.default = [...]     a dotted key — no header to find at all
/// ```
///
/// Each one made `default` look EMPTY, and an empty `default` closure is
/// exactly what [`no_ungranted_artifact_is_wired_into_default`] reads as
/// "nothing ungranted ships without an opt-in". A reader that cannot see a
/// spelling Cargo obeys is not a check; it is a check-shaped comment.
///
/// # Fails closed
///
/// A manifest that does not decode, that declares no `[features]` table, or
/// whose entries are not arrays of strings PANICS here. The alternative —
/// returning an empty map — would let every reachability check pass vacuously
/// on a manifest nobody could read, which is the failure mode this function
/// exists to remove.
fn declared_features(manifest: &str) -> BTreeMap<String, Vec<String>> {
  let document: toml::Table = toml::from_str(manifest).unwrap_or_else(|e| {
    panic!(
      "the manifest under test does not decode as TOML: {e}. This check reads `default`'s \
       closure to decide whether an ungranted artifact ships; a manifest it cannot read is a \
       manifest it cannot clear."
    )
  });
  let features = document.get("features").cloned().unwrap_or_else(|| {
    panic!(
      "the manifest under test declares no `features` table. An absent feature graph is not an \
       empty one: every reachability check here would pass vacuously on it."
    )
  });
  features.try_into().unwrap_or_else(|e| {
    panic!(
      "the manifest's `[features]` table is not a map of string arrays: {e}. A feature whose \
       entries this check cannot decode is a feature whose closure it cannot compute."
    )
  })
}

/// The `[features]` block of `manifest` as TEXT, comments included.
///
/// Only [`feature_docs`] and its vacuity guard read this: comments are the one
/// thing a TOML parser drops, so the doc rule has no alternative to scanning
/// lines. Nothing that decides REACHABILITY comes through here — that is
/// [`declared_features`]'s job, and the split is the point.
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

/// The declared feature names, as the TOML parser sees them.
fn feature_names(manifest: &str) -> BTreeSet<String> {
  declared_features(manifest).into_keys().collect()
}

/// One feature's entries, as the TOML parser sees them. Empty for a feature the
/// manifest does not declare — which [`declared_features`] guarantees is a real
/// absence rather than a spelling this reader could not see.
fn feature_entries(manifest: &str, feature: &str) -> Vec<String> {
  declared_features(manifest)
    .remove(feature)
    .unwrap_or_default()
}

/// Every feature transitively enabled by `seed`, `seed` included.
///
/// Entries naming a dependency (`dep:x`) or a dependency's own feature (`x/y`)
/// are not this crate's features and do not extend the closure.
fn feature_closure(manifest: &str, seed: &str) -> BTreeSet<String> {
  let declared = declared_features(manifest);
  let mut seen = BTreeSet::new();
  let mut queue = vec![seed.to_string()];
  while let Some(feature) = queue.pop() {
    if !seen.insert(feature.clone()) {
      continue;
    }
    for entry in declared.get(&feature).into_iter().flatten() {
      if !entry.starts_with("dep:") && !entry.contains('/') {
        queue.push(entry.clone());
      }
    }
  }
  seen
}

/// The contiguous `#` comment block immediately above each feature.
///
/// A blank line ends a block, so a comment about the section above cannot be
/// mistaken for documentation of the feature below it.
///
/// Necessarily textual — a TOML parser drops comments — and therefore
/// deliberately conservative about which keys it recognises: it reads the
/// unindented, unquoted spelling and nothing else. That is safe here in a way
/// it was NOT safe in the reachability readers, because the set of features
/// this rule must find documentation FOR comes from [`declared_features`]. A
/// feature spelled in a way this scanner cannot see therefore arrives with no
/// documentation and is reported as undocumented — red.
///
/// Conservative about which keys it recognises is not the same as conservative
/// about which COMMENTS it attaches, and that is where it was once fail-open:
/// an indented comment inside a multi-line array used to attach to the next
/// key, which could document a feature nobody wrote a word about into green.
/// So an indented line ends the pending block, and that test comes first.
fn feature_docs(manifest: &str) -> BTreeMap<String, String> {
  let block = features_block_of(manifest);
  let mut docs = BTreeMap::new();
  let mut pending: Vec<&str> = Vec::new();
  for line in block.lines() {
    let trimmed = line.trim();
    if trimmed.is_empty() {
      pending.clear();
      continue;
    }
    // An INDENTED line is inside a multi-line value, not at the top level of
    // the table, and this reader documents a key from the contiguous block
    // ABOVE it. So an indented line — comment or not — ends the pending block
    // rather than extending it. Checking this BEFORE the `#` is what stops a
    // comment inside `a = [ .. ]` from becoming the documentation of whatever
    // key follows the array's closing bracket.
    if line.starts_with(char::is_whitespace) {
      pending.clear();
      continue;
    }
    if let Some(comment) = trimmed.strip_prefix('#') {
      pending.push(comment.trim());
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

/// The cargo features the TREE makes a module declaration conditional on.
///
/// Reads `<crate-relative source>::<module>`, finds the one `mod <module>;`
/// declaration in that file, and collects the `#[cfg(feature = "...")]`
/// attributes directly above it. An empty result means the module compiles
/// unconditionally — which is a finding, not a default.
///
/// This is the fact directions 2 and 3 run on. `Artifact::gate` is a claim
/// about which feature controls an artifact; only the tree decides it, and a
/// predicate that reads the claim is reconciling the table against itself.
fn loader_gates(locator: &str) -> BTreeSet<String> {
  let (rel, module) = locator
    .split_once("::")
    .unwrap_or_else(|| panic!("loader locator {locator:?} is not `<source>::<module>`"));
  gates_of_module(&read_rel(rel), module)
    .unwrap_or_else(|why| panic!("loader locator {locator:?}: {rel} {why}"))
}

/// [`loader_gates`] over source TEXT, so every cfg spelling is exercisable
/// without a file.
///
/// Parses with `syn` and derives a gate ONLY from the exact positive form
/// `#[cfg(feature = "name")]`, on the declaration or on any module enclosing
/// it. Everything else in the `cfg` family is an error, never a name — see
/// [`required_features`].
fn gates_of_module(source: &str, module: &str) -> Result<BTreeSet<String>, String> {
  let file = syn::parse_file(source).map_err(|e| {
    format!(
      "does not parse as Rust ({e}). This reader derives the gate that directions 2 and 3 \
       reason about; source it cannot parse is source whose gate it cannot establish."
    )
  })?;
  let mut chains = Vec::new();
  find_module_chains(&file.items, module, &mut Vec::new(), &mut chains);
  let [chain] = chains.as_slice() else {
    return Err(format!(
      "holds {} declarations of `mod {module}`; exactly one must be present, or this reader \
       could be reading the wrong module's gate",
      chains.len()
    ));
  };
  let mut gates = BTreeSet::new();
  for attrs in chain {
    gates.extend(required_features(attrs)?);
  }
  Ok(gates)
}

/// Every declaration of `mod <module>` reachable from `items`, each as the
/// chain of attribute lists that must all admit it: the enclosing modules'
/// attributes outermost, its own last.
///
/// Only ancestors are collected. A sibling module's `#[cfg(test)]` is never
/// looked at, so a spelling [`required_features`] refuses cannot fail a read it
/// has no bearing on.
fn find_module_chains<'a>(
  items: &'a [syn::Item],
  module: &str,
  chain: &mut Vec<&'a [syn::Attribute]>,
  out: &mut Vec<Vec<&'a [syn::Attribute]>>,
) {
  for item in items {
    let syn::Item::Mod(declared) = item else {
      continue;
    };
    if declared.ident == module {
      let mut hit = chain.clone();
      hit.push(&declared.attrs);
      out.push(hit);
    }
    if let Some((_, inner)) = &declared.content {
      chain.push(&declared.attrs);
      find_module_chains(inner, module, chain, out);
      chain.pop();
    }
  }
}

/// The features one item's attributes make REQUIRED for it to compile.
///
/// # Fails closed
///
/// Exactly `#[cfg(feature = "name")]` is understood. Every other `cfg`-family
/// spelling is an error rather than a name, because in each of them the feature
/// mentioned is not one the item requires:
///
/// | spelling | what the name would have meant |
/// |---|---|
/// | `#[cfg(not(feature = "x"))]` | compiles when `x` is OFF — the opposite |
/// | `#[cfg(any(target_os = "macos", feature = "x"))]` | compiles with `x` off, on that target |
/// | `#[cfg(all(feature = "x", feature = "y"))]` | two requirements, and a row claims one gate |
/// | `#[cfg(target_os = "macos")]` | a real gate, but not a cargo feature |
/// | `#[cfg_attr(feature = "x", ...)]` | attaches an attribute; gates nothing by itself |
///
/// A gate derived from any of them would be believed by
/// [`ungranted_wired_into_default`], which asks whether `default` enables
/// the gate — and concludes an artifact is withheld whenever it does not. A
/// loader gated on a NEGATION would then read as withheld from `default` while
/// compiling in `default`, which is the exact reassurance this file exists to
/// refuse.
///
/// Attributes outside the `cfg` family are ignored outright rather than
/// scanned: a `#[doc]` string that quotes a `#[cfg]` is prose, not compilation.
///
/// # One `cfg` per item
///
/// Two `#[cfg]` attributes on one item are a conjunction, and so is
/// `#[cfg(all(..))]`. Accepting the first while refusing the second would be
/// the same rule written twice with different answers, so this reader takes
/// neither: it reads one PREDICATE per item and does not evaluate cfg
/// EXPRESSIONS at all. Nesting is not an expression — a module inside a gated
/// module genuinely requires both, and [`gates_of_module`] unions the chain.
fn required_features(attrs: &[syn::Attribute]) -> Result<BTreeSet<String>, String> {
  let mut gates = BTreeSet::new();
  let cfgs = attrs
    .iter()
    .filter(|attr| attr.path().is_ident("cfg"))
    .count();
  if cfgs > 1 {
    return Err(format!(
      "carries {cfgs} `#[cfg(...)]` attributes on one item. Together they are a conjunction,        which is what `#[cfg(all(..))]` spells and what this reader refuses there; it reads one        predicate per item and does not evaluate cfg expressions."
    ));
  }
  for attr in attrs {
    let path = attr.path();
    if path.is_ident("cfg_attr") {
      return Err(format!(
        "carries `#[cfg_attr(...)]` on the module that loads it. A `cfg_attr` attaches an \
         attribute conditionally — it can even attach a further `#[cfg]` — and the feature it \
         names is not one the module requires. This reader does not evaluate it and will not \
         guess: {}",
        rendered(attr)
      ));
    }
    if !path.is_ident("cfg") {
      continue;
    }
    let name = attr.parse_args_with(positive_feature).map_err(|e| {
      format!(
        "carries a `#[cfg(...)]` this reader will not read as a feature requirement ({e}): \
           {}. Only the positive form `#[cfg(feature = \"name\")]` derives a gate; a negation, \
           a target alternative, a combination, or several predicates each make the name mean \
           something other than \"required to compile\", and a gate that means something else \
           is one directions 2 and 3 would reason about wrongly.",
        rendered(attr)
      )
    })?;
    gates.insert(name);
  }
  Ok(gates)
}

/// One attribute rendered back to source, for a failure message.
fn rendered(attr: &syn::Attribute) -> String {
  match &attr.meta {
    syn::Meta::List(list) => format!("`#[{}({})]`", joined(&list.path), list.tokens),
    syn::Meta::Path(path) => format!("`#[{}]`", joined(path)),
    syn::Meta::NameValue(pair) => format!("`#[{} = ...]`", joined(&pair.path)),
  }
}

/// An attribute path as `a::b`.
fn joined(path: &syn::Path) -> String {
  path
    .segments
    .iter()
    .map(|segment| segment.ident.to_string())
    .collect::<Vec<_>>()
    .join("::")
}

/// Parses exactly `feature = "name"` and nothing else.
///
/// A `syn` parser rather than a matcher: `parse_args_with` requires the WHOLE
/// argument list to be consumed, so a second predicate, a wrapping `not`/`any`/
/// `all`, or a non-`feature` key each fail here rather than contributing a
/// name.
fn positive_feature(input: syn::parse::ParseStream<'_>) -> syn::Result<String> {
  let key: syn::Ident = input.parse()?;
  if key != "feature" {
    return Err(syn::Error::new(
      key.span(),
      format!("expected the predicate `feature`, found `{key}`"),
    ));
  }
  input.parse::<syn::Token![=]>()?;
  Ok(input.parse::<syn::LitStr>()?.value())
}

/// Every feature name a `#[cfg(feature = "...")]` in this crate's `src/` tree
/// actually names.
///
/// A feature declared in `[features]` that appears nowhere here compiles
/// nothing differently whether it is on or off. That is the shape direction 3
/// could not see while it only asked whether a ROW named the feature.
fn cfg_features_in_source() -> BTreeSet<String> {
  let mut found = BTreeSet::new();
  let mut files = Vec::new();
  collect_rust_files(
    &Path::new(env!("CARGO_MANIFEST_DIR")).join("src"),
    &mut files,
  );
  assert!(
    files.len() >= 100,
    "only {} .rs files walked under src/; the walk is broken and every feature would read as \
     ungated",
    files.len()
  );
  for file in files {
    let text = std::fs::read_to_string(&file).unwrap_or_else(|e| panic!("read {file:?}: {e}"));
    found.extend(cfg_features_in(&text));
  }
  found
}

/// Every feature name a conditional-compilation site in one source TEXT names,
/// wherever it sits inside the predicate.
///
/// # A different question from [`required_features`], deliberately
///
/// That one asks which feature an item REQUIRES, and refuses every spelling
/// where the answer is not exactly one name. This one asks whether a feature
/// changes what compiles AT ALL, so a negation, an alternative and a `cfg_attr`
/// all count — enabling the feature does compile something differently in each.
/// The two must not share a rule.
///
/// # Why this is not a substring search any more
///
/// It was: `text.find("feature = \"")` over the whole file. That matched prose
/// and string literals as readily as attributes, so a feature named only in a
/// SENTENCE — this tree has several, e.g. ``//! `#![cfg(feature = "…")]` `` in
/// `audio/speaker/mod.rs` — read as a live gate. The clause that consumes this,
/// [`commercial_features_gating_nothing_restricted`], reds when a
/// `commercial-` feature names no conditional compilation at all; a phantom
/// from a comment is exactly what makes that clause pass over the gate it was
/// written to catch.
///
/// Reading TOKENS instead removes the whole class: the lexer has already
/// decided what is a comment (gone), what is a string (one `Literal`), and what
/// is an attribute — no rule here has to approximate that. Missing a real site
/// remains possible only if a file does not tokenise, which panics.
fn cfg_features_in(source: &str) -> BTreeSet<String> {
  let tokens: proc_macro2::TokenStream = source.parse().unwrap_or_else(|e| {
    panic!(
      "source does not tokenise ({e}). This sweep decides whether a `commercial-` feature gates \
       any code at all; a file it cannot read is a file whose gates it cannot count."
    )
  });
  let mut found = BTreeSet::new();
  collect_cfg_sites(tokens, &mut found);
  found
}

/// Walks a token stream for `#[cfg(..)]` / `#![cfg(..)]` / `#[cfg_attr(..)]`
/// attributes and `cfg!(..)` invocations, collecting the feature names inside
/// each. Recurses through every group, so an attribute inside a `macro_rules!`
/// body or on a deeply nested item counts like any other.
fn collect_cfg_sites(tokens: proc_macro2::TokenStream, out: &mut BTreeSet<String>) {
  use proc_macro2::{Delimiter, TokenTree};
  let trees: Vec<TokenTree> = tokens.into_iter().collect();
  let mut at = 0;
  while at < trees.len() {
    match &trees[at] {
      // `#[..]` or `#![..]`
      TokenTree::Punct(hash) if hash.as_char() == '#' => {
        let mut next = at + 1;
        if matches!(trees.get(next), Some(TokenTree::Punct(p)) if p.as_char() == '!') {
          next += 1;
        }
        if let Some(TokenTree::Group(body)) = trees.get(next)
          && body.delimiter() == Delimiter::Bracket
        {
          collect_cfg_sites(body.stream(), out);
          let inner: Vec<TokenTree> = body.stream().into_iter().collect();
          if let (Some(TokenTree::Ident(name)), Some(TokenTree::Group(args))) =
            (inner.first(), inner.get(1))
            && (name == "cfg" || name == "cfg_attr")
            && args.delimiter() == Delimiter::Parenthesis
          {
            collect_feature_names(args.stream(), out);
          }
          at = next + 1;
          continue;
        }
      }
      // `cfg!(..)`
      TokenTree::Ident(name) if name == "cfg" => {
        if let (Some(TokenTree::Punct(bang)), Some(TokenTree::Group(args))) =
          (trees.get(at + 1), trees.get(at + 2))
          && bang.as_char() == '!'
          && args.delimiter() == Delimiter::Parenthesis
        {
          collect_feature_names(args.stream(), out);
          at += 3;
          continue;
        }
      }
      TokenTree::Group(group) => collect_cfg_sites(group.stream(), out),
      _ => {}
    }
    at += 1;
  }
}

/// Every `feature = "name"` in one cfg predicate, at any nesting depth.
fn collect_feature_names(tokens: proc_macro2::TokenStream, out: &mut BTreeSet<String>) {
  use proc_macro2::TokenTree;
  let trees: Vec<TokenTree> = tokens.into_iter().collect();
  for (at, tree) in trees.iter().enumerate() {
    match tree {
      TokenTree::Ident(key) if key == "feature" => {
        if let (Some(TokenTree::Punct(eq)), Some(TokenTree::Literal(value))) =
          (trees.get(at + 1), trees.get(at + 2))
          && eq.as_char() == '='
          && let Ok(name) = syn::parse_str::<syn::LitStr>(&value.to_string())
        {
          out.insert(name.value());
        }
      }
      TokenTree::Group(group) => collect_feature_names(group.stream(), out),
      _ => {}
    }
  }
}

/// Every `.rs` file under `dir`, recursively.
fn collect_rust_files(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
  let Ok(entries) = std::fs::read_dir(dir) else {
    return;
  };
  for entry in entries.flatten() {
    let path = entry.path();
    if path.is_dir() {
      collect_rust_files(&path, out);
    } else if path.extension().is_some_and(|e| e == "rs") {
      out.push(path);
    }
  }
}

/// Each `MODELS_LOCK` table's declared `kit`, keyed by table name.
fn kits_of(tables: &[LockTable]) -> BTreeMap<&str, &str> {
  tables
    .iter()
    .map(|t| {
      (
        t.name.as_str(),
        t.fields
          .get("kit")
          .unwrap_or_else(|| panic!("MODELS_LOCK table {:?} has no `kit`", t.name))
          .as_str(),
      )
    })
    .collect()
}

/// Every row's [`ArtifactNames`], read off the lock table that stages it and
/// the loader locator it names.
fn artifact_names<'row>(
  rows: &'row [Artifact],
  tables: &[LockTable],
) -> BTreeMap<&'row str, ArtifactNames> {
  rows
    .iter()
    .map(|row| {
      let table = tables
        .iter()
        .find(|t| t.name == row.staged_by)
        .unwrap_or_else(|| {
          panic!(
            "{}: staged_by {:?} names no MODELS_LOCK table",
            row.file, row.staged_by
          )
        });
      let local_dir = table
        .fields
        .get("local-dir")
        .unwrap_or_else(|| panic!("MODELS_LOCK table {:?} has no `local-dir`", table.name))
        .clone();
      let kit_dir = local_dir
        .rsplit('/')
        .next()
        .unwrap_or(local_dir.as_str())
        .to_string();
      let (_, module) = row.loader.split_once("::").unwrap_or_else(|| {
        panic!(
          "{}: loader {:?} is not `<source>::<module>`",
          row.file, row.loader
        )
      });
      (
        row.file,
        ArtifactNames {
          local_dir,
          kit_dir,
          module: module.to_string(),
        },
      )
    })
    .collect()
}

/// Every `[[test]]`, `[[bench]]` and `[[example]]` target `manifest` declares,
/// each with the source set it compiles under no `commercial-` feature.
///
/// # Fails closed
///
/// A target with no `name`, no `path`, or a `required-features` that is not an
/// array of strings PANICS. Those are the three fields the reconciliation runs
/// on, and a target this reader quietly dropped is a suite direction 2 would
/// clear without ever having read it.
///
/// `read` hands back BYTES rather than text, because the source set is not all
/// source: an `include_bytes!` target may be a binary fixture, and "not on
/// disk" and "on disk but not UTF-8" are two different answers this reader
/// must not conflate — the first is a refusal, the second is a file that
/// contributes no names.
fn compiled_targets(manifest: &str, read: &dyn Fn(&str) -> Option<Vec<u8>>) -> Vec<CompiledTarget> {
  let document: toml::Table = toml::from_str(manifest).unwrap_or_else(|e| {
    panic!(
      "the manifest under test does not decode as TOML: {e}. The TEST channel of direction 2 \
       reads every target's `required-features` out of it; a manifest it cannot read is a set \
       of claims it cannot reconcile."
    )
  });
  let mut targets = Vec::new();
  for kind in ["test", "bench", "example"] {
    let Some(declared) = document.get(kind) else {
      continue;
    };
    let entries = declared
      .as_array()
      .unwrap_or_else(|| panic!("the manifest's `{kind}` entry is not an array of tables"));
    for entry in entries {
      let name = entry
        .get("name")
        .and_then(toml::Value::as_str)
        .unwrap_or_else(|| panic!("a `[[{kind}]]` target declares no `name`"))
        .to_string();
      let path = entry
        .get("path")
        .and_then(toml::Value::as_str)
        .unwrap_or_else(|| {
          panic!(
            "the `[[{kind}]]` target {name:?} declares no `path`. This reader resolves a \
             target's source set from its entry file, so a target without one is a target \
             whose suites it cannot enumerate."
          )
        })
        .to_string();
      let required_features = entry
        .get("required-features")
        .map_or_else(Vec::new, |value| {
          value
            .as_array()
            .unwrap_or_else(|| {
              panic!(
                "the `[[{kind}]]` target {name:?} has a `required-features` that is not an array"
              )
            })
            .iter()
            .map(|feature| {
              feature
                .as_str()
                .unwrap_or_else(|| {
                  panic!(
                    "the `[[{kind}]]` target {name:?} has a non-string entry in `required-features`"
                  )
                })
                .to_string()
            })
            .collect()
        });
      let sources = ordinary_sources(&path, read);
      targets.push(CompiledTarget {
        kind: kind.to_string(),
        name,
        path,
        required_features,
        sources,
      });
    }
  }
  targets
}

/// The files a target compiles when no `commercial-` feature is enabled: its
/// entry `path`, plus every module those files declare and every file they
/// `include!`, transitively — each with the text of whatever it embeds with
/// `include_str!`/`include_bytes!`.
///
/// # Refuses rather than skips
///
/// A missing entry file, a `mod` that resolves to no file, a `mod` that
/// resolves to BOTH `x.rs` and `x/mod.rs`, and an `include!`/`include_str!`/
/// `include_bytes!` whose file is not on disk all panic. A source set this
/// reader silently gave up on is a target whose suites direction 2 would clear
/// without having read them.
///
/// # What it still cannot see, stated rather than chased
///
/// This closes the source set under the two splicings whose spelling is exact
/// — a `mod` declaration and an `include`-family literal, both of which name a
/// file rustc resolves the same way every time. It does not close it under
/// macro-generated modules and code, build-script output, environment
/// variables read at run time, or paths a target composes at run time out of
/// separately harmless parts. Those are not name-scannable, and refusing every
/// `macro_rules!` instead would red this repository's own helper macros for no
/// finding. What it will not do is stay quiet about a construct that reaches
/// for one of them: an `include`-family invocation whose path is a `concat!`
/// or an `env!` rather than a literal is refused, not skipped
/// ([`include_literal`]). The guarantee this channel makes is a name TRIPWIRE
/// over what the ordinary build compiles, and the module doc says so where it
/// defines `wired`.
fn ordinary_sources(entry: &str, read: &dyn Fn(&str) -> Option<Vec<u8>>) -> Vec<OrdinarySource> {
  let mut sources: Vec<OrdinarySource> = Vec::new();
  // The entry file is the test crate's ROOT, and a root's module directory is
  // its own directory — `tests/align/model_io.rs` reaches `mod common;` at
  // `tests/align/common/mod.rs`, not at `tests/align/model_io/common/mod.rs`.
  let mut pending = vec![(entry.to_string(), true)];
  let mut seen = BTreeSet::from([entry.to_string()]);
  let mut next = 0;
  while next < pending.len() {
    let (path, root) = pending[next].clone();
    next += 1;
    let text = source_text(&path, read);
    let scan = scan_source(&text, &path);
    for declared in &scan.mods {
      let file = module_file(&path, root, declared, read);
      if seen.insert(file.clone()) {
        pending.push((file, false));
      }
    }
    let (dir, _) = split_file(&path);
    let mut embedded = Vec::new();
    for include in &scan.includes {
      // Every `include`-family macro spells its file relative to the
      // DIRECTORY OF THE FILE THE INVOCATION SITS IN, which is why this
      // resolves against `dir` and not against the module directory a `mod`
      // would use.
      let file = normalised(&joined_path(dir, &include.spelled));
      let bytes = read(&file).unwrap_or_else(|| {
        panic!(
          "{path}: `{}!({:?})` resolves to {file:?}, which is not on disk. A file the compiler \
           splices in is a file this channel has to read, so an unresolvable one is refused \
           exactly as an unresolvable `mod` is.",
          include.macro_name, include.spelled
        )
      });
      if include.macro_name == "include" {
        // `include!` splices RUST, so the file joins the source set. rustc
        // treats it as a module ROOT — verified against the compiler: a `mod
        // helper;` inside `src/inner/spliced.rs` resolves to
        // `src/inner/helper.rs`, and rustc names exactly that candidate and
        // `src/inner/helper/mod.rs` when neither is there.
        if seen.insert(file.clone()) {
          pending.push((file, true));
        }
        continue;
      }
      embedded.push(EmbeddedFile {
        text: String::from_utf8(bytes).ok(),
        path: file,
      });
    }
    sources.push(OrdinarySource {
      path,
      literals: scan.literals,
      idents: scan.idents,
      embedded,
    });
  }
  sources
}

/// One compiled Rust file's text, refusing both ways it can be unreadable.
fn source_text(path: &str, read: &dyn Fn(&str) -> Option<Vec<u8>>) -> String {
  let bytes = read(path).unwrap_or_else(|| {
    panic!(
      "the target source {path:?} is not on disk. A target whose sources cannot be enumerated is \
       one the TEST channel of direction 2 cannot clear, so this refuses rather than reading it \
       as a target that references nothing."
    )
  });
  String::from_utf8(bytes).unwrap_or_else(|_| {
    panic!(
      "the target source {path:?} is not valid UTF-8, so it is not source this reader can search \
       — and it is not source rustc can compile either"
    )
  })
}

/// Where a `mod` declaration's file lives, by the rule rustc applies.
///
/// Without a `#[path]` the candidates are `<module dir>/<name>.rs` and
/// `<module dir>/<name>/mod.rs`, where the module directory of a crate root or
/// a `mod.rs` is its own directory and that of any other file is a directory
/// named after the file. With one, the spelling is relative to the directory
/// the declaring FILE sits in — plus the inline module names, when the
/// declaration is nested inside `mod x { … }`.
fn module_file(
  declaring: &str,
  root: bool,
  decl: &ModDecl,
  read: &dyn Fn(&str) -> Option<Vec<u8>>,
) -> String {
  let (dir, stem) = split_file(declaring);
  let mod_rs = root || stem == "mod";
  let module_dir = if mod_rs {
    dir.to_string()
  } else {
    joined_path(dir, stem)
  };
  let nested = decl.inside.join("/");
  if let Some(spelled) = &decl.path_attr {
    let base = if decl.inside.is_empty() {
      dir.to_string()
    } else {
      joined_path(&module_dir, &nested)
    };
    let file = normalised(&joined_path(&base, spelled));
    assert!(
      read(&file).is_some(),
      "{declaring}: `#[path = {spelled:?}] mod {}` resolves to {file:?}, which is not on disk",
      decl.name
    );
    return file;
  }
  let base = if decl.inside.is_empty() {
    module_dir
  } else {
    joined_path(&module_dir, &nested)
  };
  let flat = normalised(&joined_path(&base, &format!("{}.rs", decl.name)));
  let folder = normalised(&joined_path(&base, &format!("{}/mod.rs", decl.name)));
  match (read(&flat).is_some(), read(&folder).is_some()) {
    (true, false) => flat,
    (false, true) => folder,
    (true, true) => panic!(
      "{declaring}: `mod {}` resolves to BOTH {flat:?} and {folder:?}. rustc refuses that too, \
       and a reader that picked one would be enumerating a source set the compiler never has.",
      decl.name
    ),
    (false, false) => panic!(
      "{declaring}: `mod {}` resolves to neither {flat:?} nor {folder:?}. A module this reader \
       cannot find is code it cannot search for a restricted artifact.",
      decl.name
    ),
  }
}

/// A path's directory and its file stem, `/`-separated and extension-free.
fn split_file(path: &str) -> (&str, &str) {
  let (dir, name) = path.rsplit_once('/').unwrap_or(("", path));
  (dir, name.strip_suffix(".rs").unwrap_or(name))
}

/// `dir/tail`, or `tail` alone when `dir` is the package root.
fn joined_path(dir: &str, tail: &str) -> String {
  if dir.is_empty() {
    tail.to_string()
  } else {
    format!("{dir}/{tail}")
  }
}

/// A `/`-separated path with `.` and `..` resolved lexically.
fn normalised(path: &str) -> String {
  let mut parts: Vec<&str> = Vec::new();
  for part in path.split('/') {
    match part {
      "" | "." => {}
      ".." => assert!(
        parts.pop().is_some(),
        "the module path {path:?} climbs above the package root"
      ),
      named => parts.push(named),
    }
  }
  parts.join("/")
}

/// One file's tokens, reduced to what [`ordinary_sources`] and
/// [`CompiledTarget::references_to`] read out of them.
struct FileScan {
  /// Every string literal in code no `commercial-` feature gates.
  literals: Vec<String>,
  /// Every identifier in that same code.
  idents: BTreeSet<String>,
  /// Every `mod` declaration in it whose body is another file.
  mods: Vec<ModDecl>,
  /// Every `include!`, `include_str!` and `include_bytes!` in that same code.
  includes: Vec<IncludeDecl>,
}

/// One `include`-family invocation, which names a file the compiler splices in.
struct IncludeDecl {
  /// `include`, `include_str` or `include_bytes`.
  macro_name: String,
  /// The path literal, as it is spelled — relative to the directory of the
  /// file the invocation sits in.
  spelled: String,
}

/// One `mod` declaration whose body lives in another file.
struct ModDecl {
  /// The declared name.
  name: String,
  /// Its `#[path = "…"]`, when it carries one.
  path_attr: Option<String>,
  /// The inline `mod x { … }` blocks it sits inside, outermost first.
  inside: Vec<String>,
}

/// One source file read as the compiler would read it with no `commercial-`
/// feature enabled.
///
/// TOKENS, for the reason [`cfg_features_in`] already gives: the lexer has
/// decided what is a comment (gone), what is a string (one `Literal`) and what
/// is an attribute, so no rule here has to approximate that. The one thing
/// added on top is `syn` — [`required_features`] over each attribute run, so a
/// `#[cfg]` is read here in exactly the words [`gates_of_module`] reads it in.
fn scan_source(source: &str, whence: &str) -> FileScan {
  syn::parse_file(source).unwrap_or_else(|e| {
    panic!(
      "{whence} does not parse as Rust ({e}). This reader enumerates what a target COMPILES; \
       source it cannot read is source whose references to a restricted artifact it cannot see."
    )
  });
  let tokens: proc_macro2::TokenStream = source.parse().unwrap_or_else(|e| {
    panic!("{whence} does not tokenise ({e}); its source set cannot be read");
  });
  let mut scan = FileScan {
    literals: Vec::new(),
    idents: BTreeSet::new(),
    mods: Vec::new(),
    includes: Vec::new(),
  };
  scan_tokens(tokens, &mut Vec::new(), whence, &mut scan);
  scan
}

/// Walks one token stream, dropping every construct a `commercial-` feature
/// gates and collecting the literals, identifiers, `mod` declarations and
/// `include`-family invocations of what is left.
///
/// **Attributes are not scanned.** An attribute is not code, and `#[doc]` —
/// what every `//!` and `///` becomes — would otherwise make the PROSE of
/// `tests/face/arcface/mod.rs` ("staged under `Models/facekit/`")
/// indistinguishable from a load. The three things read out of an attribute
/// run are the `#[cfg(feature = ...)]` that decides whether the construct is
/// here at all, the `#[path = "…"]` that says where a module's file is, and
/// the `#[cfg_attr(…)]` that could quietly be either of those two
/// ([`AttributeRun`]).
fn scan_tokens(
  tokens: proc_macro2::TokenStream,
  inside: &mut Vec<String>,
  whence: &str,
  out: &mut FileScan,
) {
  use proc_macro2::{Delimiter, Punct, Spacing, TokenStream, TokenTree};
  let trees: Vec<TokenTree> = tokens.into_iter().collect();
  let mut at = 0;
  while at < trees.len() {
    let mut outer = TokenStream::new();
    while at < trees.len() {
      let TokenTree::Punct(hash) = &trees[at] else {
        break;
      };
      if hash.as_char() != '#' {
        break;
      }
      let mut body_at = at + 1;
      let inner = matches!(trees.get(body_at), Some(TokenTree::Punct(p)) if p.as_char() == '!');
      if inner {
        body_at += 1;
      }
      let Some(TokenTree::Group(body)) = trees.get(body_at) else {
        break;
      };
      if body.delimiter() != Delimiter::Bracket {
        break;
      }
      // Rendered back as an OUTER attribute either way: `#![cfg(..)]` and
      // `#[cfg(..)]` name the same feature, and only the scope they apply to
      // differs.
      let rendered = TokenStream::from_iter([
        TokenTree::Punct(Punct::new('#', Spacing::Alone)),
        trees[body_at].clone(),
      ]);
      at = body_at + 1;
      if inner {
        // `#![…]` applies to the whole enclosing block rather than to the item
        // after it, so a `commercial-` gate here removes everything still to
        // come in this stream.
        if attribute_run(rendered, whence)
          .gates
          .iter()
          .any(|gate| gate.starts_with(COMMERCIAL_PREFIX))
        {
          return;
        }
      } else {
        outer.extend(rendered);
      }
    }
    if at >= trees.len() {
      break;
    }
    let run = attribute_run(outer, whence);
    if run
      .gates
      .iter()
      .any(|gate| gate.starts_with(COMMERCIAL_PREFIX))
    {
      skip_item(&trees, &mut at);
      continue;
    }
    // `pub mod x;` and `pub(crate) mod x;` declare modules too.
    let mut cursor = at;
    if matches!(trees.get(cursor), Some(TokenTree::Ident(vis)) if vis == "pub") {
      cursor += 1;
      if matches!(trees.get(cursor), Some(TokenTree::Group(g)) if g.delimiter() == Delimiter::Parenthesis)
      {
        cursor += 1;
      }
    }
    if matches!(trees.get(cursor), Some(TokenTree::Ident(kw)) if kw == "mod")
      && let Some(TokenTree::Ident(name)) = trees.get(cursor + 1)
    {
      // `mod $m { … }` inside a `macro_rules!` body is NOT a declaration: the
      // token after `mod` is a `$`, not an identifier, so it never lands here.
      match trees.get(cursor + 2) {
        Some(TokenTree::Punct(semi)) if semi.as_char() == ';' => {
          run.refuse_a_conditional_module_attribute(whence, &name.to_string());
          out.mods.push(ModDecl {
            name: name.to_string(),
            path_attr: run.path_attr,
            inside: inside.clone(),
          });
          at = cursor + 3;
          continue;
        }
        Some(TokenTree::Group(body)) if body.delimiter() == Delimiter::Brace => {
          run.refuse_a_conditional_module_attribute(whence, &name.to_string());
          inside.push(name.to_string());
          scan_tokens(body.stream(), inside, whence, out);
          inside.pop();
          at = cursor + 3;
          continue;
        }
        _ => {}
      }
    }
    match &trees[at] {
      TokenTree::Literal(value) => {
        if let Ok(text) = syn::parse_str::<syn::LitStr>(&value.to_string()) {
          out.literals.push(text.value());
        }
      }
      TokenTree::Ident(name) => {
        // `include!`, `include_str!` and `include_bytes!` each splice a file
        // the compiler names by a LITERAL, which is the one other spelling
        // that widens the source set exactly. The literal itself still falls
        // through to the arm above on the next pass, so a path that spells the
        // artifact is caught as a string as well as resolved as a file.
        if let Some(spelled) = include_literal(&trees, at, whence) {
          out.includes.push(IncludeDecl {
            macro_name: name.to_string(),
            spelled,
          });
        }
        out.idents.insert(name.to_string());
      }
      TokenTree::Group(group) => scan_tokens(group.stream(), inside, whence, out),
      TokenTree::Punct(_) => {}
    }
    at += 1;
  }
}

/// One attribute run, reduced to the three things the source-set reader has to
/// know about it.
struct AttributeRun {
  /// The features the construct REQUIRES, by [`required_features`].
  gates: BTreeSet<String>,
  /// Its `#[path = "…"]`, when it carries one directly.
  path_attr: Option<String>,
  /// Every `#[cfg_attr(…)]` in the run: the attributes it would attach, and
  /// the line it sits on.
  conditional: Vec<ConditionalAttribute>,
}

/// One `#[cfg_attr(<predicate>, <attached>…)]`.
struct ConditionalAttribute {
  /// The line of the file it sits on.
  line: usize,
  /// The head identifier of each attribute it would attach — `path`, `cfg`,
  /// `allow`, `serde`, … — in order.
  attaches: Vec<String>,
}

impl AttributeRun {
  /// Refuses a `cfg_attr` on a `mod` declaration that could attach a `#[path]`,
  /// a further `#[cfg]`, or another `cfg_attr` carrying either one.
  ///
  /// **The finding this exists for.** `#[cfg_attr(feature = "face", path =
  /// "restricted.rs")] mod helper;` compiles `restricted.rs` under `face` and
  /// `helper.rs` without it. A resolver that read only the DIRECT `#[path]`
  /// saw neither condition, resolved `helper.rs`, and searched a file the
  /// compiler was not building — so a reference in `restricted.rs` was
  /// invisible to the whole channel. The `cfg(…)` form is the same defect
  /// pointed at existence rather than location: it decides whether the module
  /// is compiled AT ALL, which is the one question this reader answers.
  ///
  /// Evaluating the predicate is not on offer: a `cfg_attr` is a conditional
  /// this file's readers deliberately do not evaluate anywhere — see
  /// [`required_features`], which refuses one on a loader for the same reason.
  /// So the fail-closed answer is to refuse the source, naming the file and
  /// the line, and let a human reshape it. Everything else a `cfg_attr`
  /// attaches — `allow`, `doc`, a test-only lint — changes no file and no
  /// existence, and stays ungated exactly as before.
  ///
  /// `cfg_attr` is in the refused set beside `path` and `cfg` because it
  /// NESTS: `#[cfg_attr(test, cfg_attr(feature = "face", path =
  /// "restricted.rs"))]` is the same redirection one level down. What this
  /// reads is the HEAD identifier of each attribute the run would attach, so
  /// the answer to a form that could carry either of the other two forward is
  /// to refuse it rather than to unwrap the tree and start evaluating.
  fn refuse_a_conditional_module_attribute(&self, whence: &str, module: &str) {
    for conditional in &self.conditional {
      for attached in &conditional.attaches {
        assert!(
          !matches!(attached.as_str(), "path" | "cfg" | "cfg_attr"),
          "{whence}:{}: `mod {module}` carries a `cfg_attr` that would attach `{attached}`. That \
           makes the module's FILE, or whether it is compiled at all, conditional on a predicate \
           this reader does not evaluate — so the source set it enumerated would be one the \
           compiler never has, and a reference in the redirected file would be invisible. Spell \
           the condition as a plain `#[cfg]` on the declaration instead.",
          conditional.line
        );
      }
    }
  }
}

/// One attribute run's required features, its `#[path = "…"]`, and its
/// `cfg_attr`s.
///
/// # Fails closed, in the direction this reader needs
///
/// [`required_features`] refuses every `cfg` spelling that is not the positive
/// `#[cfg(feature = "name")]`, and an `Err` here means the item STAYS in the
/// ordinary source set. That is the safe direction for this channel and the
/// opposite of the one [`gates_of_module`] needs: there an unreadable `#[cfg]`
/// must not become a gate somebody is reassured by; here it must not become an
/// exclusion that hides a reference. An item under
/// `#[cfg(all(feature = "face", feature = "commercial-face-arcface"))]` is
/// therefore read as ordinary and reported — a finding to reshape, not a hole.
///
/// # And it refuses a `cfg_attr` predicated on a commercial feature
///
/// That one cannot fail closed by being read as ordinary. `#[cfg_attr(feature
/// = "commercial-face-arcface", …)]` is a gate whose effect this reader does
/// not evaluate, anywhere in the ordinary source set, so it is refused on
/// sight — file and line — rather than resolved one way and believed. The
/// module-scoped half of the same rule is
/// [`AttributeRun::refuse_a_conditional_module_attribute`].
fn attribute_run(attrs: proc_macro2::TokenStream, whence: &str) -> AttributeRun {
  use syn::parse::Parser as _;
  let parsed = syn::Attribute::parse_outer
    .parse2(attrs)
    .unwrap_or_else(|e| panic!("{whence}: an attribute this reader cannot parse ({e})"));
  let gates = required_features(&parsed).unwrap_or_default();
  let path_attr = parsed.iter().find_map(|attr| {
    if !attr.path().is_ident("path") {
      return None;
    }
    let syn::Meta::NameValue(pair) = &attr.meta else {
      return None;
    };
    let syn::Expr::Lit(syn::ExprLit {
      lit: syn::Lit::Str(spelled),
      ..
    }) = &pair.value
    else {
      return None;
    };
    Some(spelled.value())
  });
  let mut conditional = Vec::new();
  for attr in &parsed {
    if !attr.path().is_ident("cfg_attr") {
      continue;
    }
    let syn::Meta::List(list) = &attr.meta else {
      panic!("{whence}: a `cfg_attr` with no argument list, which is not a `cfg_attr` at all");
    };
    let line = attr.path().segments[0].ident.span().start().line;
    let (predicate, attaches) = cfg_attr_parts(&list.tokens);
    let mut named = BTreeSet::new();
    collect_feature_names(predicate, &mut named);
    if let Some(gate) = named
      .iter()
      .find(|name| name.starts_with(COMMERCIAL_PREFIX))
    {
      panic!(
        "{whence}:{line}: a `cfg_attr` predicated on {gate:?}. A `commercial-` feature decides \
         what this channel may find, and a `cfg_attr` is a conditional this file's readers do \
         not evaluate — so whichever way it were resolved, the ordinary source set would be a \
         guess. Spell the gate as a plain `#[cfg(feature = ...)]` instead."
      );
    }
    conditional.push(ConditionalAttribute { line, attaches });
  }
  AttributeRun {
    gates,
    path_attr,
    conditional,
  }
}

/// A `cfg_attr`'s argument list split into its PREDICATE and the head
/// identifier of each attribute it would attach.
///
/// The split is at the top-level commas, because that is where the grammar
/// puts it: `cfg_attr(<predicate>, <attr>, <attr>…)`. Reading the whole list
/// as a predicate would let `cfg_attr(test, cfg(feature = "x"))` contribute
/// `x` as if the item required it.
fn cfg_attr_parts(tokens: &proc_macro2::TokenStream) -> (proc_macro2::TokenStream, Vec<String>) {
  use proc_macro2::{TokenStream, TokenTree};
  let mut chunks: Vec<Vec<TokenTree>> = vec![Vec::new()];
  for tree in tokens.clone() {
    if matches!(&tree, TokenTree::Punct(comma) if comma.as_char() == ',') {
      chunks.push(Vec::new());
      continue;
    }
    chunks
      .last_mut()
      .expect("`chunks` is seeded with one element and only ever grows")
      .push(tree);
  }
  let mut chunks = chunks.into_iter();
  let predicate = TokenStream::from_iter(chunks.next().unwrap_or_default());
  let attaches = chunks
    .filter_map(|chunk| {
      chunk.into_iter().find_map(|tree| match tree {
        TokenTree::Ident(head) => Some(head.to_string()),
        _ => None,
      })
    })
    .collect();
  (predicate, attaches)
}

/// The path literal of an `include`-family invocation starting at `at`, if
/// that is what sits there.
///
/// `include!`, `include_str!` and `include_bytes!` are the three macros whose
/// argument is a file the compiler resolves by NAME, which is the only reason
/// a name scan can follow them at all.
///
/// **An invocation whose argument is not one string literal is refused, not
/// skipped.** `include_str!(concat!(env!("OUT_DIR"), "/plan.json"))` names a
/// file through a composition this reader does not evaluate, and returning
/// `None` for it would drop that file out of the source set in SILENCE — the
/// one thing every other resolver here refuses to do. The stated residual is
/// then honest: what is beyond this channel is beyond it loudly.
fn include_literal(trees: &[proc_macro2::TokenTree], at: usize, whence: &str) -> Option<String> {
  use proc_macro2::TokenTree;
  let TokenTree::Ident(name) = &trees[at] else {
    return None;
  };
  let macro_name = name.to_string();
  if !matches!(
    macro_name.as_str(),
    "include" | "include_str" | "include_bytes"
  ) {
    return None;
  }
  if !matches!(trees.get(at + 1), Some(TokenTree::Punct(bang)) if bang.as_char() == '!') {
    return None;
  }
  let Some(TokenTree::Group(args)) = trees.get(at + 2) else {
    return None;
  };
  let mut inner = args.stream().into_iter();
  let sole = match (inner.next(), inner.next()) {
    (Some(tree), None) => Some(tree),
    _ => None,
  };
  let spelled = sole
    .and_then(|tree| syn::parse_str::<syn::LitStr>(&tree.to_string()).ok())
    .unwrap_or_else(|| {
      panic!(
        "{whence}: `{macro_name}!({})` — its argument is not one string literal, so the file it \
         splices in is named by something this reader does not evaluate: a `concat!`, an `env!`, \
         a build script's output. Skipping the invocation would drop that file out of the source \
         set in silence, which is the one thing this channel must not do. Spell the path as a \
         literal.",
        args.stream()
      )
    });
  Some(spelled.value())
}

/// Advances `at` past one attributed construct whose attributes have already
/// been consumed.
///
/// # It is not only ITEMS that carry attributes
///
/// That assumption was the defect. An item ends at a brace group or a `;`, but
/// a `#[cfg]` is equally legal on a MATCH ARM, a struct or enum field, an enum
/// variant, an element of a tuple, array or call, and a closure parameter —
/// and every one of those ends at a `,` at this token level instead. Under the
/// brace-or-semicolon rule alone,
///
/// ```text
///   #[cfg(feature = "commercial-face-arcface")]
///   0 => (),
///   _ => { load("Models/facekit/…") }
/// ```
///
/// skipped past the comma and consumed the UNGATED arm's brace group unread,
/// so the gate hid the very reference the channel exists to find. A `,` at
/// this level therefore ends the skip too.
///
/// # The residual, and its direction
///
/// A `,` inside a generic argument list is NOT inside a token-tree group —
/// `<` and `>` are ordinary puncts — so `#[cfg(…)] fn f<A, B>() { … }` stops
/// the skip at that comma and the rest of the signature and body is scanned as
/// ordinary code. That is FAIL-CLOSED: it can only ADD names to the ordinary
/// set, so the worst it can produce is a false RED on a real target, never a
/// false green. If one ever appears, the answer is to report it and reshape
/// the source, not to weaken the rule. The same holds for the older stopping
/// case the brace rule already had (`static X: fn() = || {};` ends at the
/// closure's braces and leaves a stray `;` to be scanned like any other
/// token). What the rule must never do is stop LATE, which is the direction
/// that swallows the construct AFTER the gated one and hides a reference with
/// it.
fn skip_item(trees: &[proc_macro2::TokenTree], at: &mut usize) {
  use proc_macro2::{Delimiter, TokenTree};
  while *at < trees.len() {
    let tree = &trees[*at];
    *at += 1;
    match tree {
      TokenTree::Group(body) if body.delimiter() == Delimiter::Brace => return,
      TokenTree::Punct(end) if end.as_char() == ';' || end.as_char() == ',' => return,
      _ => {}
    }
  }
}

/// `MODELS_LOCK`'s tables reduced to [`StagedTable`] — where each downloads to,
/// and what it selects.
fn staged_tables(tables: &[LockTable]) -> Vec<StagedTable> {
  tables
    .iter()
    .map(|t| {
      let local_dir = t
        .fields
        .get("local-dir")
        .unwrap_or_else(|| panic!("MODELS_LOCK table {:?} has no `local-dir`", t.name));
      let vendor_dir = local_dir
        .strip_prefix("Models/")
        .unwrap_or_else(|| panic!("`local-dir` {local_dir:?} does not start with `Models/`"))
        .to_string();
      let selection = match (t.fields.get("files"), t.fields.get("include")) {
        (Some(files), None) => {
          Selection::Files(files.split_whitespace().map(str::to_string).collect())
        }
        (None, Some(include)) => {
          Selection::Include(include.split_whitespace().map(str::to_string).collect())
        }
        (Some(_), Some(_)) => panic!(
          "MODELS_LOCK table {:?} declares BOTH `files` and `include`; this reader cannot say \
           which selects, and guessing would decide coverage",
          t.name
        ),
        (None, None) => panic!(
          "MODELS_LOCK table {:?} declares neither `files` nor `include`, so nothing can be said \
           about what it stages",
          t.name
        ),
      };
      StagedTable {
        name: t.name.clone(),
        vendor_dir,
        selection,
      }
    })
    .collect()
}

/// The table-relative file set a row demonstrably accounts for.
///
/// A row keyed on one file inside a `.mlmodelc` covers the WHOLE bundle only
/// when the pin it names is a per-file manifest — the coverage set is then read
/// out of that manifest, not assumed from the bundle path. A row that names a
/// bundle directly ([`Key::Unmanifested`]) covers the bundle by declaration,
/// which is exactly why that key carries a reason.
fn row_coverage<'a>(row: &'a Artifact, table: &StagedTable) -> Covered<'a> {
  let tail = table
    .table_relative(row.file)
    .unwrap_or_else(|| panic!("{}: not under {}/", row.file, table.vendor_dir));
  let mut covered = BTreeSet::from([tail.to_string()]);
  if !row.pin.is_empty()
    && let Pins::Manifest(manifest) = pins_at(row.pin)
    && let Some(bundle) = row.bundle()
  {
    let bundle_tail = table
      .table_relative(bundle)
      .unwrap_or_else(|| panic!("{bundle}: not under {}/", table.vendor_dir));
    // The BUNDLE itself, because a manifest pin is what makes the row cover
    // the whole of it rather than the one file it keys on. A row whose pin is
    // a bare hex literal, or which has no pin, gets no bundle entry — and is
    // then correctly NOT a row over its bundle.
    covered.insert(bundle_tail.to_string());
    for key in manifest.keys() {
      covered.insert(format!("{bundle_tail}/{key}"));
    }
  }
  Covered {
    file: row.file,
    staged_by: row.staged_by,
    covered,
  }
}

/// The rows, each paired with what it covers under its own table.
fn coverage<'a>(rows: &'a [Artifact], tables: &[StagedTable]) -> Vec<Covered<'a>> {
  rows
    .iter()
    .map(|row| {
      tables.iter().find(|t| t.name == row.staged_by).map_or_else(
        || Covered {
          file: row.file,
          staged_by: row.staged_by,
          covered: BTreeSet::new(),
        },
        |table| row_coverage(row, table),
      )
    })
    .collect()
}

/// Every row's loader gate, read from the tree — the map directions 2 and 3
/// run on.
fn derived_gates(rows: &[Artifact]) -> BTreeMap<&str, BTreeSet<String>> {
  rows
    .iter()
    .map(|row| (row.file, loader_gates(row.loader)))
    .collect()
}

/// Every declared feature's closure, keyed by the feature it is seeded from.
fn feature_closures(block: &str) -> BTreeMap<String, BTreeSet<String>> {
  feature_names(block)
    .into_iter()
    .map(|f| {
      let closure = feature_closure(block, &f);
      (f, closure)
    })
    .collect()
}

// ---------------------------------------------------------------------------
// The live checks — the real table, the real lock, the real manifest
// ---------------------------------------------------------------------------

/// **Direction 1.** Every file the lock NAMES is covered by a licence row, and
/// every row names a file its own table stages.
///
/// Both halves, because either one alone rots: coverage-only lets a row outlive
/// the table it describes, and reverse-only lets a new table arrive with nobody
/// having asked what its bytes permit. And both at FILE granularity — see
/// [`unmatched_coverage`] for why the repository-name comparison this replaces
/// could not see three staged files behind one row.
#[test]
fn every_staged_file_has_a_licence_row_and_every_row_names_a_staged_file() {
  let Some(tables) = lock_tables() else {
    return;
  };
  assert!(
    tables.len() >= 8,
    "only {} MODELS_LOCK tables parsed; this reader has stopped matching the lock's shape and \
     would pass vacuously",
    tables.len()
  );
  let staged = staged_tables(&tables);
  let named: usize = staged
    .iter()
    .filter_map(|t| match &t.selection {
      Selection::Files(files) => Some(files.len()),
      Selection::Include(_) => None,
    })
    .sum();
  assert!(
    named >= 3,
    "no MODELS_LOCK table names an explicit `files` list any more ({named} named files), so the \
     exact-bijection half of this check sees nothing and would pass vacuously"
  );
  let rows = coverage(ARTIFACTS, &staged);
  let failures = unmatched_coverage(&staged, &rows);
  assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Bundles the fp16 sweep pins under a staged vendor directory that no licence
/// row covers.
///
/// The forward half of direction 1 can only be exact where `MODELS_LOCK` names
/// its files; a globbed table's contents exist only after a download. This is a
/// SECOND, independent enumeration of what those globs bring in — the bundle
/// paths `tests/fp16_guards.rs` pins guard sites for — and it is a repository
/// fact rather than a restatement of this table. Without it, "every bundle a
/// glob stages has a row" would rest on nobody having forgotten one.
fn fp16_pinned_bundles_without_a_row(
  pinned: &[String],
  tables: &[StagedTable],
  rows: &[Covered<'_>],
) -> Vec<String> {
  let mut failures = Vec::new();
  for path in pinned {
    // Two MODELS_LOCK tables can share a `local-dir` — speakerkit's base and
    // overlay do — so the question is whether ANY of them has a row over this
    // bundle, not whether the first one does.
    let candidates: Vec<&StagedTable> = tables
      .iter()
      .filter(|t| path.starts_with(&format!("{}/", t.vendor_dir)))
      .collect();
    if candidates.is_empty() {
      // Pinned under a vendor no MODELS_LOCK table stages. ci.yml's own
      // `UNSTAGED_DEFECT_VENDORS` records that gap; it is not this file's.
      continue;
    }
    let covered = candidates.iter().any(|table| {
      let tail = table
        .table_relative(path)
        .expect("the prefix was just matched");
      rows
        .iter()
        .any(|r| r.staged_by == table.name && r.covers(tail))
    });
    if !covered {
      let names: Vec<&str> = candidates.iter().map(|t| t.name.as_str()).collect();
      failures.push(format!(
        "tests/fp16_guards.rs pins guard sites in {path:?}, which MODELS_LOCK stages ({}) and no \
         licence row covers. A glob's contents cannot be enumerated from the lock, so this roster \
         is the second enumeration that stops a staged bundle from having terms nobody wrote \
         down.",
        names.join(", ")
      ));
    }
  }
  failures
}

/// Every `.mlmodelc` path pinned by `tests/fp16_guards.rs`'s defect and
/// load-bearing rosters.
///
/// Reads the `path` FIELD of the roster entries, at token level. The line-based
/// reader this replaces required the literal `path: "` to open a trimmed line,
/// so a rustfmt wrap put a roster entry out of its sight — and a missed entry
/// is one staged bundle whose licence row nobody checks, which is the single
/// thing this second enumeration exists to prevent.
///
/// Anchored on the field NAME rather than on "any string ending in
/// `.mlmodelc`", because that file's prose and its `note` fields both quote
/// bundle paths; widening to every literal would invent bundles instead of
/// missing them.
fn fp16_pinned_bundles() -> Vec<String> {
  let text = read_rel("tests/fp16_guards.rs");
  let tokens: proc_macro2::TokenStream = text.parse().unwrap_or_else(|e| {
    panic!("tests/fp16_guards.rs does not tokenise ({e}); its roster cannot be read")
  });
  let mut paths = Vec::new();
  collect_field_literals(tokens, "path", &mut paths);
  paths.retain(|path| path.ends_with(".mlmodelc"));
  assert!(
    paths.len() >= 8,
    "only {} `.mlmodelc` paths read out of tests/fp16_guards.rs; the reader has stopped matching \
     its rosters and this check would pass vacuously",
    paths.len()
  );
  paths
}

/// Every string literal assigned to a struct-literal field named `field`, at
/// any nesting depth.
fn collect_field_literals(tokens: proc_macro2::TokenStream, field: &str, out: &mut Vec<String>) {
  use proc_macro2::TokenTree;
  let trees: Vec<TokenTree> = tokens.into_iter().collect();
  for (at, tree) in trees.iter().enumerate() {
    match tree {
      TokenTree::Ident(name) if name == field => {
        if let (Some(TokenTree::Punct(colon)), Some(TokenTree::Literal(value))) =
          (trees.get(at + 1), trees.get(at + 2))
          && colon.as_char() == ':'
          && let Ok(literal) = syn::parse_str::<syn::LitStr>(&value.to_string())
        {
          out.push(literal.value());
        }
      }
      TokenTree::Group(group) => collect_field_literals(group.stream(), field, out),
      _ => {}
    }
  }
}

/// **Direction 1's second enumeration.** Every bundle the fp16 sweep pins under
/// a staged vendor directory has a licence row.
#[test]
fn every_fp16_pinned_bundle_under_a_staged_vendor_has_a_licence_row() {
  let Some(tables) = lock_tables() else {
    return;
  };
  let staged = staged_tables(&tables);
  let pinned = fp16_pinned_bundles();
  let matched = pinned
    .iter()
    .filter(|p| {
      staged
        .iter()
        .any(|t| p.starts_with(&format!("{}/", t.vendor_dir)))
    })
    .count();
  assert!(
    matched >= 5,
    "only {matched} of the {} fp16-pinned bundles sit under a staged vendor directory; the vendor \
     names have diverged and this check would pass vacuously",
    pinned.len()
  );
  let rows = coverage(ARTIFACTS, &staged);
  let failures = fp16_pinned_bundles_without_a_row(&pinned, &staged, &rows);
  assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// **Direction 2.** No research-only artifact is WIRED — manifested, staged or
/// tested — by any feature closure that is not itself a commercial opt-in.
///
/// The name is the claim, so it says `wired` and not `reachable`: what this
/// reads is coremlit's own wiring of the artifact, never a caller's ability to
/// load bytes they already hold. See the module doc's residual (issue #138 §8)
/// for the half no check here covers.
///
/// **It has a row in scope**: `facekit/w600k_r50.mlmodelc`, research-only at
/// both layers. What makes it pass is that the tree gates its loader on
/// `commercial-face-arcface` and that no non-`commercial-` feature's closure
/// contains that gate — two live facts, neither of them the row's own claim.
/// Put `commercial-face-arcface` into any ordinary feature's list and this
/// reds with no doctoring at all. `falsifiers::direction_two_*` remain the
/// proof that it can fire on shapes the real table does not currently hold.
#[test]
fn no_research_only_artifact_is_wired_without_a_commercial_gate() {
  let manifest = manifest_text();
  let closures = feature_closures(&manifest);
  let derived = derived_gates(ARTIFACTS);
  let failures = research_only_wired(ARTIFACTS, &derived, &closures);
  assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// **Direction 2's wide clause.** No artifact whose terms leave a shipping
/// claim with nothing to rest on — research-only OR unresolved — is WIRED into
/// `default`.
///
/// `wired`, again, is the precise word and therefore the name: `default` must
/// not manifest, stage or test such an artifact. This test is the MANIFEST
/// third of that; the staged third is
/// [`every_rows_loader_module_is_the_kit_its_lock_table_names`] and the tested
/// third is [`no_ungranted_artifact_is_tested_under_default`]. A consumer who
/// points a public door at bytes of their own is outside every clause here,
/// and the module doc states that residual rather than leaving the name to
/// overstate.
///
/// Twenty-one rows are in scope: the nineteen with an unresolved corpus
/// layer, `redimnet/redimnet_b5.mlmodelc` whose WEIGHTS layer is unresolved,
/// and `facekit/w600k_r50.mlmodelc`, which is here for the stronger reason —
/// its terms are established and they forbid the shipping path. Two live facts
/// are what make it pass — `default = []` in the manifest, and a
/// `#[cfg(feature = ...)]` on every one of those rows' loaders. Delete either
/// and this goes red now, on today's table.
#[test]
fn no_ungranted_artifact_is_wired_into_default() {
  let manifest = manifest_text();
  let closure = feature_closure(&manifest, "default");
  let derived = derived_gates(ARTIFACTS);
  let failures = ungranted_wired_into_default(ARTIFACTS, &derived, &closure);
  assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// **Direction 2's third channel.** No target this crate declares compiles a
/// reference to a research-only artifact without a commercial opt-in.
///
/// The clause the other two could not make. `research_only_wired` reads the
/// `#[cfg]` on the module that MANIFESTS the artifact — which says nothing
/// about a suite that never names that module and opens `Models/facekit`
/// through the generic `FaceEmbedder` door instead. Under
/// `required-features = ["face"]` such a target would test research-only bytes
/// on an ordinary feature with both of the other clauses green.
///
/// Six face targets are in scope today and every one of them passes on a live
/// fact rather than on an absence. `face_align_golden` reaches the artifact by
/// no channel at all; `face_gate` names it only inside
/// `#[cfg(feature = "commercial-face-arcface")] fn …`, so the module the
/// compiler reaches under plain `face` does not name it; and the four suites
/// that pull in `tests/face/arcface/mod.rs` all ask for the commercial feature
/// by name. Move that `#[cfg]` off the gated function, or drop
/// `commercial-face-arcface` from any of the four, and this reds on the real
/// manifest with no doctoring.
///
/// **Its scope is the targets the MANIFEST declares.** Cargo also
/// auto-discovers `tests/*.rs`, and those carry no `required-features` at all;
/// the only files there that name this artifact are the registers that RECORD
/// it — this file's own table — and a predicate that read a record as a load
/// would fail on the very register written to describe the artifact.
///
/// **What passing means, exactly.** This is a FAIL-CLOSED TRIPWIRE over the
/// NAMES the ordinary build compiles — the loader module's file and
/// identifier, the lock's `local-dir` and staged directory name, searched in
/// each target's source, in source it `include!`s, and in the text of a
/// fixture it embeds — and not a proof that an ordinary-feature suite cannot
/// reach the bytes. Outside it: an auto-discovered target, a module or a load
/// a macro generates, build-script output, an environment variable read at run
/// time, and a path composed at run time. The semantic guarantee is
/// [`every_rows_loader_module_is_the_kit_its_lock_table_names`]'s — the only
/// CI shard that stages `Models/facekit` is the kit's own, and that kit must
/// BE the `commercial-`gated loader module — and this test is what catches a
/// target that reaches for the directory anyway.
#[test]
fn no_research_only_artifact_is_tested_without_a_commercial_gate() {
  let Some(tables) = lock_tables() else {
    return;
  };
  let manifest = manifest_text();
  let closures = feature_closures(&manifest);
  let names = artifact_names(ARTIFACTS, &tables);
  let package = Path::new(env!("CARGO_MANIFEST_DIR"));
  let targets = compiled_targets(&manifest, &|rel| std::fs::read(package.join(rel)).ok());
  assert!(
    targets.len() >= 40,
    "only {} declared targets read out of the manifest; this reader has stopped matching its \
     shape and every suite would read as one that references nothing",
    targets.len()
  );
  assert!(
    targets
      .iter()
      .any(|target| target.sources.len() > 1 && target.name.starts_with("face_")),
    "no declared face target resolved a `mod` of its own, so the module resolver has stopped \
     resolving and the fixture module that holds the staged path would be invisible"
  );
  let failures = research_only_tested(ARTIFACTS, &names, &targets, &closures);
  assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// **Direction 2's wide clause, in the test channel.** No target a plain
/// `cargo add coremlit` already builds compiles a reference to an artifact
/// nobody has found a grant for.
///
/// The last of the six cells — two clauses over three channels — and the one
/// with nothing in scope today. `default = []` and every declared target
/// requires a kit feature, so no target's feature closure sits inside
/// `default`'s and the clause cannot fire; both of those are live facts read
/// out of the manifest rather than an empty set, and either one changing reds
/// this. `falsifiers::direction_two_reds_when_a_default_built_target_names_an_ungranted_artifact`
/// is what makes it worth having before that day.
#[test]
fn no_ungranted_artifact_is_tested_under_default() {
  let Some(tables) = lock_tables() else {
    return;
  };
  let manifest = manifest_text();
  let closures = feature_closures(&manifest);
  let default_closure = feature_closure(&manifest, "default");
  let names = artifact_names(ARTIFACTS, &tables);
  let package = Path::new(env!("CARGO_MANIFEST_DIR"));
  let targets = compiled_targets(&manifest, &|rel| std::fs::read(package.join(rel)).ok());
  let failures =
    ungranted_tested_under_default(ARTIFACTS, &names, &targets, &closures, &default_closure);
  assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// **Direction 3.** No `commercial-`prefixed feature gates only artifacts that
/// are granted at both layers, and none gates nothing at all in the source.
///
/// One feature is in scope: `commercial-face-arcface`. It passes on two live
/// facts — `src/embeddings/face/mod.rs` puts that `#[cfg]` on the `arcface`
/// module, so the feature compiles something, and the row behind it is
/// research-only, so the gate stands on a found prohibition. Re-read
/// InsightFace's terms as permissive and this reds, which is the direction
/// people forget: the day an upstream relicenses, the gate becomes false
/// reassurance and has to be retired rather than left standing.
#[test]
fn every_commercial_feature_gates_an_artifact_with_no_shipping_grant() {
  let manifest = manifest_text();
  let features = feature_names(&manifest);
  let derived = derived_gates(ARTIFACTS);
  let failures = commercial_features_gating_nothing_restricted(
    ARTIFACTS,
    &derived,
    &features,
    &cfg_features_in_source(),
  );
  assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Every row's claimed `gate` is the feature the TREE puts on its loader, and
/// that feature is one the manifest declares.
///
/// The row's claim is kept because it makes the table readable; this is what
/// stops it from being believed. A row may claim `speaker` while the module
/// that loads it is gated on something else, or on nothing, or on a feature no
/// `[features]` entry declares — and directions 2 and 3 would then be reasoning
/// about a gate that does not exist.
#[test]
fn every_rows_gate_matches_the_cfg_that_guards_its_loader() {
  let declared = feature_names(&manifest_text());
  for row in ARTIFACTS {
    let gates = loader_gates(row.loader);
    assert_eq!(
      gates,
      BTreeSet::from([row.gate.to_string()]),
      "{}: the row claims gate {:?}, but {} puts {:?} on the module that loads it. The claim is \
       not the fact — only the `#[cfg]` decides whether the shipping path can load these bytes.",
      row.file,
      row.gate,
      row.loader,
      gates
    );
    assert!(
      declared.contains(row.gate),
      "{}: gate {:?} is not declared in this crate's `[features]` ({declared:?}). A gate nobody \
       can enable is not an opt-in, and a gate nobody can DISABLE is not a gate.",
      row.file,
      row.gate
    );
  }
}

/// **Direction 2's staging channel.** Every row's loader module is the module
/// named by its own `MODELS_LOCK` table's `kit`.
///
/// The third independent leg, and the derivation of the middle word in
/// `wired`. `Artifact::loader` is still written down by hand, so on its own it
/// could point at any module in the tree; tying it to the kit the LOCK
/// declares means the row cannot borrow an unrelated module's `#[cfg]`. Lock,
/// tree and manifest then have to agree before a gate is believed — see
/// [`rows_whose_loader_is_not_their_kit`] for the chain the failure text now
/// spells out.
///
/// It reds on today's lock with no doctoring: retag the arcface table's `kit`
/// to `face` or to `identity` and this is the assertion that goes off. Both
/// mutations are pinned hermetically as
/// `falsifiers::direction_two_reds_when_a_lock_rows_kit_*`.
#[test]
fn every_rows_loader_module_is_the_kit_its_lock_table_names() {
  let Some(tables) = lock_tables() else {
    return;
  };
  let failures = rows_whose_loader_is_not_their_kit(ARTIFACTS, &kits_of(&tables));
  assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Every `commercial-` feature's first documented sentence says a commercial
/// licence is required — the correction for a prefix that can be read
/// backwards.
#[test]
fn every_commercial_feature_says_it_requires_a_commercial_licence_first() {
  let Some(manifest) = repository_manifest_text() else {
    return;
  };
  assert!(
    features_block_of(&manifest).contains('#'),
    "the `[features]` block read for the doc rule carries no comments at all, so the rule would \
     pass vacuously. That is the stripped manifest `cargo package` writes, not the checked-in one."
  );
  let features = feature_names(&manifest);
  let docs = feature_docs(&manifest);
  let failures = commercial_features_without_the_phrase(&features, &docs);
  assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// No `commercial-` feature is reachable from `default`.
///
/// **`reachable` is the right word HERE and stays**, where direction 2's two
/// clauses now say `wired`: the subject of this one is a FEATURE and the
/// relation is cargo's own feature-graph closure, which is literally
/// reachability. Direction 2's subject is an ARTIFACT, and what it can see
/// about an artifact is only what this crate wires.
///
/// Stronger than direction 2's clauses and independent of any row: even before
/// a research-only artifact exists, a gate that `default` turns on is not a
/// gate. Today `default = []`, so the closure is `{"default"}` and this holds
/// trivially; it stops holding the moment somebody adds one.
#[test]
fn no_commercial_feature_is_reachable_from_default() {
  let manifest = manifest_text();
  let closure = feature_closure(&manifest, "default");
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
        assert!(
          !row.file.ends_with(".mlmodelc"),
          "{}: a SHA-256 keys ONE file, and this row names a bundle directory. A bundle's \
           identity is its whole manifest, not any one member's hash.",
          row.file
        );
      }
      Key::Unpinned(reason) | Key::Unmanifested(reason) => {
        assert!(
          !reason.trim().is_empty(),
          "{}: an exempt row with no reason is an exemption nobody can retire",
          row.file
        );
        assert!(
          row.pin.is_empty(),
          "{}: an exempt row names the pin {:?}. If those bytes are pinned, key on them.",
          row.file,
          row.pin
        );
      }
    }
    if let Key::Unmanifested(_) = row.key {
      assert!(
        row.file.ends_with(".mlmodelc"),
        "{}: `Key::Unmanifested` says this repository holds no per-file manifest for the bundle, \
         so the row must NAME the bundle. Naming one file inside it claims a file identity the \
         row has no way to check.",
        row.file
      );
    }
  }
}

/// A row may be [`Key::Unpinned`] only while its table is on
/// `revision = "main"`, [`Key::Unmanifested`] only on a table that GLOBS, and
/// every other row on a commit-pinned table must be hashed.
///
/// The staleness half, in the `CHECKSUMLESS_KITS` style: each exemption is tied
/// to its cause in both directions, so `MODELS_LOCK`'s LOUD FOLLOW-UP landing —
/// whisper's two tables moving from `main` to an immutable commit — turns this
/// red and forces the hashes in, instead of leaving rows describing bytes
/// nobody can identify. And a table that stops globbing names every file it
/// stages, at which point an unmanifested bundle row has to be replaced by
/// per-file rows.
#[test]
fn unpinned_rows_exist_only_where_the_lock_pins_a_moving_revision() {
  let Some(tables) = lock_tables() else {
    return;
  };
  let staged = staged_tables(&tables);
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
  let mut unmanifested = 0usize;
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
      Key::Unmanifested(_) => {
        unmanifested += 1;
        assert_ne!(
          revision, "main",
          "{}: `Key::Unmanifested` says the lock pins the bytes and only the MANIFEST is missing, \
           but MODELS_LOCK's {:?} is on `revision = \"main\"`, so the bytes are not pinned either. \
           That is `Key::Unpinned`.",
          row.file, row.staged_by
        );
        let table = staged
          .iter()
          .find(|t| t.name == row.staged_by)
          .expect("revision lookup above succeeded");
        assert!(
          matches!(table.selection, Selection::Include(_)),
          "{}: exempt from hashing because no per-file manifest exists, but MODELS_LOCK's {:?} \
           names its files explicitly — so the file list IS enumerable and this row must be \
           replaced by per-file rows.",
          row.file,
          row.staged_by
        );
      }
    }
  }
  assert!(
    moving > 0,
    "no row sits on a `revision = \"main\"` table, so this check no longer sees the case it \
     exists for. Delete it, or the exemption it guards."
  );
  assert!(
    unmanifested > 0,
    "no row is `Key::Unmanifested` any more, so the glob-table half of this check sees nothing. \
     Delete it, or the exemption it guards."
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

/// A pin locator belongs to the kit it is read for, and no two bundles share
/// one.
///
/// [`bundle_relative`] looks a row's hash up by the path AFTER the last
/// `.mlmodelc/`, and every CoreML bundle in the tree has a `weights/weight.bin`
/// — so a row could name ANY per-file manifest in the repository and match on
/// name alone. That is the repo-keyed mistake in miniature: the reader would
/// verify a hash that belongs to different bytes. Two ties close it — the pin
/// must live under a path component equal to the row's kit, and two rows in
/// different bundles may not share a locator.
#[test]
fn every_pin_locator_belongs_to_the_kit_and_bundle_it_is_read_for() {
  let Some(tables) = lock_tables() else {
    return;
  };
  let kits: BTreeMap<&str, &str> = tables
    .iter()
    .map(|t| {
      (
        t.name.as_str(),
        t.fields.get("kit").map_or("", String::as_str),
      )
    })
    .collect();
  let mut owner: BTreeMap<&str, &str> = BTreeMap::new();
  let mut checked = 0usize;
  for row in ARTIFACTS {
    if row.pin.is_empty() {
      continue;
    }
    let kit = kits.get(row.staged_by).copied().unwrap_or_else(|| {
      panic!(
        "{}: staged_by {:?} names no MODELS_LOCK table",
        row.file, row.staged_by
      )
    });
    let (source, _) = row.pin.split_once("::").expect("checked by pins_at");
    assert!(
      source.split('/').any(|component| component == kit),
      "{}: its pin {:?} lives outside the {kit:?} kit's sources. A pin is matched by \
       bundle-relative NAME, so a locator from another kit would verify a hash belonging to other \
       bytes and read clean doing it.",
      row.file,
      row.pin
    );
    let scope = row.bundle().unwrap_or(row.file);
    if let Some(previous) = owner.insert(row.pin, scope) {
      assert_eq!(
        previous, scope,
        "{}: pin {:?} is already the pin for {previous:?}. Two different bundles reading one \
         manifest means at least one of them is verifying a hash cut against other bytes.",
        row.file, row.pin
      );
    }
    checked += 1;
  }
  assert!(
    checked >= 10,
    "only {checked} pin locators checked; the table has shrunk and this would pass vacuously"
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
///
/// Compares the EFFECTIVE terms — class, canonical identifier and obligation
/// set — not the four-way class alone. Class-only comparison passed identical
/// bytes called MIT by one row and Apache-2.0 by another (both `permissive`),
/// and two research-only rows forbidding materially different things.
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
      for (layer, a, b) in [
        ("weights", first.weights, other.weights),
        ("corpus", first.corpus, other.corpus),
      ] {
        if a.effective() == b.effective() {
          continue;
        }
        failures.push(format!(
          "sha256 {hex}: {} records the {layer} layer as {} / {:?} / {:?} and {} records it as {} \
           / {:?} / {:?}. Identical bytes cannot carry different terms — one row is repeating a \
           repository tag rather than the licence of the artifact it re-hosts.",
          first.file,
          a.verdict(),
          a.licence(),
          a.restrictions(),
          other.file,
          b.verdict(),
          b.licence(),
          b.restrictions(),
        ));
      }
    }
  }
  failures
}

/// Every verdict carries prose, every unresolved one names what is open, and
/// every resolved one names the identifier it resolved to.
///
/// An empty payload turns the table back into a bare SPDX list, which is the
/// thing it was built not to be — and an `Unresolved` with nothing to follow
/// is indistinguishable from nobody having looked. The identifier rule is the
/// other half: a resolved layer with no identifier cannot be compared against
/// a second row over the same bytes, and an unresolved layer WITH one is
/// claiming an answer it just said it did not have.
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
      if matches!(terms, Terms::Unresolved(_)) {
        assert!(
          terms.licence().is_empty() && terms.restrictions().is_empty(),
          "{}: the {layer} layer is unresolved but records the identifier {:?} and the \
           restrictions {:?}. Nothing is established, so nothing may be recorded as established.",
          row.file,
          terms.licence(),
          terms.restrictions()
        );
      } else {
        assert!(
          !terms.licence().trim().is_empty(),
          "{}: the {layer} layer is {:?} but names no canonical identifier, so a second row over \
           the same bytes has nothing to be compared against.",
          row.file,
          terms.verdict()
        );
        assert!(
          !terms.restrictions().is_empty(),
          "{}: the {layer} layer is {:?} but records no obligation at all. Even the most \
           permissive licence here requires the notice to be retained.",
          row.file,
          terms.verdict()
        );
      }
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

/// The state of the table, asserted rather than remembered — as an EXACT
/// census per layer.
///
/// The previous form of this asserted that EVERY corpus row was unresolved,
/// which froze the table's weakest state into a passing test: resolving a
/// corpus layer against its own upstream would have turned this red, so the
/// check was an incentive to leave rows unresolved. A census records what is
/// there instead, and still refuses a silent change in either direction.
#[test]
fn the_tables_verdict_census_is_what_this_file_says_it_is() {
  let restricted: Vec<&str> = ARTIFACTS
    .iter()
    .filter(|r| r.disqualifying_layer().is_some())
    .map(|r| r.file)
    .collect();
  assert_eq!(
    restricted,
    ["facekit/w600k_r50.mlmodelc/weights/weight.bin"],
    "the set of DISQUALIFYING rows moved. This is the census's sharpest cell: a row entering it \
     puts directions 2 and 3 in charge of an artifact that must now be behind a \
     `{COMMERCIAL_PREFIX}` feature, and a row leaving it retires a gate that would otherwise \
     stand as false reassurance. Either way, say what moved and why in this file's module doc \
     before re-baselining it."
  );
  let census = |pick: fn(&Artifact) -> Terms| {
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    for row in ARTIFACTS {
      *counts.entry(pick(row).verdict()).or_default() += 1;
    }
    counts
  };
  assert_eq!(
    census(|r| r.weights),
    BTreeMap::from([
      ("attribution-required", 12),
      ("permissive", 15),
      ("research-only", 1),
      ("unresolved", 1)
    ]),
    "the weights-layer census changed. Say what moved and why in this file's module doc before \
     re-baselining it — a licence verdict that changes silently is the failure this table exists \
     to prevent."
  );
  assert_eq!(
    census(|r| r.corpus),
    BTreeMap::from([
      ("attribution-required", 7),
      ("permissive", 2),
      ("research-only", 1),
      ("unresolved", 19)
    ]),
    "the corpus-layer census changed. Say what moved and why in this file's module doc before \
     re-baselining it."
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
    Artifact, ArtifactNames, CREDIT_AUTHOR, Covered, Key, NOTHING_ESTABLISHED, RETAIN_NOTICE,
    Selection, StagedTable, Terms, cfg_features_in, collect_field_literals,
    commercial_features_gating_nothing_restricted, commercial_features_without_the_phrase,
    compiled_targets, contradictory_terms, feature_closure, feature_closures, feature_docs,
    feature_entries, feature_names, first_sentence, fp16_pinned_bundles_without_a_row,
    gates_of_module, glob_matches, parse_lock, research_only_tested, research_only_wired,
    rows_whose_loader_is_not_their_kit, ungranted_tested_under_default,
    ungranted_wired_into_default, unmatched_coverage,
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
      loader: "a falsifier row, never resolved against the tree",
      gate,
      weights,
      corpus,
      source: "a falsifier, not a real record",
    }
  }

  const CLEAR: Terms = Terms::permissive("MIT", RETAIN_NOTICE, "clear, for a falsifier");
  const RESTRICTED: Terms = Terms::research_only(
    "LicenseRef-research-only",
    RETAIN_NOTICE,
    "non-commercial research purposes only, for a falsifier",
  );
  const ATTRIBUTED: Terms = Terms::attribution(
    "CC-BY-4.0",
    CREDIT_AUTHOR,
    "granted, on condition of attribution, for a falsifier",
  );
  const UNGRANTED: Terms = Terms::unresolved("nothing established, for a falsifier");

  /// The manifest shape that exposed the hole: an ordinary kit feature, and
  /// `default` turning it on.
  const SHIPS_IT_BY_DEFAULT: &str = "\
[features]
default = [\"identity\"]
identity = [\"dep:rustfft\"]
";

  /// The same manifest with the feature left as an opt-in — this crate's
  /// actual shape.
  const OPT_IN_ONLY: &str = "\
[features]
default = []
identity = [\"dep:rustfft\"]
";

  /// The row `redimnet/redimnet_b5.mlmodelc` is: no grant over the WEIGHTS,
  /// an attribution grant over the corpus, behind a plain kit feature.
  const REDIMNET_SHAPED: &str = "redimnet/redimnet_b5.mlmodelc/weights/weight.bin";

  /// A roster of `Models/`-relative bundle paths, as a reader would return it.
  fn staged_paths(paths: &[&str]) -> Vec<String> {
    paths.iter().map(|p| (*p).to_string()).collect()
  }

  fn features(names: &[&str]) -> BTreeSet<String> {
    names.iter().map(|n| (*n).to_string()).collect()
  }

  /// One `include`-globbed table, the shape every model bundle rides.
  fn globbed(name: &str, vendor_dir: &str, patterns: &[&str]) -> StagedTable {
    StagedTable {
      name: name.to_string(),
      vendor_dir: vendor_dir.to_string(),
      selection: Selection::Include(patterns.iter().map(|p| (*p).to_string()).collect()),
    }
  }

  /// One `files`-listed table — the shape `openai/whisper-tiny` has, and the
  /// one the repository-keyed coverage check could not see into.
  fn listed(name: &str, vendor_dir: &str, files: &[&str]) -> StagedTable {
    StagedTable {
      name: name.to_string(),
      vendor_dir: vendor_dir.to_string(),
      selection: Selection::Files(files.iter().map(|f| (*f).to_string()).collect()),
    }
  }

  /// A row covering exactly the paths given, table-relative.
  fn covering<'a>(file: &'a str, staged_by: &'a str, covers: &[&str]) -> Covered<'a> {
    Covered {
      file,
      staged_by,
      covered: covers.iter().map(|c| (*c).to_string()).collect(),
    }
  }

  // --- direction 1 ---------------------------------------------------------

  #[test]
  fn direction_one_passes_when_every_table_and_row_line_up() {
    let tables = [globbed("vendor/one", "one", &["*.mlmodelc/*"])];
    let rows = [covering(
      "one/a.mlmodelc/weights/weight.bin",
      "vendor/one",
      &["a.mlmodelc/weights/weight.bin"],
    )];
    assert!(unmatched_coverage(&tables, &rows).is_empty());
  }

  #[test]
  fn direction_one_reds_when_a_staged_repo_has_no_row() {
    let tables = [
      globbed("vendor/one", "one", &["*.mlmodelc/*"]),
      globbed("vendor/two", "two", &["*.mlmodelc/*"]),
    ];
    let rows = [covering(
      "one/a.mlmodelc/weights/weight.bin",
      "vendor/one",
      &["a.mlmodelc/weights/weight.bin"],
    )];
    let failures = unmatched_coverage(&tables, &rows);
    assert_eq!(failures.len(), 1, "{failures:?}");
    assert!(failures[0].contains("vendor/two"), "{failures:?}");
  }

  #[test]
  fn direction_one_reds_when_a_row_names_no_staged_repo() {
    let tables = [globbed("vendor/one", "one", &["*.mlmodelc/*"])];
    let rows = [
      covering(
        "one/a.mlmodelc/weights/weight.bin",
        "vendor/one",
        &["a.mlmodelc/weights/weight.bin"],
      ),
      covering("gone/b.bin", "vendor/gone", &["b.bin"]),
    ];
    let failures = unmatched_coverage(&tables, &rows);
    assert_eq!(failures.len(), 1, "{failures:?}");
    assert!(failures[0].contains("vendor/gone"), "{failures:?}");
  }

  /// **The finding.** `openai/whisper-tiny`'s exact shape: a table that names
  /// three files, and a table of licence rows that carries one. A check that
  /// compares repository NAMES sees a covered repository and passes.
  #[test]
  fn direction_one_reds_when_a_files_table_stages_a_file_no_row_covers() {
    let tables = [listed(
      "openai/whisper-tiny",
      "tokenizers/whisper-tiny",
      &["tokenizer.json", "tokenizer_config.json", "config.json"],
    )];
    let rows = [covering(
      "tokenizers/whisper-tiny/tokenizer.json",
      "openai/whisper-tiny",
      &["tokenizer.json"],
    )];
    let failures = unmatched_coverage(&tables, &rows);
    assert_eq!(failures.len(), 2, "{failures:?}");
    assert!(
      failures.iter().any(|f| f.contains("tokenizer_config.json")),
      "{failures:?}"
    );
    assert!(
      failures.iter().any(|f| f.contains("config.json")),
      "{failures:?}"
    );
  }

  /// **CHARACTERIZATION, not a check. This records a defect, and it is
  /// INVERTED when coremlit #139 lands.**
  ///
  /// Direction 1's forward loop — "every file the table names must be covered
  /// by some row" — runs only under `if let Selection::Files(listed)`. An
  /// `include` glob's file list exists only after a download, so a table that
  /// stages by glob is reconciled ONLY against the rows that already exist:
  /// **a bundle the glob stages with no `Artifact` row is never discovered.**
  /// Nine of ten `MODELS_LOCK` tables are `include` globs, so this is the
  /// common shape rather than the exception.
  ///
  /// The fixture below is that situation exactly. `vendor/one` stages
  /// `*.mlmodelc/*`; one row covers `a.mlmodelc`, and the same glob also stages
  /// a `b.mlmodelc` that no row covers. Because the lock names no file list for
  /// this table, `b.mlmodelc` cannot be written into the fixture at all — which
  /// IS the defect: it is invisible to the checker, so the fixture cannot state
  /// it either, and the check comes back clean.
  ///
  /// Whose defect: **verified pre-existing on `main` `bf8b3f6`** at
  /// `model_licences.rs:1336` — the same `if let Selection::Files(listed)`,
  /// rows identical main→branch apart from the one new redimnet row. The
  /// branch that added this test neither widened nor narrowed it, so per the
  /// standing rule it is recorded here and fixed in its own issue rather than
  /// absorbed.
  ///
  /// The fix, and what to do with this test: coremlit **#139** commits a
  /// per-table manifest (`MODELS_LOCK.d/<vendor_dir>@<revision>.sha256`,
  /// upstream's `CHECKSUMS.sha256` verbatim where one ships) and derives every
  /// table's staged set from it, running the forward loop for ALL tables. When
  /// that lands, this assertion becomes `!is_empty()` and the doc above becomes
  /// the falsifier it describes. Until then it is here so the gap is a runnable
  /// statement rather than a sentence in an issue.
  #[test]
  fn direction_one_does_not_enumerate_a_glob_tables_contents_forward() {
    let tables = [globbed("vendor/one", "one", &["*.mlmodelc/*"])];
    let rows = [covering(
      "one/a.mlmodelc/weights/weight.bin",
      "vendor/one",
      &["a.mlmodelc/weights/weight.bin"],
    )];

    let failures = unmatched_coverage(&tables, &rows);
    assert!(
      failures.is_empty(),
      "CHARACTERIZATION: this test records coremlit #139 (direction 1 does not enumerate an \
       `include` table's contents forward), and asserting a clean result is how it records it. \
       A non-empty result here means #139 landed — INVERT this assertion to `!is_empty()` and \
       rewrite the doc comment as the falsifier it already describes. Got: {failures:?}"
    );
  }

  #[test]
  fn direction_one_passes_when_every_named_file_has_a_row() {
    let tables = [listed(
      "openai/whisper-tiny",
      "tokenizers/whisper-tiny",
      &["tokenizer.json", "tokenizer_config.json", "config.json"],
    )];
    let rows = [
      covering(
        "tokenizers/whisper-tiny/tokenizer.json",
        "openai/whisper-tiny",
        &["tokenizer.json"],
      ),
      covering(
        "tokenizers/whisper-tiny/tokenizer_config.json",
        "openai/whisper-tiny",
        &["tokenizer_config.json"],
      ),
      covering(
        "tokenizers/whisper-tiny/config.json",
        "openai/whisper-tiny",
        &["config.json"],
      ),
    ];
    assert!(
      unmatched_coverage(&tables, &rows).is_empty(),
      "{:?}",
      unmatched_coverage(&tables, &rows)
    );
  }

  /// A row inside a bundle covers the WHOLE bundle only because its pin is a
  /// per-file manifest — so a bundle row satisfies a `files` entry under it.
  #[test]
  fn a_bundle_row_covers_every_file_beneath_it() {
    let tables = [listed(
      "vendor/one",
      "one",
      &["a.mlmodelc/model.mil", "a.mlmodelc/weights/weight.bin"],
    )];
    let rows = [covering("one/a.mlmodelc", "vendor/one", &["a.mlmodelc"])];
    assert!(
      unmatched_coverage(&tables, &rows).is_empty(),
      "{:?}",
      unmatched_coverage(&tables, &rows)
    );
  }

  #[test]
  fn direction_one_reds_when_a_row_names_a_file_its_table_does_not_stage() {
    let tables = [listed(
      "openai/whisper-tiny",
      "tokenizers/whisper-tiny",
      &["tokenizer.json"],
    )];
    let rows = [
      covering(
        "tokenizers/whisper-tiny/tokenizer.json",
        "openai/whisper-tiny",
        &["tokenizer.json"],
      ),
      covering(
        "tokenizers/whisper-tiny/vocab.txt",
        "openai/whisper-tiny",
        &["vocab.txt"],
      ),
    ];
    let failures = unmatched_coverage(&tables, &rows);
    assert_eq!(failures.len(), 1, "{failures:?}");
    assert!(failures[0].contains("vocab.txt"), "{failures:?}");
  }

  #[test]
  fn direction_one_reds_when_a_glob_table_does_not_select_the_rows_path() {
    let tables = [globbed(
      "vendor/one",
      "one",
      &["pyannote_segmentation.mlmodelc/*"],
    )];
    let rows = [covering(
      "one/wespeaker.mlmodelc/weights/weight.bin",
      "vendor/one",
      &["wespeaker.mlmodelc/weights/weight.bin"],
    )];
    let failures = unmatched_coverage(&tables, &rows);
    assert_eq!(failures.len(), 1, "{failures:?}");
    assert!(failures[0].contains("does not stage"), "{failures:?}");
  }

  #[test]
  fn direction_one_reds_when_a_row_sits_outside_its_tables_vendor_directory() {
    let tables = [globbed("vendor/one", "one", &["*.mlmodelc/*"])];
    let rows = [covering(
      "elsewhere/a.mlmodelc/weights/weight.bin",
      "vendor/one",
      &["a.mlmodelc/weights/weight.bin"],
    )];
    let failures = unmatched_coverage(&tables, &rows);
    assert_eq!(failures.len(), 1, "{failures:?}");
    assert!(failures[0].contains("must start with"), "{failures:?}");
  }

  /// The selector semantics `MODELS_LOCK` inherits from `huggingface_hub`:
  /// `*` crosses `/`, and a `dir/*` pattern also stands for the directory when
  /// a row names the bundle rather than a file in it.
  #[test]
  fn the_glob_matcher_matches_what_hugging_face_would_select() {
    assert!(glob_matches(
      "*.mlmodelc/*",
      "wespeaker_v2.mlmodelc/weights/weight.bin"
    ));
    assert!(glob_matches(
      "openai_whisper-tiny/*",
      "openai_whisper-tiny/AudioEncoder.mlmodelc/weights/weight.bin"
    ));
    assert!(glob_matches("CHECKSUMS.sha256", "CHECKSUMS.sha256"));
    assert!(!glob_matches("*.mlmodelc/*", "CHECKSUMS.sha256"));
    assert!(!glob_matches("*.mlmodelc/*", "wespeaker_v2.mlmodelc"));
    assert!(!glob_matches(
      "pyannote_segmentation.mlmodelc/*",
      "wespeaker.mlmodelc/weights/weight.bin"
    ));
  }

  #[test]
  fn a_bundle_row_is_selected_by_the_glob_that_stages_its_files() {
    let table = globbed("vendor/one", "one", &["*.mlmodelc/*"]);
    assert!(table.selects("PLDA.mlmodelc"));
    assert!(table.selects("PLDA.mlmodelc/weights/weight.bin"));
    assert!(!table.selects("CHECKSUMS.sha256"));
  }

  /// A manifest-derived row covers its bundle AND every file in it; the
  /// per-file entries alone do not answer "is this bundle covered", which is
  /// the question the sweep roster asks. This pins the contract
  /// [`super::row_coverage`] builds to, because a hand-written coverage set
  /// that only ever names bundles would agree with a broken builder.
  #[test]
  fn a_manifest_derived_row_covers_its_bundle_and_every_file_in_it() {
    let manifest_derived = covering(
      "speakerkit/wespeaker_v2.mlmodelc/weights/weight.bin",
      "vendor/one",
      &[
        "wespeaker_v2.mlmodelc",
        "wespeaker_v2.mlmodelc/model.mil",
        "wespeaker_v2.mlmodelc/weights/weight.bin",
      ],
    );
    assert!(manifest_derived.covers("wespeaker_v2.mlmodelc"));
    assert!(manifest_derived.covers("wespeaker_v2.mlmodelc/metadata.json"));

    let one_file_only = covering(
      "speakerkit/PLDA.mlmodelc/model.mil",
      "vendor/one",
      &["PLDA.mlmodelc/model.mil"],
    );
    assert!(one_file_only.covers("PLDA.mlmodelc/model.mil"));
    assert!(
      !one_file_only.covers("PLDA.mlmodelc"),
      "a row over one file inside a bundle is not a row over the bundle"
    );
    assert!(!one_file_only.covers("PLDA.mlmodelc/weights/weight.bin"));
  }

  /// **The second enumeration.** A bundle the fp16 sweep pins under a staged
  /// vendor directory, with no licence row over it — the shape that stays
  /// invisible to a glob-table coverage check, because the lock cannot list
  /// what a glob brings in.
  #[test]
  fn a_swept_bundle_with_no_licence_row_reds() {
    let tables = [globbed("vendor/one", "speakerkit", &["*.mlmodelc/*"])];
    let rows = [covering(
      "speakerkit/wespeaker.mlmodelc",
      "vendor/one",
      &["wespeaker.mlmodelc"],
    )];
    let pinned = staged_paths(&["speakerkit/wespeaker.mlmodelc", "speakerkit/PLDA.mlmodelc"]);
    let failures = fp16_pinned_bundles_without_a_row(&pinned, &tables, &rows);
    assert_eq!(failures.len(), 1, "{failures:?}");
    assert!(failures[0].contains("PLDA.mlmodelc"), "{failures:?}");
  }

  #[test]
  fn a_swept_bundle_with_a_row_passes_and_an_unstaged_vendor_is_left_alone() {
    let tables = [globbed("vendor/one", "speakerkit", &["*.mlmodelc/*"])];
    let rows = [covering(
      "speakerkit/wespeaker.mlmodelc",
      "vendor/one",
      &["wespeaker.mlmodelc"],
    )];
    let pinned = staged_paths(&[
      "speakerkit/wespeaker.mlmodelc",
      // No MODELS_LOCK table stages `alignkit/`; ci.yml's own
      // `UNSTAGED_DEFECT_VENDORS` records that gap, so this file must not
      // claim it as a licence failure.
      "alignkit/base960h_aligner.mlmodelc",
    ]);
    assert!(
      fp16_pinned_bundles_without_a_row(&pinned, &tables, &rows).is_empty(),
      "{:?}",
      fp16_pinned_bundles_without_a_row(&pinned, &tables, &rows)
    );
  }

  /// A row keyed on ONE file inside a bundle covers the bundle only through its
  /// per-file manifest — so a swept bundle whose row covers just `model.mil`
  /// is not covered.
  #[test]
  fn a_row_covering_one_file_does_not_cover_the_bundle_the_sweep_pins() {
    let tables = [globbed("vendor/one", "speakerkit", &["*.mlmodelc/*"])];
    let rows = [covering(
      "speakerkit/PLDA.mlmodelc/model.mil",
      "vendor/one",
      &["PLDA.mlmodelc/model.mil"],
    )];
    let pinned = staged_paths(&["speakerkit/PLDA.mlmodelc"]);
    let failures = fp16_pinned_bundles_without_a_row(&pinned, &tables, &rows);
    assert_eq!(failures.len(), 1, "{failures:?}");
  }

  // --- direction 2 ---------------------------------------------------------

  /// The gates directions 2 and 3 run on, as the tree would report them.
  fn tree_gates(pairs: &[(&'static str, &[&str])]) -> BTreeMap<&'static str, BTreeSet<String>> {
    pairs
      .iter()
      .map(|(file, gates)| (*file, gates.iter().map(|g| (*g).to_string()).collect()))
      .collect()
  }

  /// A manifest whose `default` is empty and whose kit features are ordinary —
  /// the shape this crate has today.
  const CLEAN_FEATURES: &str = "\
[features]
default = []
speaker = [\"dep:diaric\"]
# Requires a commercial licence from the weights' author.
commercial-face = [\"dep:facelib\"]
";

  /// **The finding.** `default = []` while an ORDINARY kit feature pulls the
  /// commercial gate in. Consulting only `default`'s closure sees nothing, and
  /// the row's claimed gate carries the prefix, so a claim-driven check passes
  /// while `cargo add coremlit --features speaker` ships the restricted bytes.
  const WIRED_VIA_A_PLAIN_FEATURE: &str = "\
[features]
default = []
speaker = [\"dep:diaric\", \"commercial-face\"]
# Requires a commercial licence from the weights' author.
commercial-face = [\"dep:facelib\"]
";

  #[test]
  fn direction_two_passes_when_a_restricted_row_sits_behind_an_opt_in_commercial_gate() {
    let rows = [row(
      "a/w.bin",
      "vendor/one",
      "commercial-face",
      CLEAR,
      RESTRICTED,
    )];
    let derived = tree_gates(&[("a/w.bin", &["commercial-face"])]);
    let closures = feature_closures(CLEAN_FEATURES);
    assert!(
      research_only_wired(&rows, &derived, &closures).is_empty(),
      "{:?}",
      research_only_wired(&rows, &derived, &closures)
    );
  }

  #[test]
  fn direction_two_reds_when_a_plain_feature_closure_reaches_the_commercial_gate() {
    let rows = [row(
      "a/w.bin",
      "vendor/one",
      "commercial-face",
      CLEAR,
      RESTRICTED,
    )];
    let derived = tree_gates(&[("a/w.bin", &["commercial-face"])]);
    let closures = feature_closures(WIRED_VIA_A_PLAIN_FEATURE);
    let failures = research_only_wired(&rows, &derived, &closures);
    assert_eq!(failures.len(), 1, "{failures:?}");
    assert!(failures[0].contains("\"speaker\""), "{failures:?}");
    assert!(failures[0].contains("training corpus"), "{failures:?}");
  }

  #[test]
  fn direction_two_reds_when_a_research_only_row_is_wired_into_default() {
    const LEAKY: &str = "\
[features]
default = [\"commercial-face\"]
# Requires a commercial licence from the weights' author.
commercial-face = [\"dep:facelib\"]
";
    let rows = [row(
      "a/w.bin",
      "vendor/one",
      "commercial-face",
      CLEAR,
      RESTRICTED,
    )];
    let derived = tree_gates(&[("a/w.bin", &["commercial-face"])]);
    let failures = research_only_wired(&rows, &derived, &feature_closures(LEAKY));
    assert_eq!(failures.len(), 1, "{failures:?}");
    assert!(
      failures[0].contains("plain `cargo add coremlit`"),
      "{failures:?}"
    );
  }

  /// The tree, not the row, decides. A row may CLAIM a commercial gate while
  /// the module that loads it is gated on an ordinary kit feature.
  #[test]
  fn direction_two_reds_when_the_tree_gates_the_loader_on_a_plain_feature() {
    let rows = [row(
      "a/w.bin",
      "vendor/one",
      "commercial-face",
      RESTRICTED,
      CLEAR,
    )];
    let derived = tree_gates(&[("a/w.bin", &["speaker"])]);
    let failures = research_only_wired(&rows, &derived, &feature_closures(CLEAN_FEATURES));
    assert!(
      failures.iter().any(|f| f.contains("does not carry")),
      "{failures:?}"
    );
    assert!(
      failures.iter().any(|f| f.contains("weights layer")),
      "{failures:?}"
    );
  }

  /// And a row whose loader carries no `#[cfg]` at all is behind nothing,
  /// however confident its `gate` field is.
  #[test]
  fn direction_two_reds_when_the_tree_gates_the_loader_on_nothing() {
    let rows = [row(
      "a/w.bin",
      "vendor/one",
      "commercial-face",
      CLEAR,
      RESTRICTED,
    )];
    let derived = tree_gates(&[("a/w.bin", &[])]);
    let failures = research_only_wired(&rows, &derived, &feature_closures(CLEAN_FEATURES));
    assert_eq!(failures.len(), 1, "{failures:?}");
    assert!(
      failures[0].contains("compiles unconditionally"),
      "{failures:?}"
    );
  }

  #[test]
  fn direction_two_names_the_corpus_layer_when_that_is_what_disqualifies() {
    let rows = [row("a/w.bin", "vendor/one", "speaker", CLEAR, RESTRICTED)];
    let derived = tree_gates(&[("a/w.bin", &["speaker"])]);
    let failures = research_only_wired(&rows, &derived, &feature_closures(CLEAN_FEATURES));
    assert!(
      failures.iter().any(|f| f.contains("training corpus layer")),
      "{failures:?}"
    );
  }

  /// **THE HOLE, as an assertion.** `default = ["identity"]`, so a plain
  /// `cargo add coremlit` compiles the loader for an artifact whose WEIGHTS
  /// layer has no grant at all.
  ///
  /// Handed to the strong clause this same input returned `[]`, and all 64
  /// checks the file then held stayed green: only `Terms::ResearchOnly` set
  /// `forbids_commercial_use`, so `disqualifying_layer()` was `None` and the
  /// row was skipped before anything looked at the feature graph.
  #[test]
  fn direction_two_reds_when_an_unresolved_row_is_wired_into_default() {
    let rows = [row(
      REDIMNET_SHAPED,
      "vendor/one",
      "identity",
      UNGRANTED,
      ATTRIBUTED,
    )];
    let derived = tree_gates(&[(REDIMNET_SHAPED, &["identity"])]);
    let failures = ungranted_wired_into_default(
      &rows,
      &derived,
      &feature_closure(SHIPS_IT_BY_DEFAULT, "default"),
    );
    assert_eq!(failures.len(), 1, "{failures:?}");
    assert!(failures[0].contains("\"identity\""), "{failures:?}");
    assert!(
      failures[0].contains("plain `cargo add coremlit`"),
      "{failures:?}"
    );
    assert!(
      failures[0].contains("NOTHING is established"),
      "{failures:?}"
    );
    assert!(
      !failures[0].contains("forbid commercial use"),
      "an unresolved row must not be reported as a prohibition: {failures:?}"
    );
  }

  /// The same row with the feature left an opt-in — this crate's real shape.
  /// The wide clause asks only that the consumer had to ask for it.
  #[test]
  fn direction_two_leaves_an_unresolved_row_behind_an_opt_in_feature_alone() {
    let rows = [row(
      REDIMNET_SHAPED,
      "vendor/one",
      "identity",
      UNGRANTED,
      ATTRIBUTED,
    )];
    let derived = tree_gates(&[(REDIMNET_SHAPED, &["identity"])]);
    assert!(
      ungranted_wired_into_default(&rows, &derived, &feature_closure(OPT_IN_ONLY, "default"))
        .is_empty()
    );
  }

  /// The wide clause covers research-only rows too, and says something
  /// DIFFERENT about them — a found prohibition, not an open question.
  #[test]
  fn direction_two_names_a_prohibition_when_the_row_wired_into_default_is_research_only() {
    let rows = [row(
      REDIMNET_SHAPED,
      "vendor/one",
      "identity",
      RESTRICTED,
      CLEAR,
    )];
    let derived = tree_gates(&[(REDIMNET_SHAPED, &["identity"])]);
    let failures = ungranted_wired_into_default(
      &rows,
      &derived,
      &feature_closure(SHIPS_IT_BY_DEFAULT, "default"),
    );
    assert_eq!(failures.len(), 1, "{failures:?}");
    assert!(
      failures[0].contains("ESTABLISHED and they forbid commercial use"),
      "{failures:?}"
    );
    assert!(
      !failures[0].contains("NOTHING is established"),
      "{failures:?}"
    );
  }

  /// A loader with no `#[cfg]` at all is wired into `default` however empty
  /// `default` is — the clause that keeps the wide check non-vacuous against
  /// today's table.
  #[test]
  fn direction_two_reds_when_an_ungranted_loader_carries_no_cfg() {
    let rows = [row(
      REDIMNET_SHAPED,
      "vendor/one",
      "identity",
      UNGRANTED,
      CLEAR,
    )];
    let derived = tree_gates(&[(REDIMNET_SHAPED, &[])]);
    let failures =
      ungranted_wired_into_default(&rows, &derived, &feature_closure(OPT_IN_ONLY, "default"));
    assert_eq!(failures.len(), 1, "{failures:?}");
    assert!(
      failures[0].contains("wired into EVERY configuration"),
      "{failures:?}"
    );
  }

  /// And the widening stops where the grants start. A permissive or
  /// attribution row in `default` is what this crate ships on purpose; a
  /// clause that flagged those would red the whole table.
  #[test]
  fn direction_two_leaves_granted_rows_in_default_alone() {
    let rows = [
      row("a/w.bin", "vendor/one", "identity", CLEAR, CLEAR),
      row("b/w.bin", "vendor/one", "identity", ATTRIBUTED, ATTRIBUTED),
    ];
    let derived = tree_gates(&[("a/w.bin", &["identity"]), ("b/w.bin", &["identity"])]);
    assert!(
      ungranted_wired_into_default(
        &rows,
        &derived,
        &feature_closure(SHIPS_IT_BY_DEFAULT, "default")
      )
      .is_empty()
    );
  }

  /// The STRONG clause still does not sweep unresolved rows in, and that is a
  /// decision rather than the hole above.
  ///
  /// Requiring a `commercial-` gate here would demand a feature whose first
  /// documented sentence says a commercial licence is REQUIRED over a row that
  /// says nobody has established anything — and it would demand it of the
  /// nineteen rows in the real table with an unresolved CORPUS layer, which is
  /// most of this crate's public feature surface. See the module doc.
  #[test]
  fn direction_twos_prefix_clause_does_not_sweep_in_unresolved_rows() {
    let rows = [row("a/w.bin", "vendor/one", "speaker", UNGRANTED, CLEAR)];
    let derived = tree_gates(&[("a/w.bin", &["speaker"])]);
    assert!(research_only_wired(&rows, &derived, &feature_closures(CLEAN_FEATURES)).is_empty());
  }

  /// The two axes, pinned for every variant of the vocabulary — the
  /// per-variant audit made executable instead of written in a comment.
  ///
  /// `unresolved` is the cell the hole lived in: `(false, false)`. Not
  /// forbidden, because nobody found a prohibition; not permitted either,
  /// which is the half that had no predicate. A FIFTH variant is caught by the
  /// compiler rather than here — both axes are exhaustive `match`es — and this
  /// pins what the four existing answers are.
  #[test]
  fn the_two_shipping_axes_are_pinned_for_every_terms_variant() {
    let axes: Vec<(&str, bool, bool)> = [CLEAR, ATTRIBUTED, RESTRICTED, UNGRANTED]
      .into_iter()
      .map(|t| {
        (
          t.verdict(),
          t.forbids_commercial_use(),
          t.permits_a_shipping_claim(),
        )
      })
      .collect();
    assert_eq!(
      axes.as_slice(),
      [
        ("permissive", false, true),
        ("attribution-required", false, true),
        ("research-only", true, false),
        ("unresolved", false, false),
      ]
      .as_slice(),
      "the shipping vocabulary moved. `forbids_commercial_use` is what the \
       `commercial-` prefix hangs on and `permits_a_shipping_claim` is what \
       `default`-reachability hangs on; they are not each other's negation, and \
       the day they become one, one of the two directions stops seeing a class of row."
    );
  }

  // --- direction 2, the STAGING channel ------------------------------------

  /// The falsifier row with a loader locator that names a real-looking module
  /// — the only part of it the kit reconciliation reads.
  const fn row_loaded_by(loader: &'static str) -> Artifact {
    Artifact {
      loader,
      ..row(
        "a/w.bin",
        "vendor/one",
        "commercial-face",
        CLEAR,
        RESTRICTED,
      )
    }
  }

  /// **The lock mutation, pinned.** Retagging the staging table's `kit` from
  /// the module that manifests the artifact to the PLAIN kit feature's name is
  /// the exact shape `MODELS_LOCK`'s own comment warns against — and it is red
  /// here rather than merely commented against.
  ///
  /// Run against the real lock it reds too, on
  /// `every_rows_loader_module_is_the_kit_its_lock_table_names`; this is that
  /// run with no repository files and no models.
  #[test]
  fn direction_two_reds_when_a_lock_rows_kit_is_the_plain_kit_feature() {
    let rows = [row_loaded_by("src/embeddings/face/mod.rs::arcface")];
    let kits = BTreeMap::from([("vendor/one", "face")]);
    let failures = rows_whose_loader_is_not_their_kit(&rows, &kits);
    assert_eq!(failures.len(), 1, "{failures:?}");
    assert!(failures[0].contains("kit \"face\""), "{failures:?}");
    assert!(failures[0].contains("\"arcface\" module"), "{failures:?}");
    // The message has to state the CHAIN, because the reason a string
    // comparison is enough here is that the string is a link in one.
    assert!(failures[0].contains("research_only_wired"), "{failures:?}");
  }

  /// And retagging it to an UNRELATED kit is the same finding: the row would
  /// then read the `identity` module's `#[cfg]`, which is a plain feature.
  #[test]
  fn direction_two_reds_when_a_lock_rows_kit_names_another_kits_module() {
    let rows = [row_loaded_by("src/embeddings/face/mod.rs::arcface")];
    let kits = BTreeMap::from([("vendor/one", "identity")]);
    let failures = rows_whose_loader_is_not_their_kit(&rows, &kits);
    assert_eq!(failures.len(), 1, "{failures:?}");
    assert!(failures[0].contains("kit \"identity\""), "{failures:?}");
  }

  #[test]
  fn direction_two_accepts_a_lock_row_whose_kit_is_its_loader_module() {
    let rows = [row_loaded_by("src/embeddings/face/mod.rs::arcface")];
    let kits = BTreeMap::from([("vendor/one", "arcface")]);
    assert!(rows_whose_loader_is_not_their_kit(&rows, &kits).is_empty());
  }

  // --- direction 2, the TEST channel ---------------------------------------

  /// A synthetic package tree: manifest-relative path to source text.
  fn tree(files: &[(&str, &str)]) -> BTreeMap<String, Vec<u8>> {
    files
      .iter()
      .map(|(path, text)| ((*path).to_string(), (*text).as_bytes().to_vec()))
      .collect()
  }

  /// The same, for a fixture whose bytes are not text.
  fn tree_with_bytes(
    files: &[(&str, &str)],
    binary: &[(&str, &[u8])],
  ) -> BTreeMap<String, Vec<u8>> {
    let mut tree = tree(files);
    for (path, bytes) in binary {
      tree.insert((*path).to_string(), (*bytes).to_vec());
    }
    tree
  }

  /// The features every target falsifier runs against: one ordinary kit
  /// feature, and one commercial opt-in that implies it.
  const TARGET_FEATURES: &str = "\
[features]
default = []
face = []
# Requires a commercial licence from the weights' author.
commercial-face = [\"face\"]
";

  /// A manifest declaring one `[[test]]` target over [`TARGET_FEATURES`].
  fn one_target(required: &str) -> String {
    format!(
      "{TARGET_FEATURES}
[[test]]
name = \"kit_gate\"
path = \"tests/kit/gate.rs\"
{required}
"
    )
  }

  /// The artifact the falsifier row stands for, in names no lock table stages
  /// — so none of this depends on the one artifact the real table carries
  /// today.
  fn restricted_names() -> BTreeMap<&'static str, ArtifactNames> {
    BTreeMap::from([(
      "a/w.bin",
      ArtifactNames {
        local_dir: "Models/restrictedkit".to_string(),
        kit_dir: "restrictedkit".to_string(),
        module: "restricted".to_string(),
      },
    )])
  }

  /// The test channel run over a synthetic manifest and a synthetic tree.
  fn tested(manifest: &str, files: &BTreeMap<String, Vec<u8>>) -> Vec<String> {
    let rows = [row(
      "a/w.bin",
      "vendor/one",
      "commercial-face",
      CLEAR,
      RESTRICTED,
    )];
    let targets = compiled_targets(manifest, &|rel| files.get(rel).cloned());
    research_only_tested(
      &rows,
      &restricted_names(),
      &targets,
      &feature_closures(manifest),
    )
  }

  /// **The finding, in the smallest shape that shows it.** A target on the
  /// ORDINARY kit feature whose source spells the staged directory. Nothing
  /// about the module that manifests the artifact has changed, so
  /// [`research_only_wired`] and [`ungranted_wired_into_default`] both stay
  /// green while this suite runs against the restricted bytes.
  #[test]
  fn direction_two_reds_when_a_plain_feature_target_names_the_staged_directory() {
    let files = tree(&[(
      "tests/kit/gate.rs",
      "fn main() {\n  let bundle = \"Models/restrictedkit/w.mlmodelc\";\n  let _ = bundle;\n}\n",
    )]);
    let failures = tested(&one_target("required-features = [\"face\"]"), &files);
    assert_eq!(failures.len(), 1, "{failures:?}");
    assert!(failures[0].contains("\"kit_gate\""), "{failures:?}");
    assert!(failures[0].contains("tests/kit/gate.rs"), "{failures:?}");
    assert!(
      failures[0].contains("Models/restrictedkit/w.mlmodelc"),
      "{failures:?}"
    );
    assert!(
      failures[0].contains("`required-features = [\"face\"]`"),
      "{failures:?}"
    );
  }

  /// The second reach: an UNGATED `mod` that resolves to the fixture module,
  /// which names the artifact's own module path. The declaration is in the
  /// entry file and the reference is one file further on, which is why the
  /// source set has to be transitive rather than a scan of the `path` file.
  #[test]
  fn direction_two_reds_when_a_plain_feature_target_pulls_in_the_artifacts_module() {
    let files = tree(&[
      ("tests/kit/gate.rs", "mod restricted;\n\nfn main() {}\n"),
      (
        "tests/kit/restricted/mod.rs",
        "pub fn staged() -> &'static str {\n  the_crate::embeddings::restricted::STAGED_PATH\n}\n",
      ),
    ]);
    let failures = tested(&one_target("required-features = [\"face\"]"), &files);
    assert_eq!(failures.len(), 1, "{failures:?}");
    assert!(
      failures[0].contains("tests/kit/restricted/mod.rs"),
      "{failures:?}"
    );
    assert!(failures[0].contains("\"restricted\""), "{failures:?}");
  }

  /// The flat spelling of the same reach. `mod restricted;` resolves to
  /// `restricted.rs` just as readily as to `restricted/mod.rs`, and a channel
  /// that only looked at DIRECTORY names would see the second and walk past
  /// the first — so the file's own stem counts too, and this fixture names the
  /// artifact nowhere else.
  #[test]
  fn direction_two_reds_when_the_artifacts_module_resolves_to_a_flat_file() {
    let files = tree(&[
      ("tests/kit/gate.rs", "mod restricted;\n\nfn main() {}\n"),
      ("tests/kit/restricted.rs", "pub const DIM: usize = 512;\n"),
    ]);
    let failures = tested(&one_target("required-features = [\"face\"]"), &files);
    assert_eq!(failures.len(), 1, "{failures:?}");
    assert!(
      failures[0].contains("tests/kit/restricted.rs"),
      "{failures:?}"
    );
  }

  /// The same declaration behind the gate is GREEN, and that is the shape
  /// `face_gate` has: a target on plain `face` may name the artifact in code
  /// the commercial feature is what compiles.
  #[test]
  fn direction_two_leaves_a_gated_declaration_of_the_artifacts_module_alone() {
    let files = tree(&[
      (
        "tests/kit/gate.rs",
        "#[cfg(feature = \"commercial-face\")]\nmod restricted;\n\nfn main() {}\n",
      ),
      (
        "tests/kit/restricted/mod.rs",
        "pub fn staged() -> &'static str {\n  the_crate::embeddings::restricted::STAGED_PATH\n}\n",
      ),
    ]);
    let failures = tested(&one_target("required-features = [\"face\"]"), &files);
    assert!(failures.is_empty(), "{failures:?}");
  }

  /// `face_gate`'s other half: the reference inside an INLINE module the gate
  /// removes.
  #[test]
  fn direction_two_does_not_read_inside_an_inline_module_the_gate_removes() {
    let files = tree(&[(
      "tests/kit/gate.rs",
      "fn main() {}\n\n#[cfg(feature = \"commercial-face\")]\nmod under_the_gate {\n  pub fn dir() \
       -> &'static str {\n    \"Models/restrictedkit\"\n  }\n}\n",
    )]);
    let failures = tested(&one_target("required-features = [\"face\"]"), &files);
    assert!(failures.is_empty(), "{failures:?}");
  }

  /// And the mutation that proves the line above is the `#[cfg]` doing the
  /// work rather than the reader failing to look: the same inline module with
  /// the gate deleted is red.
  #[test]
  fn direction_two_reads_inside_the_same_inline_module_once_the_gate_is_gone() {
    let files = tree(&[(
      "tests/kit/gate.rs",
      "fn main() {}\n\nmod under_the_gate {\n  pub fn dir() -> &'static str {\n    \
       \"Models/restrictedkit\"\n  }\n}\n",
    )]);
    let failures = tested(&one_target("required-features = [\"face\"]"), &files);
    assert_eq!(failures.len(), 1, "{failures:?}");
    assert!(failures[0].contains("Models/restrictedkit"), "{failures:?}");
  }

  /// A target that asks for the commercial feature may name the artifact by
  /// every channel there is — including through a `#[path]`-redirected module,
  /// which is how all four real gate suites reach their fixtures.
  #[test]
  fn direction_two_accepts_a_commercial_target_that_names_the_artifact_everywhere() {
    let files = tree(&[
      (
        "tests/kit/gate.rs",
        "#[path = \"restricted/mod.rs\"]\nmod common;\n\nfn main() {\n  let _ = \
         \"Models/restrictedkit\";\n  let _ = the_crate::embeddings::restricted::STAGED_PATH;\n}\n",
      ),
      (
        "tests/kit/restricted/mod.rs",
        "pub const DIR: &str = \"restrictedkit\";\n",
      ),
    ]);
    let failures = tested(
      &one_target("required-features = [\"commercial-face\"]"),
      &files,
    );
    assert!(failures.is_empty(), "{failures:?}");
  }

  /// A target with NO `required-features` compiles in every configuration
  /// there is, `default` included, so naming the artifact there is the widest
  /// version of the same finding.
  #[test]
  fn direction_two_reds_when_a_target_that_names_the_artifact_asks_for_no_feature() {
    let files = tree(&[(
      "tests/kit/gate.rs",
      "fn main() {\n  let _ = \"Models/restrictedkit/w.mlmodelc\";\n}\n",
    )]);
    let failures = tested(&one_target(""), &files);
    assert_eq!(failures.len(), 1, "{failures:?}");
    assert!(
      failures[0].contains("`required-features = []`"),
      "{failures:?}"
    );
  }

  /// A `[[bench]]` is read the same way, and the manifest is why: this crate
  /// already declares `whisper_rtf_gate` as a `[[test]]` whose `path` is under
  /// `benches/`. The array a target is declared in is not where the boundary
  /// between "runs against the bytes" and "does not" lies.
  #[test]
  fn direction_two_reds_on_a_plain_feature_bench_as_readily_as_on_a_test() {
    let manifest = format!(
      "{TARGET_FEATURES}
[[bench]]
name = \"kit_encode\"
path = \"benches/kit/encode.rs\"
harness = false
required-features = [\"face\"]
"
    );
    let files = tree(&[(
      "benches/kit/encode.rs",
      "fn main() {\n  let _ = \"Models/restrictedkit/w.mlmodelc\";\n}\n",
    )]);
    let failures = tested(&manifest, &files);
    assert_eq!(failures.len(), 1, "{failures:?}");
    assert!(failures[0].contains("[[bench]]"), "{failures:?}");
  }

  /// **Prose is not compilation.** `tests/face/arcface/mod.rs` documents
  /// itself as "staged under `Models/facekit/`", and a reader that counted a
  /// `#[doc]` string would report a file's own documentation as a load — the
  /// same conflation [`cfg_features_in`] refuses one level down.
  #[test]
  fn direction_two_never_reads_a_reference_out_of_a_doc_comment() {
    let files = tree(&[(
      "tests/kit/gate.rs",
      "//! Mentions `Models/restrictedkit/` and the `restricted` module, in prose.\n\n/// So does \
       this: `Models/restrictedkit`, `restricted`.\nfn main() {}\n",
    )]);
    let failures = tested(&one_target("required-features = [\"face\"]"), &files);
    assert!(failures.is_empty(), "{failures:?}");
  }

  /// The wide clause in the test channel: the row is UNRESOLVED rather than
  /// forbidden, and what reds is not the absence of a commercial feature but
  /// the fact that `default` already turns on everything the target asks for.
  #[test]
  fn direction_two_reds_when_a_default_built_target_names_an_ungranted_artifact() {
    const SHIPS_THE_KIT: &str = "\
[features]
default = [\"face\"]
face = []
# Requires a commercial licence from the weights' author.
commercial-face = [\"face\"]

[[test]]
name = \"kit_gate\"
path = \"tests/kit/gate.rs\"
required-features = [\"face\"]
";
    let files = tree(&[(
      "tests/kit/gate.rs",
      "fn main() {\n  let _ = \"Models/restrictedkit/w.mlmodelc\";\n}\n",
    )]);
    let rows = [row("a/w.bin", "vendor/one", "face", CLEAR, UNGRANTED)];
    let targets = compiled_targets(SHIPS_THE_KIT, &|rel| files.get(rel).cloned());
    let failures = ungranted_tested_under_default(
      &rows,
      &restricted_names(),
      &targets,
      &feature_closures(SHIPS_THE_KIT),
      &feature_closure(SHIPS_THE_KIT, "default"),
    );
    assert_eq!(failures.len(), 1, "{failures:?}");
    assert!(
      failures[0].contains("already inside `default`'s"),
      "{failures:?}"
    );
    // The wording has to stay an OPEN QUESTION rather than a prohibition: this
    // row's terms are unresolved, and asserting they forbid anything would be
    // the register's own over-claim defect pointed backwards.
    assert!(
      failures[0].contains("NOTHING is established"),
      "{failures:?}"
    );
  }

  /// And the same target is green the moment `default` stops turning its
  /// feature on, which is this crate's actual shape.
  #[test]
  fn direction_two_leaves_an_ungranted_artifact_behind_an_opt_in_suite_alone() {
    let files = tree(&[(
      "tests/kit/gate.rs",
      "fn main() {\n  let _ = \"Models/restrictedkit/w.mlmodelc\";\n}\n",
    )]);
    let manifest = one_target("required-features = [\"face\"]");
    let rows = [row("a/w.bin", "vendor/one", "face", CLEAR, UNGRANTED)];
    let targets = compiled_targets(&manifest, &|rel| files.get(rel).cloned());
    let failures = ungranted_tested_under_default(
      &rows,
      &restricted_names(),
      &targets,
      &feature_closures(&manifest),
      &feature_closure(&manifest, "default"),
    );
    assert!(failures.is_empty(), "{failures:?}");
  }

  /// A `mod` this reader cannot resolve is a refusal, never a skip: a source
  /// set it gave up on silently is a suite direction 2 would clear without
  /// having read it.
  #[test]
  #[should_panic(expected = "resolves to neither")]
  fn the_source_set_reader_refuses_a_module_it_cannot_resolve() {
    let files = tree(&[("tests/kit/gate.rs", "mod missing;\n\nfn main() {}\n")]);
    let _ = tested(&one_target("required-features = [\"face\"]"), &files);
  }

  /// And so is a target whose entry file is not there at all.
  #[test]
  #[should_panic(expected = "is not on disk")]
  fn the_source_set_reader_refuses_a_target_whose_entry_file_is_missing() {
    let _ = tested(&one_target("required-features = [\"face\"]"), &tree(&[]));
  }

  /// A target that declares no `path` is refused rather than read as a target
  /// that compiles nothing.
  #[test]
  #[should_panic(expected = "declares no `path`")]
  fn the_target_reader_refuses_a_target_with_no_entry_file_declared() {
    let manifest = format!(
      "{TARGET_FEATURES}
[[test]]
name = \"kit_gate\"
required-features = [\"face\"]
"
    );
    let _ = tested(&manifest, &tree(&[]));
  }

  /// **Round 3, finding 1.** An attribute does not only sit on an item. On a
  /// MATCH ARM it ends at the `,`, and a skipper that ran to the next brace
  /// group swallowed the arm AFTER the gated one — the ungated arm that opens
  /// the bundle — and reported the target as naming nothing.
  #[test]
  fn direction_two_reds_when_a_gated_match_arm_hides_the_next_one() {
    let files = tree(&[(
      "tests/kit/gate.rs",
      "fn main() {\n  let path = match 0 {\n    #[cfg(feature = \"commercial-face\")]\n    0 => \
       (),\n    _ => {\n      \"Models/restrictedkit/w.mlmodelc\"\n    }\n  };\n  let _ = \
       path;\n}\n",
    )]);
    let failures = tested(&one_target("required-features = [\"face\"]"), &files);
    assert_eq!(failures.len(), 1, "{failures:?}");
    assert!(
      failures[0].contains("Models/restrictedkit/w.mlmodelc"),
      "{failures:?}"
    );
  }

  /// And the control: when the gate covers the arm that names the artifact
  /// TOO, there is nothing left to find. This is what proves the line above is
  /// the comma doing the work rather than the reader having stopped skipping.
  #[test]
  fn direction_two_leaves_a_match_whose_naming_arm_is_gated_alone() {
    let files = tree(&[(
      "tests/kit/gate.rs",
      "fn main() {\n  let path = match 0 {\n    #[cfg(feature = \"commercial-face\")]\n    0 => \
       (),\n    #[cfg(feature = \"commercial-face\")]\n    _ => {\n      \
       \"Models/restrictedkit/w.mlmodelc\"\n    }\n    _ => (),\n  };\n  let _ = path;\n}\n",
    )]);
    let failures = tested(&one_target("required-features = [\"face\"]"), &files);
    assert!(failures.is_empty(), "{failures:?}");
  }

  /// The same shape one grammar over: a struct literal's FIELDS are attributed
  /// and comma-separated too, so a gated field must not consume the field
  /// after it.
  #[test]
  fn direction_two_reds_when_a_gated_struct_field_hides_the_next_one() {
    let files = tree(&[(
      "tests/kit/gate.rs",
      "struct Fixture {\n  a: usize,\n  bundle: &'static str,\n}\n\nfn main() {\n  let _ = \
       Fixture {\n    #[cfg(feature = \"commercial-face\")]\n    a: 1,\n    bundle: \
       \"Models/restrictedkit/w.mlmodelc\",\n  };\n}\n",
    )]);
    let failures = tested(&one_target("required-features = [\"face\"]"), &files);
    assert_eq!(failures.len(), 1, "{failures:?}");
    assert!(
      failures[0].contains("Models/restrictedkit/w.mlmodelc"),
      "{failures:?}"
    );
  }

  /// **Round 3, finding 2.** `cfg_attr` can attach a `#[path]`, and a resolver
  /// that read only the DIRECT `#[path]` resolved `helper.rs` while rustc
  /// compiled `restricted.rs`. The redirected file is where the reference
  /// hides, so this is refused rather than resolved conventionally.
  #[test]
  #[should_panic(expected = "cfg_attr")]
  fn the_source_set_reader_refuses_a_cfg_attr_that_redirects_a_modules_path() {
    let files = tree(&[
      (
        "tests/kit/gate.rs",
        "#[cfg_attr(feature = \"face\", path = \"restricted.rs\")]\nmod helper;\n\nfn main() {}\n",
      ),
      ("tests/kit/helper.rs", "pub const N: usize = 1;\n"),
      (
        "tests/kit/restricted.rs",
        "pub const DIR: &str = \"Models/restrictedkit\";\n",
      ),
    ]);
    let _ = tested(&one_target("required-features = [\"face\"]"), &files);
  }

  /// The other half of the same refusal: a `cfg_attr` that attaches a further
  /// `#[cfg]` decides whether the module is compiled AT ALL, which is the one
  /// question this channel exists to answer.
  #[test]
  #[should_panic(expected = "cfg_attr")]
  fn the_source_set_reader_refuses_a_cfg_attr_that_gates_a_module() {
    let files = tree(&[
      (
        "tests/kit/gate.rs",
        "#[cfg_attr(feature = \"face\", cfg(feature = \"face\"))]\nmod helper;\n\nfn main() {}\n",
      ),
      ("tests/kit/helper.rs", "pub const N: usize = 1;\n"),
    ]);
    let _ = tested(&one_target("required-features = [\"face\"]"), &files);
  }

  /// And the nesting, which is the same redirection one level down: a
  /// `cfg_attr` may attach another `cfg_attr`, so `cfg_attr` is refused on a
  /// `mod` beside `path` and `cfg`.
  #[test]
  #[should_panic(expected = "cfg_attr")]
  fn the_source_set_reader_refuses_a_cfg_attr_that_nests_another() {
    let files = tree(&[
      (
        "tests/kit/gate.rs",
        "#[cfg_attr(test, cfg_attr(feature = \"face\", path = \"restricted.rs\"))]\nmod \
         helper;\n\nfn main() {}\n",
      ),
      ("tests/kit/helper.rs", "pub const N: usize = 1;\n"),
      (
        "tests/kit/restricted.rs",
        "pub const DIR: &str = \"Models/restrictedkit\";\n",
      ),
    ]);
    let _ = tested(&one_target("required-features = [\"face\"]"), &files);
  }

  /// A `cfg_attr` that attaches none of the three is left alone, and the module
  /// resolves conventionally. Refusing every `cfg_attr` would red the ordinary
  /// `#[cfg_attr(test, ...)]` lint plumbing for nothing.
  #[test]
  fn the_source_set_reader_accepts_a_cfg_attr_that_attaches_a_lint() {
    let manifest = one_target("required-features = [\"face\"]");
    let files = tree(&[
      (
        "tests/kit/gate.rs",
        "#[cfg_attr(test, allow(dead_code))]\nmod helper;\n\nfn main() {}\n",
      ),
      ("tests/kit/helper.rs", "pub const N: usize = 1;\n"),
    ]);
    let failures = tested(&manifest, &files);
    assert!(failures.is_empty(), "{failures:?}");
    let targets = compiled_targets(&manifest, &|rel| files.get(rel).cloned());
    assert_eq!(
      targets[0]
        .sources
        .iter()
        .map(|source| source.path.as_str())
        .collect::<Vec<_>>(),
      ["tests/kit/gate.rs", "tests/kit/helper.rs"],
      "the conventional resolution has to survive the lint attribute"
    );
  }

  /// A `cfg_attr` whose PREDICATE names a commercial feature is a conditional
  /// gate this reader cannot evaluate, wherever it sits.
  #[test]
  #[should_panic(expected = "cfg_attr")]
  fn the_source_set_reader_refuses_a_cfg_attr_predicated_on_a_commercial_feature() {
    let files = tree(&[(
      "tests/kit/gate.rs",
      "#[cfg_attr(feature = \"commercial-face\", allow(dead_code))]\nfn main() {}\n",
    )]);
    let _ = tested(&one_target("required-features = [\"face\"]"), &files);
  }

  /// **Round 3, finding 3.** `include!` splices another file's tokens in, so
  /// the source set is not closed under `mod` alone.
  #[test]
  fn direction_two_reds_when_an_included_file_names_the_artifact() {
    let files = tree(&[
      (
        "tests/kit/gate.rs",
        "include!(\"fixtures/paths.rs\");\n\nfn main() {}\n",
      ),
      (
        "tests/kit/fixtures/paths.rs",
        "const BUNDLE: &str = \"Models/restrictedkit/w.mlmodelc\";\n",
      ),
    ]);
    let failures = tested(&one_target("required-features = [\"face\"]"), &files);
    assert_eq!(failures.len(), 1, "{failures:?}");
    assert!(
      failures[0].contains("tests/kit/fixtures/paths.rs"),
      "{failures:?}"
    );
  }

  /// An `include!` this reader cannot resolve is refused exactly as an
  /// unresolvable `mod` is.
  #[test]
  #[should_panic(expected = "include!")]
  fn the_source_set_reader_refuses_an_include_it_cannot_resolve() {
    let files = tree(&[(
      "tests/kit/gate.rs",
      "include!(\"fixtures/missing.rs\");\n\nfn main() {}\n",
    )]);
    let _ = tested(&one_target("required-features = [\"face\"]"), &files);
  }

  /// `include_str!` embeds a fixture's TEXT in the binary, and a fixture that
  /// names the staged directory is a reference to it.
  #[test]
  fn direction_two_reds_when_an_embedded_text_file_names_the_staged_directory() {
    let files = tree(&[
      (
        "tests/kit/gate.rs",
        "const PLAN: &str = include_str!(\"fixtures/plan.json\");\n\nfn main() {\n  let _ = \
         PLAN;\n}\n",
      ),
      (
        "tests/kit/fixtures/plan.json",
        "{\"bundle\": \"Models/restrictedkit/w.mlmodelc\"}\n",
      ),
    ]);
    let failures = tested(&one_target("required-features = [\"face\"]"), &files);
    assert_eq!(failures.len(), 1, "{failures:?}");
    assert!(
      failures[0].contains("tests/kit/fixtures/plan.json"),
      "{failures:?}"
    );
  }

  /// And so is an invocation whose path this reader cannot even read. A
  /// `concat!` names a real file the compiler resolves at compile time, so
  /// returning "no include here" would drop it from the source set in silence
  /// — the difference between a residual that is stated and a hole that is
  /// not.
  #[test]
  #[should_panic(expected = "not one string literal")]
  fn the_source_set_reader_refuses_an_include_whose_path_is_not_a_literal() {
    let files = tree(&[(
      "tests/kit/gate.rs",
      "const PLAN: &str = include_str!(concat!(\"fixtures/\", \"plan.json\"));\n\nfn main() {\n  \
       let _ = PLAN;\n}\n",
    )]);
    let _ = tested(&one_target("required-features = [\"face\"]"), &files);
  }

  /// And an embedded file this reader cannot resolve is a refusal too.
  #[test]
  #[should_panic(expected = "include_str!")]
  fn the_source_set_reader_refuses_an_embedded_file_it_cannot_resolve() {
    let files = tree(&[(
      "tests/kit/gate.rs",
      "const PLAN: &str = include_str!(\"fixtures/missing.json\");\n\nfn main() {\n  let _ = \
       PLAN;\n}\n",
    )]);
    let _ = tested(&one_target("required-features = [\"face\"]"), &files);
  }

  /// `include_bytes!` embeds a fixture on exactly the same terms, and the
  /// reader searches it on exactly the same terms: what makes an embedded file
  /// scannable is that its bytes are TEXT, not which of the two macros named
  /// it.
  #[test]
  fn direction_two_reds_when_embedded_bytes_are_text_that_name_the_staged_directory() {
    let files = tree(&[
      (
        "tests/kit/gate.rs",
        "const PLAN: &[u8] = include_bytes!(\"fixtures/plan.txt\");\n\nfn main() {\n  let _ = \
         PLAN;\n}\n",
      ),
      (
        "tests/kit/fixtures/plan.txt",
        "bundle = Models/restrictedkit/w.mlmodelc\n",
      ),
    ]);
    let failures = tested(&one_target("required-features = [\"face\"]"), &files);
    assert_eq!(failures.len(), 1, "{failures:?}");
    assert!(
      failures[0].contains("tests/kit/fixtures/plan.txt"),
      "{failures:?}"
    );
  }

  /// **The stated limit, pinned rather than described.** Bytes that are not
  /// UTF-8 hold no text to search, so a reference spelled inside them is one
  /// this channel cannot make — and the fixture here spells the staged
  /// directory in plain ASCII behind a single invalid byte, so it is exactly
  /// that reference. What the reader must still do is RECORD the file as read
  /// and undecodable, which is the difference between a residual it can name
  /// and a hole it dropped in silence.
  #[test]
  fn direction_two_records_an_embedded_file_whose_bytes_are_not_text() {
    const NOT_TEXT: &[u8] = b"\xffModels/restrictedkit/w.mlmodelc\xff";
    let manifest = one_target("required-features = [\"face\"]");
    let files = tree_with_bytes(
      &[(
        "tests/kit/gate.rs",
        "const W: &[u8] = include_bytes!(\"fixtures/w.bin\");\n\nfn main() {\n  let _ = W;\n}\n",
      )],
      &[("tests/kit/fixtures/w.bin", NOT_TEXT)],
    );
    assert!(
      String::from_utf8(NOT_TEXT.to_vec()).is_err(),
      "the fixture only pins the limit while its bytes really are undecodable"
    );
    let failures = tested(&manifest, &files);
    assert!(failures.is_empty(), "{failures:?}");
    let targets = compiled_targets(&manifest, &|rel| files.get(rel).cloned());
    let embedded = &targets[0].sources[0].embedded;
    assert_eq!(embedded.len(), 1, "the file has to be resolved and READ");
    assert_eq!(embedded[0].path, "tests/kit/fixtures/w.bin");
    assert!(
      embedded[0].text.is_none(),
      "bytes that are not UTF-8 are recorded as unread, never decoded lossily into names the \
       compiler does not have"
    );
  }

  /// An `include!` the commercial gate removes contributes nothing, the same
  /// way a gated `mod` does.
  #[test]
  fn direction_two_leaves_an_included_file_behind_the_gate_alone() {
    let files = tree(&[
      (
        "tests/kit/gate.rs",
        "#[cfg(feature = \"commercial-face\")]\ninclude!(\"fixtures/paths.rs\");\n\nfn main() \
         {}\n",
      ),
      (
        "tests/kit/fixtures/paths.rs",
        "const BUNDLE: &str = \"Models/restrictedkit/w.mlmodelc\";\n",
      ),
    ]);
    let failures = tested(&one_target("required-features = [\"face\"]"), &files);
    assert!(failures.is_empty(), "{failures:?}");
  }

  /// **Round 3, finding 4.** Two tables over the SAME repository split every
  /// consumer of the lock: `kits_of` collects into a map and keeps the last,
  /// while `artifact_names` and direction 1's coverage take the first match. A
  /// staging kit hidden behind a duplicate header is refused at parse instead.
  #[test]
  #[should_panic(expected = "declared twice")]
  fn the_lock_reader_refuses_a_repeated_table_header() {
    let _ = parse_lock(
      "cache-epoch = \"1\"\n\n[\"FinDIT-Studio/facekit-coreml\"]\nkit = \"face\"\nlocal-dir = \
       \"Models/facekit\"\n\n[\"FinDIT-Studio/facekit-coreml\"]\nkit = \"arcface\"\nlocal-dir = \
       \"Models/facekit\"\n",
    );
  }

  /// Distinct headers are what the real lock has — the two speakerkit tables
  /// and the two whisper tables name different repositories — so the rule is
  /// keyed on the NAME and not on how alike two tables look.
  #[test]
  fn the_lock_reader_accepts_two_tables_that_stage_alike_under_different_names() {
    let tables = parse_lock(
      "[\"FinDIT-Studio/facekit-coreml\"]\nkit = \"face\"\nlocal-dir = \
       \"Models/facekit\"\n\n[\"FinDIT-Studio/facekit2-coreml\"]\nkit = \"arcface\"\nlocal-dir = \
       \"Models/facekit\"\n",
    );
    assert_eq!(tables.len(), 2);
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
    let derived = tree_gates(&[("a/w.bin", &["commercial-face"])]);
    let declared = features(&["default", "commercial-face"]);
    let in_source = features(&["commercial-face"]);
    assert!(
      commercial_features_gating_nothing_restricted(&rows, &derived, &declared, &in_source)
        .is_empty()
    );
  }

  #[test]
  fn direction_three_reds_when_a_commercial_feature_gates_only_granted_artifacts() {
    let rows = [
      row("a/w.bin", "vendor/one", "commercial-face", CLEAR, CLEAR),
      row(
        "b/w.bin",
        "vendor/one",
        "commercial-face",
        ATTRIBUTED,
        ATTRIBUTED,
      ),
    ];
    let derived = tree_gates(&[
      ("a/w.bin", &["commercial-face"]),
      ("b/w.bin", &["commercial-face"]),
    ]);
    let failures = commercial_features_gating_nothing_restricted(
      &rows,
      &derived,
      &features(&["default", "commercial-face"]),
      &features(&["commercial-face"]),
    );
    assert_eq!(failures.len(), 1, "{failures:?}");
    assert!(
      failures[0].contains("every artifact it gates is GRANTED at both layers"),
      "{failures:?}"
    );
    assert!(
      failures[0].contains("RESEARCH-ONLY") && failures[0].contains("UNRESOLVED"),
      "the retire message must name BOTH causes a commercial gate can stand on, \
       or the next reader learns only one of them: {failures:?}"
    );
  }

  /// **Direction 3 runs backwards, and an unresolved row must not trip it.**
  /// The gate is not standing over nothing — the row is not clear. What it
  /// stands on is an open QUESTION rather than a found prohibition, which is
  /// why the retire message above has to name two causes and not one.
  #[test]
  fn direction_three_keeps_a_commercial_gate_that_covers_an_unresolved_row() {
    let rows = [row(
      "a/w.bin",
      "vendor/one",
      "commercial-face",
      UNGRANTED,
      ATTRIBUTED,
    )];
    let derived = tree_gates(&[("a/w.bin", &["commercial-face"])]);
    assert!(
      commercial_features_gating_nothing_restricted(
        &rows,
        &derived,
        &features(&["default", "commercial-face"]),
        &features(&["commercial-face"]),
      )
      .is_empty()
    );
  }

  #[test]
  fn direction_three_reds_when_a_commercial_feature_gates_nothing_at_all() {
    let rows = [row("a/w.bin", "vendor/one", "speaker", CLEAR, CLEAR)];
    let derived = tree_gates(&[("a/w.bin", &["speaker"])]);
    let failures = commercial_features_gating_nothing_restricted(
      &rows,
      &derived,
      &features(&["default", "commercial-face"]),
      &features(&["commercial-face"]),
    );
    assert_eq!(failures.len(), 1, "{failures:?}");
    assert!(
      failures[0].contains("no licence row is gated by it"),
      "{failures:?}"
    );
  }

  /// **The finding.** `commercial-face = []` declared in `[features]`, named by
  /// a restricted row, and referenced by no `#[cfg(feature = ...)]` anywhere.
  /// Enabling it compiles nothing differently — so the artifact is behind no
  /// gate at all, while a row-driven check reports a protected artifact.
  #[test]
  fn direction_three_reds_when_a_commercial_feature_gates_no_code_at_all() {
    let rows = [row(
      "a/w.bin",
      "vendor/one",
      "commercial-face",
      CLEAR,
      RESTRICTED,
    )];
    let derived = tree_gates(&[("a/w.bin", &["commercial-face"])]);
    let failures = commercial_features_gating_nothing_restricted(
      &rows,
      &derived,
      &features(&["default", "commercial-face"]),
      &features(&["speaker"]),
    );
    assert_eq!(failures.len(), 1, "{failures:?}");
    assert!(
      failures[0].contains("It is a name, not a gate"),
      "{failures:?}"
    );
  }

  #[test]
  fn direction_three_leaves_plain_features_alone() {
    let rows = [row("a/w.bin", "vendor/one", "speaker", CLEAR, CLEAR)];
    let derived = tree_gates(&[("a/w.bin", &["speaker"])]);
    assert!(
      commercial_features_gating_nothing_restricted(
        &rows,
        &derived,
        &features(&["default", "speaker", "whisper"]),
        &features(&["speaker", "whisper"]),
      )
      .is_empty()
    );
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
    assert!(failures[0].contains("does not BEGIN with"), "{failures:?}");
  }

  /// **The finding, counterexample one.** A substring search finds the phrase
  /// inside its own negation and passes the sentence that says the opposite.
  #[test]
  fn the_doc_rule_reds_on_a_first_sentence_that_negates_the_warning() {
    let declared = features(&["commercial-face"]);
    let written = docs(&[(
      "commercial-face",
      "This feature no longer requires a commercial license.",
    )]);
    let failures = commercial_features_without_the_phrase(&declared, &written);
    assert_eq!(failures.len(), 1, "{failures:?}");
    assert!(failures[0].contains("does not BEGIN with"), "{failures:?}");
  }

  /// **The finding, counterexample two.** `!` is a sentence terminator, so the
  /// FIRST sentence here is the endorsement — which is the reading the rule
  /// exists to prevent, arriving before the correction.
  #[test]
  fn the_doc_rule_reds_when_an_endorsement_precedes_the_warning() {
    let declared = features(&["commercial-face"]);
    let written = docs(&[(
      "commercial-face",
      "Cleared for commercial use! This feature requires a commercial license.",
    )]);
    let failures = commercial_features_without_the_phrase(&declared, &written);
    assert_eq!(failures.len(), 1, "{failures:?}");
    assert!(
      failures[0].contains("Cleared for commercial use!"),
      "{failures:?}"
    );
  }

  /// A warning that opens correctly and is then taken back in the same
  /// sentence has not warned anybody.
  #[test]
  fn the_doc_rule_reds_when_the_opening_warning_is_qualified_away() {
    let declared = features(&["commercial-face"]);
    let written = docs(&[(
      "commercial-face",
      "Requires a commercial license unless you are an academic. Adds the face embedder.",
    )]);
    let failures = commercial_features_without_the_phrase(&declared, &written);
    assert_eq!(failures.len(), 1, "{failures:?}");
    assert!(
      failures[0].contains("takes the warning back"),
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
  fn a_sentence_ends_at_a_full_stop_a_bang_or_a_question_mark() {
    assert_eq!(
      first_sentence("Requires a commercial licence."),
      "Requires a commercial licence."
    );
    assert_eq!(
      first_sentence("Requires a commercial licence. And more."),
      "Requires a commercial licence."
    );
    assert_eq!(
      first_sentence("Cleared for commercial use! Requires a commercial licence."),
      "Cleared for commercial use!"
    );
    assert_eq!(
      first_sentence("Commercial? Requires a commercial licence."),
      "Commercial?"
    );
    assert_eq!(
      first_sentence("No terminator at all"),
      "No terminator at all"
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
    assert_eq!(failures.len(), 2, "{failures:?}");
    assert!(failures.iter().all(|f| f.contains(SHA)), "{failures:?}");
  }

  /// **The finding.** Both rows are `permissive`, so a verdict-class comparison
  /// reads them as agreeing — while one says the bytes are MIT and the other
  /// says Apache-2.0. Those are different grants over one file, and at most one
  /// of them is the licence of the artifact.
  #[test]
  fn identical_bytes_called_mit_by_one_row_and_apache_by_another_red() {
    const SHA: &str = "cccc000000000000000000000000000000000000000000000000000000000000";
    let rows = [
      Artifact {
        key: Key::Sha256(SHA),
        ..row(
          "one/w.bin",
          "vendor/one",
          "kit",
          Terms::permissive("MIT", RETAIN_NOTICE, "MIT, for a falsifier"),
          Terms::unresolved("open, for a falsifier"),
        )
      },
      Artifact {
        key: Key::Sha256(SHA),
        ..row(
          "two/w.bin",
          "vendor/two",
          "kit",
          Terms::permissive("Apache-2.0", RETAIN_NOTICE, "Apache-2.0, for a falsifier"),
          Terms::unresolved("open, for a falsifier"),
        )
      },
    ];
    let failures = contradictory_terms(&rows);
    assert_eq!(failures.len(), 1, "{failures:?}");
    assert!(failures[0].contains("weights"), "{failures:?}");
    assert!(failures[0].contains("MIT"), "{failures:?}");
    assert!(failures[0].contains("Apache-2.0"), "{failures:?}");
  }

  /// **The finding, second half.** Two `research-only` rows over one SHA-256
  /// that forbid materially different things are a contradiction, not
  /// agreement — the class is identical and the obligations are not.
  #[test]
  fn two_research_only_rows_with_different_restrictions_red() {
    const SHA: &str = "dddd000000000000000000000000000000000000000000000000000000000000";
    const NO_REDISTRIBUTION: &[&str] = &["no-redistribution-of-the-weights"];
    let rows = [
      Artifact {
        key: Key::Sha256(SHA),
        ..row(
          "one/w.bin",
          "vendor/one",
          "kit",
          Terms::research_only(
            "LicenseRef-research-only",
            RETAIN_NOTICE,
            "research only, redistribution permitted, for a falsifier",
          ),
          Terms::unresolved("open, for a falsifier"),
        )
      },
      Artifact {
        key: Key::Sha256(SHA),
        ..row(
          "two/w.bin",
          "vendor/two",
          "kit",
          Terms::research_only(
            "LicenseRef-research-only",
            NO_REDISTRIBUTION,
            "research only, redistribution FORBIDDEN, for a falsifier",
          ),
          Terms::unresolved("open, for a falsifier"),
        )
      },
    ];
    let failures = contradictory_terms(&rows);
    assert_eq!(failures.len(), 1, "{failures:?}");
    assert!(
      failures[0].contains("no-redistribution-of-the-weights"),
      "{failures:?}"
    );
  }

  /// And an attribution row and a permissive row over the same bytes disagree
  /// even before the identifiers are read — one makes credit a condition of
  /// the grant and the other does not.
  #[test]
  fn identical_bytes_with_different_obligations_red() {
    const SHA: &str = "eeee000000000000000000000000000000000000000000000000000000000000";
    let rows = [
      Artifact {
        key: Key::Sha256(SHA),
        ..row(
          "one/w.bin",
          "vendor/one",
          "kit",
          Terms::attribution(
            "CC-BY-4.0",
            CREDIT_AUTHOR,
            "credit required, for a falsifier",
          ),
          Terms::unresolved("open, for a falsifier"),
        )
      },
      Artifact {
        key: Key::Sha256(SHA),
        ..row(
          "two/w.bin",
          "vendor/two",
          "kit",
          Terms::permissive("CC-BY-4.0", RETAIN_NOTICE, "notice only, for a falsifier"),
          Terms::unresolved("open, for a falsifier"),
        )
      },
    ];
    assert_eq!(contradictory_terms(&rows).len(), 1);
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

  /// An unresolved layer carries no identifier and no obligation, so two
  /// unresolved rows over one hash agree by construction — which is correct:
  /// neither has claimed anything to contradict.
  #[test]
  fn unresolved_rows_over_the_same_bytes_do_not_contradict() {
    assert!(
      Terms::unresolved("open, for a falsifier")
        .licence()
        .is_empty()
    );
    assert_eq!(
      Terms::unresolved("open, for a falsifier").restrictions(),
      NOTHING_ESTABLISHED.iter().copied().collect::<BTreeSet<_>>()
    );
  }

  // --- the manifest readers ------------------------------------------------

  const DOCTORED_FEATURES: &str = "\
[features]
default = [\"speaker\"]
speaker = [\"dep:diaric\"]
# Requires a commercial licence from the weights' author.
# Adds the face embedder.
commercial-face = [\"dep:facelib\", \"speaker\"]

# A comment about the section, not about the feature after the blank line.

lid = [\"dep:rustfft\"]
";

  // ── The gate reader: what a `#[cfg]` on a loader may and may not derive ──

  /// Wrap one `#[cfg]` spelling around the `identity` loader declaration.
  fn loader_with(attributes: &str) -> String {
    format!("//! a module.\n\n{attributes}\npub mod identity;\n\n#[cfg(test)]\nmod tests;\n")
  }

  /// **The whole enumeration, in one run.** Every one of these makes the name
  /// `identity` appear in the text above the declaration, and NONE of them
  /// makes `identity` a feature the loader REQUIRES in order to compile. The
  /// substring reader derived `{identity}` from every single one, row
  /// reconciliation then agreed with the row's claim, and default-reachability
  /// concluded the artifact was withheld — while the loader compiled by
  /// default.
  ///
  /// Reported together rather than one assertion per shape: the point is the
  /// class, and a reader fixed for the negation alone would still pass three of
  /// these.
  #[test]
  fn the_gate_reader_refuses_every_cfg_it_cannot_read_as_a_requirement() {
    let cases: &[(&str, String)] = &[
      (
        "a negation — `identity` OFF is what compiles the loader",
        loader_with(r#"#[cfg(not(feature = "identity"))]"#),
      ),
      (
        "an alternative — the target arm compiles the loader with `identity` off",
        loader_with(r#"#[cfg(any(target_os = "macos", feature = "identity"))]"#),
      ),
      (
        "a conjunction — two features, and the row can only claim one",
        loader_with(r#"#[cfg(all(feature = "identity", feature = "speaker"))]"#),
      ),
      (
        "a `cfg_attr` — it attaches an attribute conditionally, it is not a gate",
        loader_with(r#"#[cfg_attr(feature = "identity", allow(dead_code))]"#),
      ),
      (
        "a nested negation inside an otherwise positive `all`",
        loader_with(r#"#[cfg(all(not(feature = "identity"), unix))]"#),
      ),
      (
        "a non-feature predicate — a real gate, but not one `[features]` declares",
        loader_with(r#"#[cfg(target_os = "macos")]"#),
      ),
      (
        "two `cfg` attributes — the loader needs BOTH, and `all(..)` is refused above",
        loader_with("#[cfg(feature = \"identity\")]\n#[cfg(feature = \"speaker\")]"),
      ),
    ];

    let mut accepted = Vec::new();
    for (what, source) in cases {
      if let Ok(gates) = gates_of_module(source, "identity") {
        accepted.push(format!("  {what}\n    read as {gates:?}"));
      }
    }
    assert!(
      accepted.is_empty(),
      "these `#[cfg]` spellings were read as a feature REQUIREMENT, and not one of them is \
       one. A derived gate is what directions 2 and 3 reason about: an ungranted loader whose \
       gate is derived from a negation reads as withheld from `default` while it compiles in \
       `default`. Each must fail closed instead.\n{}",
      accepted.join("\n")
    );
  }

  /// A comment is not a gate. `//`, `///` and `//!` above the declaration all
  /// carried their `feature = "..."` into the derived set, so a sentence
  /// mentioning a feature by name invented a gate that no `#[cfg]` imposed.
  #[test]
  fn the_gate_reader_never_derives_a_gate_from_a_comment() {
    let cases: &[(&str, String)] = &[
      (
        "a line comment",
        loader_with(
          "// unlike feature = \"phantom\", this one ships\n#[cfg(feature = \"identity\")]",
        ),
      ),
      (
        "a doc comment",
        loader_with(
          "/// Behind feature = \"phantom\" until the terms resolve.\n#[cfg(feature = \"identity\")]",
        ),
      ),
      (
        "an inner doc comment",
        loader_with("//! See feature = \"phantom\".\n#[cfg(feature = \"identity\")]"),
      ),
      (
        "a comment between two attributes",
        loader_with(
          "#[allow(unused)]\n// feature = \"phantom\" is not a gate\n#[cfg(feature = \"identity\")]",
        ),
      ),
    ];

    let mut wrong = Vec::new();
    for (what, source) in cases {
      match gates_of_module(source, "identity") {
        Ok(gates) if gates == BTreeSet::from(["identity".to_string()]) => {}
        other => wrong.push(format!("  {what}: {other:?}")),
      }
    }
    assert!(
      wrong.is_empty(),
      "each of these carries exactly one gate — `identity` — and prose naming another feature \
       beside it. Only the `#[cfg]` decides:\n{}",
      wrong.join("\n")
    );
  }

  /// The supported form is read, however it is laid out, and an enclosing
  /// `mod` block's gate counts as much as the declaration's own.
  #[test]
  fn the_gate_reader_reads_the_supported_form() {
    assert_eq!(
      gates_of_module(&loader_with("#[cfg(feature = \"identity\")]"), "identity"),
      Ok(BTreeSet::from(["identity".to_string()]))
    );
    assert_eq!(
      gates_of_module(
        &loader_with("#[cfg(\n  feature = \"identity\"\n)]"),
        "identity"
      ),
      Ok(BTreeSet::from(["identity".to_string()])),
      "an attribute rustfmt wrapped over three lines is the same attribute"
    );
    assert_eq!(
      gates_of_module(
        &loader_with(
          "#[doc = \"gated on feature = \\\"phantom\\\"\"]\n#[cfg(feature = \"identity\")]"
        ),
        "identity"
      ),
      Ok(BTreeSet::from(["identity".to_string()])),
      "a non-`cfg` attribute names no gate, whatever string it carries"
    );
    assert_eq!(
      gates_of_module(
        "#[cfg(feature = \"identity\")]\nmod outer {\n  pub mod identity;\n}\n",
        "identity"
      ),
      Ok(BTreeSet::from(["identity".to_string()])),
      "an enclosing module's gate is required for the declaration to compile too"
    );
    assert!(
      gates_of_module("pub mod identity;\n", "identity")
        .is_ok_and(|gates: BTreeSet<String>| gates.is_empty()),
      "an ungated loader derives NO gate — that is direction 2's finding, not an error"
    );
    assert!(
      gates_of_module("pub mod identity;\nmod identity;\n", "identity").is_err(),
      "two declarations mean the reader could be reading the wrong one"
    );
    assert!(
      gates_of_module("pub mod identity", "identity").is_err(),
      "source this reader cannot parse must fail closed, not read as ungated"
    );
  }

  /// **The third reader in the same class.** `fp16_pinned_bundles` is
  /// direction 1's SECOND enumeration of what a glob stages, and its value is
  /// entirely in being independent — a roster entry it cannot see is a staged
  /// bundle whose licence row nobody checks. The line reader it replaces
  /// required `path: "` to open a trimmed line, so a rustfmt wrap hid one; the
  /// token reader sees the field wherever it is laid out, and still refuses the
  /// bundle paths that file quotes in prose and in its `note` fields.
  #[test]
  fn the_roster_reader_reads_a_path_field_however_it_is_laid_out() {
    let mut found = Vec::new();
    collect_field_literals(
      "const R: &[E] = &[\n  E {\n    path:\n      \"vendorkit/Wrapped.mlmodelc\",\n    \
       note: \"same floor as Other.mlmodelc\",\n  },\n];\n"
        .parse()
        .expect("tokenises"),
      "path",
      &mut found,
    );
    assert_eq!(
      found,
      vec!["vendorkit/Wrapped.mlmodelc".to_string()],
      "the wrapped `path` field must be read, and the `note` field's quoted bundle must not be"
    );
  }

  /// Prose is not a roster. This file documents bundle paths in `//!` and `///`
  /// blocks; a reader that widened to "any literal ending in `.mlmodelc`" would
  /// invent entries out of them.
  #[test]
  fn the_roster_reader_never_reads_a_path_out_of_prose() {
    let mut found = Vec::new();
    collect_field_literals(
      "//! see `lid/Prose.mlmodelc` for the epsilon table\n/// and `doc/Prose.mlmodelc`\nfn f() {}\n"
        .parse()
        .expect("tokenises"),
      "path",
      &mut found,
    );
    assert!(found.is_empty(), "read {found:?} out of prose");
  }

  // ── The source sweep: what counts as a feature the tree NAMES ──────────────

  /// **The sibling scanner, enumerated the same way.** `cfg_features_in_source`
  /// decides whether a `commercial-` feature gates any code at all, and it
  /// looked for the substring `feature = "` anywhere in a file. Prose and
  /// string literals both matched, so a feature named only in a SENTENCE read
  /// as a live gate and direction 3's third clause passed vacuously over it.
  #[test]
  fn the_source_sweep_never_counts_prose_or_a_string_literal() {
    let cases: &[(&str, &str)] = &[
      (
        "a line comment",
        "// gated on feature = \"ghost\"\nfn f() {}\n",
      ),
      (
        "a doc comment",
        "/// Behind feature = \"ghost\".\npub fn f() {}\n",
      ),
      (
        "an inner doc comment",
        "//! `#[cfg(feature = \"ghost\")]` guards this module.\n",
      ),
      (
        "a block comment",
        "/* #[cfg(feature = \"ghost\")] */\nfn f() {}\n",
      ),
      (
        "a string literal",
        "const S: &str = \"#[cfg(feature = \\\"ghost\\\")]\";\n",
      ),
      (
        "a raw string literal",
        "const S: &str = r#\"#[cfg(feature = \"ghost\")]\"#;\n",
      ),
      (
        "a `doc` attribute",
        "#[doc = \"gated on feature = \\\"ghost\\\"\"]\npub fn f() {}\n",
      ),
    ];

    let mut counted = Vec::new();
    for (what, source) in cases {
      let found = cfg_features_in(source);
      if found.contains("ghost") {
        counted.push(format!("  {what}: read as naming {found:?}"));
      }
    }
    assert!(
      counted.is_empty(),
      "a feature named in prose or in a string compiles nothing differently. Counting it lets \
       a `commercial-` gate that guards no code at all read as live, which is precisely the \
       clause this sweep exists to enforce.\n{}",
      counted.join("\n")
    );
  }

  /// Every shape that IS conditional compilation is counted, wherever the
  /// feature sits inside the predicate: this sweep asks whether the feature
  /// changes what compiles, not whether it is REQUIRED — a negation qualifies
  /// as much as a plain gate. (That is the opposite of what
  /// [`gates_of_module`] asks, and the split is the point.)
  #[test]
  fn the_source_sweep_counts_every_conditional_compilation_site() {
    let cases: &[(&str, &str)] = &[
      ("a plain gate", "#[cfg(feature = \"plain\")]\nfn f() {}\n"),
      ("a negation", "#[cfg(not(feature = \"neg\"))]\nfn f() {}\n"),
      (
        "an alternative",
        "#[cfg(any(unix, feature = \"alt\"))]\nfn f() {}\n",
      ),
      (
        "a `cfg_attr`",
        "#[cfg_attr(feature = \"ca\", derive(Debug))]\nstruct S;\n",
      ),
      (
        "a `cfg_attr` rustfmt wrapped",
        "#[cfg_attr(\n  feature = \"wrapped\",\n  serde(default)\n)]\nstruct S;\n",
      ),
      (
        "the `cfg!` macro",
        "fn f() -> bool { cfg!(feature = \"bang\") }\n",
      ),
      (
        "an inner attribute",
        "#![cfg(feature = \"inner\")]\nfn f() {}\n",
      ),
      (
        "an attribute inside a `macro_rules!` body",
        "macro_rules! m {\n  () => {\n    #[cfg(feature = \"macro\")]\n    fn g() {}\n  };\n}\n",
      ),
      (
        "an attribute on a nested item",
        "mod m {\n  impl S {\n    #[cfg(feature = \"nested\")]\n    fn g() {}\n  }\n}\n",
      ),
    ];

    let mut missed = Vec::new();
    for (what, source) in cases {
      let expected = source
        .split("feature = \"")
        .nth(1)
        .and_then(|t| t.split('"').next())
        .expect("each case names one feature");
      let found = cfg_features_in(source);
      if !found.contains(expected) {
        missed.push(format!("  {what}: expected {expected:?}, found {found:?}"));
      }
    }
    assert!(
      missed.is_empty(),
      "a conditional-compilation site this sweep cannot see makes the feature it names read as \
       gating nothing.\n{}",
      missed.join("\n")
    );
  }

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

  /// **The one fail-open cell in the reader roster.** `feature_docs` strips
  /// the `#` BEFORE it checks whether the line was indented, and a
  /// whitespace-led non-comment line did not clear `pending`. So an indented
  /// comment inside a multi-line array attached itself to the NEXT key: this
  /// manifest gave `commercial-b` the sentence "Requires a commercial licence."
  /// and the doc rule went green on a feature nobody documented.
  ///
  /// The roster claims this reader is "red, never green". That claim was wrong
  /// by exactly one cell, and this is the cell.
  const INDENTED_COMMENT_FEATURES: &str = "\
[features]
commercial-a = [
  # Requires a commercial licence.
  \"dep:x\"]
commercial-b = []
";

  #[test]
  fn a_comment_inside_a_multi_line_array_documents_nothing() {
    let docs = feature_docs(INDENTED_COMMENT_FEATURES);
    assert_eq!(
      docs.get("commercial-b").map(String::as_str),
      Some(""),
      "an indented comment inside `commercial-a`'s array must not become \
       `commercial-b`'s documentation: a feature nobody documented would pass \
       the doc rule on a sentence written about another one"
    );
    assert_eq!(
      docs.get("commercial-a").map(String::as_str),
      Some(""),
      "nor its own: the comment sits BELOW the key it is indented under, and \
       this reader documents a key from the block ABOVE it"
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
  /// mutation direction 2's `default` clause exists for, read through the real
  /// manifest reader rather than a hand-built set.
  const LEAKY_FEATURES: &str = "\
[features]
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
    let rows = [row(
      "a/w.bin",
      "vendor/one",
      "commercial-face",
      CLEAR,
      RESTRICTED,
    )];
    let derived = tree_gates(&[("a/w.bin", &["commercial-face"])]);
    let failures = research_only_wired(&rows, &derived, &feature_closures(LEAKY_FEATURES));
    assert!(
      failures
        .iter()
        .any(|f| f.contains("plain `cargo add coremlit`")),
      "{failures:?}"
    );
  }

  // --- valid TOML the hand-rolled reader could not see ---------------------
  //
  // Every constant below is a manifest Cargo obeys, spelling
  // `default = ["identity"]` — the exact mutation
  // `no_ungranted_artifact_is_wired_into_default` was verified against.
  // The old reader returned NO entries for any of them, so `default`'s closure
  // came back as `{"default"}`, every ungranted artifact looked opt-in, and the
  // check stayed green on a manifest that ships the bytes. That is what makes
  // this an enumeration and not a style note: the mutation used to prove the
  // check worked was true only for the ONE formatting it happened to be
  // written in.

  /// **Spelling 1 — an indented key.** TOML does not care about leading
  /// whitespace; the old reader skipped every line that had any.
  const DEFAULT_INDENTED: &str = "\
[features]
  default = [\"identity\"]
identity = [\"dep:rustfft\"]
";

  /// **Spelling 2 — a literal string.** TOML's single-quoted strings are
  /// strings; the old reader split the value on `\"` and found none.
  const DEFAULT_SINGLE_QUOTED: &str = "\
[features]
default = ['identity']
identity = ['dep:rustfft']
";

  /// **Spelling 3 — a quoted key.** `\"default\"` and `default` are the same
  /// key in TOML; the old reader compared the raw text before the first `=`
  /// and never matched.
  const DEFAULT_QUOTED_KEY: &str = "\
[features]
\"default\" = [\"identity\"]
identity = [\"dep:rustfft\"]
";

  /// **Spelling 4 — a comment carrying `]` inside a multi-line array.** The old
  /// reader stopped collecting at the first line containing `]`, so the value
  /// ended before the entry did.
  const DEFAULT_COMMENT_WITH_BRACKET: &str = "\
[features]
default = [ # the shipping set (see [features] above)
  \"identity\",
]
identity = [\"dep:rustfft\"]
";

  /// **Spelling 5 — a dotted key, no `[features]` header at all.** There was no
  /// header to find, so the block came back empty and so did every closure
  /// built from it.
  const DEFAULT_DOTTED_KEY: &str = "\
features.default = [\"identity\"]
features.identity = [\"dep:rustfft\"]
";

  /// **Spelling 6 — a non-canonical header.** `[ features ]` is the same table;
  /// the old reader compared the trimmed line to the literal `\"[features]\"`.
  const DEFAULT_SPACED_HEADER: &str = "\
[ features ]
default = [\"identity\"]
identity = [\"dep:rustfft\"]
";

  /// Every spelling above, named, so a failure says which one regressed.
  fn every_missed_spelling() -> [(&'static str, &'static str); 6] {
    [
      ("indented key", DEFAULT_INDENTED),
      ("literal (single-quoted) string", DEFAULT_SINGLE_QUOTED),
      ("quoted key", DEFAULT_QUOTED_KEY),
      (
        "comment carrying `]` in a multi-line array",
        DEFAULT_COMMENT_WITH_BRACKET,
      ),
      ("dotted key with no `[features]` header", DEFAULT_DOTTED_KEY),
      ("non-canonical `[ features ]` header", DEFAULT_SPACED_HEADER),
    ]
  }

  /// The reader sees `identity` in `default`'s entries under every one of them.
  ///
  /// Every spelling is reported in ONE run rather than short-circuiting on the
  /// first, because the enumeration is the result here: a reader that fixes the
  /// spelling it was last caught on and misses the next one has not been fixed.
  #[test]
  fn the_reader_sees_default_under_every_valid_spelling() {
    let mut missed = Vec::new();
    for (label, manifest) in every_missed_spelling() {
      let entries = feature_entries(manifest, "default");
      let names = feature_names(manifest);
      let closure = feature_closure(manifest, "default");
      if entries != vec!["identity".to_string()]
        || names != features(&["default", "identity"])
        || closure != features(&["default", "identity"])
      {
        missed.push(format!(
          "{label}: entries {entries:?}, names {names:?}, closure {closure:?}"
        ));
      }
    }
    assert!(
      missed.is_empty(),
      "the reader must see what Cargo sees, and does not for {} of {} spellings:\n{}",
      missed.len(),
      every_missed_spelling().len(),
      missed.join("\n")
    );
  }

  /// **The check itself, driven through every spelling.** Not the reader in
  /// isolation: an ungranted row behind `identity` must be REPORTED as WIRED
  /// into `default` for each one, because that is the state the manifest
  /// actually describes to Cargo.
  #[test]
  fn direction_two_reds_from_default_under_every_valid_spelling() {
    let rows = [row(
      REDIMNET_SHAPED,
      "vendor/one",
      "identity",
      UNGRANTED,
      ATTRIBUTED,
    )];
    let derived = tree_gates(&[(REDIMNET_SHAPED, &["identity"])]);
    let mut passed_vacuously = Vec::new();
    for (label, manifest) in every_missed_spelling() {
      let failures =
        ungranted_wired_into_default(&rows, &derived, &feature_closure(manifest, "default"));
      if failures.len() != 1 || !failures[0].contains("plain `cargo add coremlit`") {
        passed_vacuously.push(format!("{label}: {failures:?}"));
      }
    }
    assert!(
      passed_vacuously.is_empty(),
      "the check stayed green on {} of {} manifests that ship the ungranted bytes:\n{}",
      passed_vacuously.len(),
      every_missed_spelling().len(),
      passed_vacuously.join("\n")
    );
  }

  /// And the opt-in shape still passes under the same spellings, so the test
  /// above is detecting `default`'s contents rather than the parser change.
  #[test]
  fn direction_two_stays_green_when_the_same_spellings_leave_default_empty() {
    let rows = [row(
      REDIMNET_SHAPED,
      "vendor/one",
      "identity",
      UNGRANTED,
      ATTRIBUTED,
    )];
    let derived = tree_gates(&[(REDIMNET_SHAPED, &["identity"])]);
    for manifest in [
      "[features]\n  default = []\nidentity = [\"dep:rustfft\"]\n",
      "[features]\ndefault = []\nidentity = ['dep:rustfft']\n",
      "[ features ]\ndefault = []\nidentity = [\"dep:rustfft\"]\n",
      "features.default = []\nfeatures.identity = [\"dep:rustfft\"]\n",
    ] {
      assert!(
        ungranted_wired_into_default(&rows, &derived, &feature_closure(manifest, "default"))
          .is_empty(),
        "{manifest:?}"
      );
    }
  }

  /// **Fail closed.** A manifest the reader cannot decode must PANIC, not come
  /// back empty: an empty feature graph is what every reachability check here
  /// reads as "nothing ships by default", so a silent decode failure is a
  /// silent pass.
  #[test]
  fn an_undecodable_manifest_panics_rather_than_reading_as_empty() {
    let mut read_as_empty = Vec::new();
    for (label, manifest) in [
      ("not TOML at all", "default = [\"identity\"\n"),
      ("no `[features]` table", "[package]\nname = \"coremlit\"\n"),
      (
        "a feature whose value is not an array",
        "[features]\ndefault = \"identity\"\n",
      ),
      (
        "a feature whose entries are not strings",
        "[features]\ndefault = [1, 2]\n",
      ),
      ("`features` is not a table", "features = \"identity\"\n"),
    ] {
      let hook = std::panic::take_hook();
      std::panic::set_hook(Box::new(|_| {}));
      let outcome = std::panic::catch_unwind(|| feature_closure(manifest, "default"));
      std::panic::set_hook(hook);
      if let Ok(closure) = outcome {
        read_as_empty.push(format!("{label}: read as {closure:?}"));
      }
    }
    assert!(
      read_as_empty.is_empty(),
      "{} of these manifests were read rather than refused; a feature graph nobody can decode \
       is not an empty one:\n{}",
      read_as_empty.len(),
      read_as_empty.join("\n")
    );
  }

  /// A feature the manifest genuinely does not declare still reads as absent —
  /// failing closed on an undecodable document must not turn every lookup into
  /// a panic.
  #[test]
  fn an_undeclared_feature_reads_as_absent_rather_than_panicking() {
    assert!(feature_entries(CLEAN_FEATURES, "no-such-feature").is_empty());
    assert_eq!(
      feature_closure(CLEAN_FEATURES, "no-such-feature"),
      features(&["no-such-feature"])
    );
  }

  /// Every closure the manifest reader builds, not just `default`'s — the
  /// input direction 2 now runs on.
  #[test]
  fn the_closure_reader_covers_every_declared_feature() {
    let closures = feature_closures(DOCTORED_FEATURES);
    assert_eq!(
      closures.keys().cloned().collect::<BTreeSet<_>>(),
      features(&["default", "speaker", "commercial-face", "lid"])
    );
    assert_eq!(
      closures["commercial-face"],
      features(&["commercial-face", "speaker"])
    );
  }
}
