//! Golden feature-map test — pins the mono-crate flat feature contract.
//!
//! The restructure collapsed five crates into feature-gated modules and renamed
//! each per-crate feature to a flat one (`FEATURE_MAP.md`). This test PINS that
//! contract against its three sources of truth, so a renamed, dropped,
//! re-composed, or cross-kit-leaking feature — or a silently dropped CI combo or
//! clippy flag — cannot land:
//!
//!   1. `Cargo.toml` `[features]` — the exact feature-name set AND the exact
//!      dependency set of every feature (a leak like `whisper` pulling `vad`
//!      changes a set and reds). BOTH manifests: this crate's and the sibling
//!      `coremlit-parity`'s, which owns the three third-party parity oracles.
//!   2. `FEATURE_MAP.md`'s rename table — parsed (not substring-scanned) so a
//!      removed/altered bare-crate row reds even if the token survives elsewhere
//!      in the doc.
//!   3. `.github/workflows/ci.yml` — the curated `--features` combo matrices,
//!      parsed structurally PER JOB and compared as exact sets, so dropping OR
//!      commenting out any curated combo (including the bare-core `""`) reds.
//!      Two jobs carry one: `features` (this crate) and `parity` (the oracles).
//!      Those two jobs' clippy steps are pinned as well — the package,
//!      `--no-deps`, `--all-targets` and `-- -D warnings` on every arm, the
//!      `if: ${{ !cancelled() }}`, and the `clippy` component installed ahead of
//!      the step — because a row whose lint step lost a flag still runs green
//!      while it lints less (#158).
//!
//!      Four more pieces of that file are pinned that no matrix row reaches: the
//!      `check` job's toolchain install (the `rustfmt` and `clippy` components,
//!      asked for ahead of the `cargo fmt` and `cargo clippy` steps that need
//!      them), its clippy step (`--all-targets --all-features`, `-- -D
//!      warnings`, and no `-p`, since it is the one pass that lints the whole
//!      workspace with every feature on), its doc step (`--no-deps
//!      --all-features` under `RUSTDOCFLAGS: -D warnings`), and the `if:` of
//!      every `model-tests` step after the staging step, which is what keeps one
//!      red step from marking the checks after it `skipped`, together with the
//!      `id:` the closing `Gate ledger` must read for each of them and the
//!      roster of checks every shard runs.
//!
//! The oracle features (`speaker-oracle`, `clap-oracle`, `vad-bundled`) are NOT
//! this crate's any more — `dia` and `textclap` are unpublished git sources that
//! `cargo publish` rejects, so they and their nine parity binaries moved to the
//! never-published `coremlit-parity` package. `align-oracle` did NOT move: it
//! only turns on a feature of `asry`, a dependency this crate keeps either way.
//!
//! Hermetic: pure file reads (via `CARGO_MANIFEST_DIR`), no models, no cargo
//! invocation, no feature needs enabling.
//!
//! The package ships this test, `Cargo.toml` and `FEATURE_MAP.md`, but neither
//! `.github/workflows/ci.yml` nor the sibling `coremlit-parity` package, so a
//! `cargo test` from the published tarball finds neither of those. The pins that
//! read them skip there, each naming the missing file on stderr. Inside a
//! workspace a missing file is a failure, so deleting the workflow does not
//! quietly turn its pins off.

// The workspace-root anchor, FOUND by searching upward for the `[workspace]`
// manifest rather than counted in `../` hops — see its module doc.
#[path = "support/workspace_root.rs"]
#[allow(dead_code)]
mod workspace_root;

use std::{collections::BTreeSet, path::Path};

/// Read a file addressed relative to the crate manifest directory — one the
/// published tarball carries too.
fn read_rel(rel: &str) -> String {
  std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join(rel))
    .unwrap_or_else(|e| panic!("read {rel}: {e}"))
}

fn manifest() -> String {
  read_rel("Cargo.toml")
}

/// What a pin prints when the repository is not there to read.
fn skip_notice(rel: &str) -> String {
  format!("feature_map: skipped — {rel} is not in this source tree (a published tarball?)")
}

/// A file of the repository outside this crate's directory, addressed from the
/// workspace root, or `None` — said by name on stderr — when there is no
/// workspace root to address it from.
///
/// No root is what a `cargo test` from the published tarball looks like: the
/// package carries neither `.github/workflows/ci.yml` nor the sibling
/// `coremlit-parity` package, and nothing above it declares a `[workspace]`. A
/// root that IS there makes the file's absence a failure and not a skip, so a
/// deleted workflow reds its pins instead of silently ending them.
fn repo_file(root: Option<&Path>, rel: &str) -> Option<String> {
  let Some(root) = root else {
    eprintln!("{}", skip_notice(rel));
    return None;
  };
  let path = root.join(rel);
  Some(std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display())))
}

/// The sibling `coremlit-parity` manifest, found from the workspace root, or
/// `None` outside the repository.
fn parity_manifest() -> Option<String> {
  repo_file(
    workspace_root::try_workspace_root().as_deref(),
    "coremlit-parity/Cargo.toml",
  )
}

/// `.github/workflows/ci.yml`, at the workspace root — found, not counted — or
/// `None` outside the repository.
fn ci_yml() -> Option<String> {
  repo_file(
    workspace_root::try_workspace_root().as_deref(),
    ".github/workflows/ci.yml",
  )
}

/// The intended flat feature graph — the single in-test source that both the
/// name-set and the per-feature dependency-set assertions are driven from. Any
/// drift in `Cargo.toml` (a renamed feature, a dropped dep, or a CROSS-KIT LEAK
/// such as adding `"vad"` to `whisper`) changes a set here and reds.
fn expected_features() -> Vec<(&'static str, Vec<&'static str>)> {
  vec![
    ("default", vec![]),
    ("serde", vec!["dep:serde", "windit?/serde"]),
    ("tracing", vec!["dep:tracing"]),
    (
      "whisper",
      vec![
        "dep:libc",
        "dep:mach2",
        "dep:rand",
        "dep:serde_json",
        "dep:tokenizers",
        "dep:unicode_categories",
      ],
    ),
    (
      "nl-recognizer",
      vec!["whisper", "dep:objc2-natural-language"],
    ),
    // `dep:humantime` is `AlignerOptions`'s canonical `Display`; `dep:serde_json`
    // and `dep:serde` read a model's own `{token: id}` vocabulary
    // (`vocab::Vocabulary`), refusing a repeated token; `dep:tokenizers` parses
    // the tokenizer document asry's seam parses again at load, to read back the
    // columns the seam reserves. asry's `emissions` feature already pulls
    // `tokenizers` 0.23, which depends on serde and serde_json, so those entries
    // add no crate version to the lockfile.
    (
      "align",
      vec![
        "dep:asry",
        "dep:humantime",
        "dep:serde_json",
        "dep:serde",
        "dep:tokenizers",
      ],
    ),
    ("align-oracle", vec!["align", "asry/alignment"]),
    ("speaker", vec!["dep:diaric"]),
    ("vad", vec!["dep:zuoer"]),
    // `serde_json` is `embeddings::tokenizer_guard`, shared with `siglip`: the
    // guard reads a post-processor's structure back out through its serde
    // representation, because `TemplateProcessing` exposes no accessor for it.
    // Not a cross-kit leak — `tokenizers` already depends on `serde_json`, so
    // the row adds an entry here without adding a crate to the graph.
    (
      "clap",
      vec![
        "dep:rustfft",
        "dep:tokenizers",
        "dep:windit",
        "dep:serde_json",
      ],
    ),
    (
      "granite",
      // `dep:serde_json`: the BPE merge table read back out of the loaded model
      // for token_index's separatorless fast lane (#72); tokenizers depends on
      // serde_json itself, so no crate is added to the graph.
      vec![
        "dep:tokenizers",
        "dep:windit",
        "windit/text",
        "dep:sha2",
        "dep:serde_json",
      ],
    ),
    (
      "ced",
      vec!["dep:rustfft", "dep:soundevents-dataset", "dep:windit"],
    ),
    ("lid", vec!["dep:rustfft", "dep:windit"]),
    // No `windit`: this door embeds ONE window and the caller owns any
    // windowing or averaging across several of them, so the shared window
    // engine that `ced`/`lid`/`granite` need has nothing to do here.
    ("identity", vec!["dep:rustfft"]),
    // `dep:serde_json`: the same shared `embeddings::tokenizer_guard` as `clap`.
    (
      "siglip",
      vec!["dep:tokenizers", "dep:pixon", "dep:sha2", "dep:serde_json"],
    ),
    // `face`'s only dependency is `sha2`, and it is there for one reason:
    // `FaceEmbedder::load` hashes the artifact directory it loads, so an
    // embedding carries the identity of the WEIGHTS that produced it rather
    // than of a schema two unrelated artifacts could share (issue #138 §3).
    // The exact dep set is the assertion — a second one reds this row.
    ("face", vec!["dep:sha2"]),
    // The ONE `commercial-` feature. It adds no dependency at all — the artifact
    // it gates is a manifest of four constants — so the exact set is `["face"]`
    // and nothing else, and a second entry here would mean the gate had started
    // pulling code rather than protecting bytes.
    ("commercial-face-arcface", vec!["face"]),
  ]
}

/// The intended feature graph of the sibling `coremlit-parity` package — the
/// oracle half of the contract, pinned with the same exact-set discipline. Each
/// oracle rides its OWN feature (so one row's `ort` build is not forced on the
/// others) and turns on exactly the `coremlit` module feature it measures; a
/// leak like `clap-oracle` pulling `coremlit/speaker` changes a set and reds.
fn expected_parity_features() -> Vec<(&'static str, Vec<&'static str>)> {
  vec![
    ("default", vec![]),
    (
      "speaker-oracle",
      vec![
        "coremlit/speaker",
        "dep:dia",
        "dep:diaric",
        "dia/ort",
        "dia/bundled-segmentation",
      ],
    ),
    ("clap-oracle", vec!["coremlit/clap", "dep:textclap"]),
    (
      "vad-bundled",
      vec!["coremlit/vad", "dep:silero", "silero/bundled"],
    ),
  ]
}

/// The former per-crate kits and the flat module-feature each bare crate maps
/// to. Drives the rename-table row check, so a REMOVED bare-crate row reds.
const BARE_CRATE_MAP: &[(&str, &str)] = &[
  ("whisperkit", "whisper"),
  ("alignkit", "align"),
  ("speakerkit", "speaker"),
  ("vadkit", "vad"),
  ("clapkit", "clap"),
];

/// The curated CI feature combos the mono-crate restructure committed to — the
/// EXACT intended set of `jobs.features.strategy.matrix.features` entries in
/// `.github/workflows/ci.yml`, as raw (unquoted) combo strings. The empty
/// string is a real member: the bare-core `default = []` run (ci.yml `- ""`).
/// `ci_feature_combos` parses the ACTIVE matrix and the test asserts exact set
/// equality against this, so removing OR commenting out any entry (the bare-core
/// `""` included) drops it from the parsed set and reds.
const INTENDED_CI_COMBOS: &[&str] = &[
  "", // bare core / none (`default = []`)
  "whisper",
  "align",
  "speaker",
  "speaker,serde",
  "vad",
  "whisper,vad",
  "align-oracle",
  "clap",
  "granite",
  "siglip",
  "ced",
  "lid",
  "identity",
  "identity,speaker",
  "face",
  // NOT folded into the two all-on rows below, and that is the point of a
  // `commercial-` gate: an opt-in that the crate's own "everything on"
  // configuration turns on is not an opt-in. Its own row is what builds the
  // four gate suites and runs their hermetic halves.
  "commercial-face-arcface",
  "whisper,align,speaker,vad,clap,granite,siglip,ced,lid,identity,face,serde,tracing,nl-recognizer",
  "whisper,align-oracle,speaker,vad,clap,granite,siglip,ced,lid,identity,face,serde,tracing,nl-recognizer",
];

/// The curated combos of ci.yml's `parity` job — one row per third-party oracle
/// plus an all-on row, run against `coremlit-parity`. Same exact-set discipline
/// as [`INTENDED_CI_COMBOS`]: deleting or commenting out a row reds, so an
/// oracle cannot quietly stop being built once it no longer rides this crate's
/// own feature matrix.
const INTENDED_PARITY_CI_COMBOS: &[&str] = &[
  "speaker-oracle",
  "clap-oracle",
  "vad-bundled",
  "speaker-oracle,clap-oracle,vad-bundled",
];

/// The text of the `[features]` table (its lines, blank/comment lines included).
fn features_block(manifest: &str) -> String {
  let mut out = String::new();
  let mut in_features = false;
  for line in manifest.lines() {
    if line.starts_with('[') {
      in_features = line.trim() == "[features]";
      continue;
    }
    if in_features {
      out.push_str(line);
      out.push('\n');
    }
  }
  out
}

/// The feature *names* declared in the `[features]` block. A feature key sits at
/// column 0; an array's continuation lines are indented (and so skipped).
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

/// The dependency set of one feature — the quoted entries of its `[..]` value,
/// robust to a value spread over multiple (indented) lines.
fn feature_deps(block: &str, feature: &str) -> BTreeSet<String> {
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
  // Quoted contents are the odd-indexed pieces of a split on '"'.
  buf
    .split('"')
    .skip(1)
    .step_by(2)
    .map(str::to_string)
    .collect()
}

/// The column `jobs.<name>:` keys sit at in `.github/workflows/ci.yml`.
const JOB_INDENT: usize = 2;

/// Parse the ACTIVE `jobs.<job>.strategy.matrix.features` list from a ci.yml
/// text into the set of `--features` combo strings that job runs.
///
/// Structural, not substring, and SCOPED TO ONE JOB: it enters at the
/// `  <job>:` key, leaves at the next key in that column (so a later job's
/// matrix cannot be folded into this one's set — the split into `features` +
/// `parity` made that a live confusion), and within the job takes the
/// `features:` key that FOLLOWS `matrix:` (so the `features` JOB name and the
/// `cargo build --features` step cannot be mistaken for it). It collects each
/// `- "..."` item's inner value (the empty `- ""` is the empty-string member),
/// SKIPS any line whose first non-space char is `#` (a commented-out
/// `# - "..."` entry does NOT count as present, and does not end the list), and
/// stops at the first dedent to the key's column or left of it.
fn ci_feature_combos(yaml: &str, job: &str) -> BTreeSet<String> {
  let job_key = format!("{job}:");
  let mut combos = BTreeSet::new();
  let mut in_job = false;
  let mut seen_matrix = false;
  let mut key_indent: Option<usize> = None;
  for line in yaml.lines() {
    let indent = line.len() - line.trim_start().len();
    let trimmed = line.trim_start();
    // Comments and blanks are transparent everywhere: a commented-out entry is
    // skipped (not counted) WITHOUT terminating the list or the job.
    if trimmed.is_empty() || trimmed.starts_with('#') {
      continue;
    }
    if !in_job {
      in_job = indent == JOB_INDENT && trimmed == job_key;
      continue;
    }
    // A dedent to the job's own column (or left of it) ends the job.
    if indent <= JOB_INDENT {
      break;
    }
    let Some(ki) = key_indent else {
      if trimmed == "matrix:" {
        seen_matrix = true;
      } else if seen_matrix && trimmed == "features:" {
        key_indent = Some(indent);
      }
      continue;
    };
    // A dedent to the key's column (or left of it) ends the list.
    if indent <= ki {
      break;
    }
    if let Some(inner) = trimmed.strip_prefix("- ").and_then(quoted_inner) {
      combos.insert(inner);
    }
  }
  combos
}

/// The text between the first pair of `"` in `s` (`""` → the empty string).
fn quoted_inner(s: &str) -> Option<String> {
  let start = s.find('"')?;
  let rest = &s[start + 1..];
  let end = rest.find('"')?;
  Some(rest[..end].to_string())
}

/// A pinned combo list as an owned set.
fn owned(combos: &[&str]) -> BTreeSet<String> {
  combos.iter().map(|s| (*s).to_string()).collect()
}

/// The intended `features`-job combos as an owned set.
fn intended_ci_combos() -> BTreeSet<String> {
  owned(INTENDED_CI_COMBOS)
}

/// What the clippy step of one matrix job must run. The `features` job lints
/// `coremlit` under each curated combo, and its bare-core row (`""`) takes the
/// arm with no `--features`; the `parity` job lints `coremlit-parity` under each
/// oracle feature, and no row of that matrix is empty.
struct IntendedClippy {
  job: &'static str,
  package: &'static str,
  empty_arm: bool,
}

const FEATURES_CLIPPY: IntendedClippy = IntendedClippy {
  job: "features",
  package: "coremlit",
  empty_arm: true,
};

const PARITY_CLIPPY: IntendedClippy = IntendedClippy {
  job: "parity",
  package: "coremlit-parity",
  empty_arm: false,
};

/// The `if:` a clippy step carries. A step's default condition is `success()`,
/// so without it a red test step above hides the lint verdict.
const CLIPPY_STEP_CONDITION: &str = "${{ !cancelled() }}";

/// The argument that aims a clippy command at the matrix row's features.
const MATRIX_FEATURES_ARG: &str = r#"--features "${{ matrix.features }}""#;

/// One `key: value` entry of a step, with the lines under it: the inline value
/// first, then any deeper lines (a `run: |` script, a `with:` map).
type CiEntry = (String, Vec<String>);

/// `line` up to a trailing ` #` comment, which both YAML and the shell the `run`
/// scripts feed read as one.
fn without_comment(line: &str) -> &str {
  line
    .split_once(" #")
    .map_or(line, |(code, _)| code)
    .trim_end()
}

/// The lines of one ci.yml job as `(indent, text)`, scoped the way
/// [`ci_feature_combos`] scopes it — from its `  <job>:` key to the next line at
/// that column or left of it — with blank and comment lines dropped (a
/// commented-out step is not a step) and trailing comments cut.
fn ci_job_lines<'a>(yaml: &'a str, job: &str) -> Vec<(usize, &'a str)> {
  let job_key = format!("{job}:");
  let mut lines = Vec::new();
  let mut in_job = false;
  for line in yaml.lines() {
    let trimmed = line.trim_start();
    if trimmed.is_empty() || trimmed.starts_with('#') {
      continue;
    }
    let indent = line.len() - trimmed.len();
    let text = without_comment(trimmed);
    if !in_job {
      in_job = indent == JOB_INDENT && text == job_key;
      continue;
    }
    if indent <= JOB_INDENT {
      break;
    }
    lines.push((indent, text));
  }
  lines
}

/// `key: value` as an entry whose body starts with the inline value. A
/// block-scalar indicator (`|`, `>`) is not a value: the lines below it are.
fn ci_entry(text: &str) -> CiEntry {
  let (key, value) = text.split_once(':').unwrap_or((text, ""));
  let value = value.trim();
  let body = if value.is_empty() || value.starts_with(['|', '>']) {
    Vec::new()
  } else {
    vec![value.to_string()]
  };
  (key.trim().to_string(), body)
}

/// The steps of a job's `steps:` list, each as its entries. Empty when the key is
/// absent or the list does not open with a `- ` item, so a reshaped job reds the
/// checks below instead of being passed over.
fn ci_steps(job: &[(usize, &str)]) -> Vec<Vec<CiEntry>> {
  let Some(at) = job.iter().position(|&(_, text)| text == "steps:") else {
    return Vec::new();
  };
  let steps_indent = job[at].0;
  let mut steps: Vec<Vec<CiEntry>> = Vec::new();
  let mut first_indent = None;
  for &(indent, text) in &job[at + 1..] {
    if indent <= steps_indent {
      break;
    }
    let item_indent = *first_indent.get_or_insert(indent);
    // A `- ` item opens a step, and its first entry sits two columns in, where
    // every later entry of the step sits too.
    let opens_step = indent == item_indent && text.starts_with("- ");
    if opens_step {
      steps.push(Vec::new());
    }
    let Some(step) = steps.last_mut() else {
      return Vec::new();
    };
    if opens_step || indent == item_indent + 2 {
      step.push(ci_entry(if opens_step { &text[2..] } else { text }));
    } else if let Some((_, body)) = step.last_mut() {
      body.push(text.to_string());
    }
  }
  steps
}

/// The lines under `key:` in a step, empty when the step has no such entry.
fn ci_body<'a>(step: &'a [CiEntry], key: &str) -> &'a [String] {
  step
    .iter()
    .find(|(name, _)| name == key)
    .map_or(&[][..], |(_, body)| body.as_slice())
}

/// The shell commands of a step's `run` script, with a line that ends in a
/// backslash joined to the one after it, so a re-wrapped command reads as one.
fn ci_commands(step: &[CiEntry]) -> Vec<String> {
  let mut commands: Vec<String> = Vec::new();
  let mut joining = false;
  for line in ci_body(step, "run") {
    let (text, continues) = line
      .strip_suffix('\\')
      .map_or((line.as_str(), false), |head| (head.trim_end(), true));
    match commands.last_mut() {
      Some(last) if joining => {
        last.push(' ');
        last.push_str(text);
      }
      _ => commands.push(text.to_string()),
    }
    joining = continues;
  }
  commands
}

/// Whether a shell command invokes `cargo <subcommand>`.
fn runs_cargo(command: &str, subcommand: &str) -> bool {
  let tokens: Vec<&str> = command.split_whitespace().collect();
  tokens.windows(2).any(|pair| pair == ["cargo", subcommand])
}

/// Whether a shell command invokes `cargo clippy`.
fn runs_clippy(command: &str) -> bool {
  runs_cargo(command, "clippy")
}

/// Whether one of `steps` asks the toolchain install for `component`: a
/// `components:` entry under its `with:`, which is a comma-separated list.
fn installs_component(steps: &[Vec<CiEntry>], component: &str) -> bool {
  steps.iter().any(|step| {
    ci_body(step, "with").iter().any(|line| {
      line.strip_prefix("components:").is_some_and(|list| {
        list
          .split(',')
          .any(|installed| installed.trim().trim_matches(['"', '\'']) == component)
      })
    })
  })
}

/// What, if anything, has drifted in one job's clippy step, named. In order:
/// exactly one step runs `cargo clippy`; it carries `if: ${{ !cancelled() }}`; a
/// step ahead of it installs `components: clippy`; it has a command with
/// `--features "${{ matrix.features }}"` and, when the matrix has an empty row,
/// one without `--features`; and EVERY clippy command in it carries
/// `cargo clippy -p <package>`, `--no-deps` and `--all-targets` ahead of the `--`
/// and `-D warnings` after it.
///
/// Only the step's `run` lines are read. Its `name:` spells `--all-targets` as
/// well, so a read of the whole step would keep a dropped flag looking present.
fn clippy_step_drift(yaml: &str, want: &IntendedClippy) -> Result<(), String> {
  let job = want.job;
  let steps = ci_steps(&ci_job_lines(yaml, job));
  let clippy: Vec<usize> = steps
    .iter()
    .enumerate()
    .filter(|(_, step)| ci_commands(step).iter().any(|command| runs_clippy(command)))
    .map(|(at, _)| at)
    .collect();
  let [at] = clippy[..] else {
    return Err(format!(
      "ci.yml `{job}` job: expected exactly one step running `cargo clippy`, found {} among {} \
       parsed step(s) — the clippy step was dropped, duplicated or commented out, or the job's \
       `steps:` list changed shape",
      clippy.len(),
      steps.len()
    ));
  };
  let step = &steps[at];

  let condition = ci_body(step, "if");
  if condition.first().map(String::as_str) != Some(CLIPPY_STEP_CONDITION) {
    return Err(format!(
      "ci.yml `{job}` job: the clippy step must carry `if: {CLIPPY_STEP_CONDITION}` so a red \
       test step above cannot hide its verdict, found {condition:?}"
    ));
  }

  if !installs_component(&steps[..at], "clippy") {
    return Err(format!(
      "ci.yml `{job}` job: no step ahead of the clippy step installs `components: clippy`"
    ));
  }

  let commands: Vec<String> = ci_commands(step)
    .into_iter()
    .filter(|command| runs_clippy(command))
    .collect();
  let on_matrix = |command: &str| command.contains(MATRIX_FEATURES_ARG);
  if !commands.iter().any(|command| on_matrix(command)) {
    return Err(format!(
      "ci.yml `{job}` job: no clippy command in the step carries `{MATRIX_FEATURES_ARG}`, so \
       the matrix row's features are never linted"
    ));
  }
  if want.empty_arm && commands.iter().all(|command| on_matrix(command)) {
    return Err(format!(
      "ci.yml `{job}` job: no clippy command in the step runs without `--features`, so the \
       bare-core row (`\"\"`) has nothing to lint it"
    ));
  }
  for command in &commands {
    let arm = if on_matrix(command) {
      "the `--features` arm"
    } else {
      "the empty-features arm"
    };
    let tokens: Vec<&str> = command.split_whitespace().collect();
    let separator = tokens.iter().position(|token| *token == "--");
    let (cargo_args, lint_args) = tokens.split_at(separator.unwrap_or(tokens.len()));
    let package_args = format!("cargo clippy -p {}", want.package);
    let needs = [
      (
        package_args.as_str(),
        cargo_args
          .windows(4)
          .any(|run| run == ["cargo", "clippy", "-p", want.package]),
      ),
      ("--no-deps", cargo_args.contains(&"--no-deps")),
      ("--all-targets", cargo_args.contains(&"--all-targets")),
      (
        "-- -D warnings",
        lint_args.windows(2).any(|pair| pair == ["-D", "warnings"]),
      ),
    ];
    if let Some((need, _)) = needs.iter().find(|(_, present)| !present) {
      return Err(format!(
        "ci.yml `{job}` job: {arm} of the clippy step must carry `{need}`, found: {command}"
      ));
    }
  }
  Ok(())
}

/// Panic with [`clippy_step_drift`]'s message, if it has one.
fn assert_clippy_step(yaml: &str, want: &IntendedClippy) {
  if let Err(drift) = clippy_step_drift(yaml, want) {
    panic!("{drift}");
  }
}

/// The `check` job: the one pass that lints and documents the WHOLE workspace
/// with every feature on, `coremlit-parity`'s oracle features included. No
/// matrix row stands in for it — each `features` and `parity` row lints one
/// package under one feature set — so its two steps are pinned on their own.
const CHECK_JOB: &str = "check";

/// The position, among the `check` job's parsed `steps`, of the one that runs
/// `cargo <subcommand>`, or what is wrong with how many there are.
fn check_job_step_at(steps: &[Vec<CiEntry>], subcommand: &str) -> Result<usize, String> {
  let found: Vec<usize> = steps
    .iter()
    .enumerate()
    .filter(|(_, step)| {
      ci_commands(step)
        .iter()
        .any(|command| runs_cargo(command, subcommand))
    })
    .map(|(at, _)| at)
    .collect();
  let [at] = found[..] else {
    return Err(format!(
      "ci.yml `{CHECK_JOB}` job: expected exactly one step running `cargo {subcommand}`, found {} \
       among {} parsed step(s) — the step was dropped, duplicated or commented out, or the job's \
       `steps:` list changed shape",
      found.len(),
      steps.len()
    ));
  };
  Ok(at)
}

/// The `check` job's one step that runs `cargo <subcommand>`, or what is wrong
/// with how many there are.
fn check_job_step(yaml: &str, subcommand: &str) -> Result<Vec<CiEntry>, String> {
  let steps = ci_steps(&ci_job_lines(yaml, CHECK_JOB));
  let at = check_job_step_at(&steps, subcommand)?;
  Ok(steps.into_iter().nth(at).expect("`at` indexes `steps`"))
}

/// Whether a `cargo` argument aims the command at one package.
fn names_a_package(arg: &str) -> bool {
  matches!(arg, "-p" | "--package") || arg.starts_with("--package=")
}

/// What, if anything, has drifted in the `check` job's clippy step, named. In
/// order: exactly one step runs `cargo clippy`, and EVERY clippy command in it
/// names no package and carries `--all-targets` and `--all-features` ahead of the
/// `--` and `-D warnings` after it.
///
/// No package, because this is the one lint pass over the whole workspace:
/// `--all-features` reaches the oracle features only through `coremlit-parity`,
/// which a `-p coremlit` run leaves out. `--all-targets` with `--all-features` is
/// also what compiles the `harness = false` benches, which neither `cargo test`
/// nor `clippy --tests` reaches and each of which declares `required-features`.
///
/// Only the step's `run` lines are read, as for the other jobs' clippy steps.
fn check_clippy_drift(yaml: &str) -> Result<(), String> {
  let step = check_job_step(yaml, "clippy")?;
  for command in ci_commands(&step)
    .into_iter()
    .filter(|command| runs_clippy(command))
  {
    let tokens: Vec<&str> = command.split_whitespace().collect();
    let separator = tokens.iter().position(|token| *token == "--");
    let (cargo_args, lint_args) = tokens.split_at(separator.unwrap_or(tokens.len()));
    if cargo_args.iter().any(|arg| names_a_package(arg)) {
      return Err(format!(
        "ci.yml `{CHECK_JOB}` job: the clippy step must lint the whole workspace, but it names a \
         package, so `--all-features` no longer reaches `coremlit-parity`'s oracle features, \
         found: {command}"
      ));
    }
    let needs = [
      ("--all-targets", cargo_args.contains(&"--all-targets")),
      ("--all-features", cargo_args.contains(&"--all-features")),
      (
        "-- -D warnings",
        lint_args.windows(2).any(|pair| pair == ["-D", "warnings"]),
      ),
    ];
    if let Some((need, _)) = needs.iter().find(|(_, present)| !present) {
      return Err(format!(
        "ci.yml `{CHECK_JOB}` job: the clippy step must carry `{need}`, found: {command}"
      ));
    }
  }
  Ok(())
}

/// Whether a rustc or rustdoc flag string denies warnings.
fn denies_warnings(flags: &str) -> bool {
  let tokens: Vec<&str> = flags
    .trim()
    .trim_matches(['"', '\''])
    .split_whitespace()
    .collect();
  tokens.contains(&"-Dwarnings") || tokens.windows(2).any(|pair| pair == ["-D", "warnings"])
}

/// What, if anything, has drifted in the `check` job's doc step, named. In order:
/// exactly one step runs `cargo doc`, EVERY doc command in it carries `--no-deps`
/// and `--all-features`, and its `env:` sets `RUSTDOCFLAGS` to deny warnings.
///
/// `RUSTDOCFLAGS` is what makes the step a gate: rustdoc reports a broken
/// intra-doc link as a warning, so without `-D warnings` the step passes over
/// every one of them. `--all-features` is what documents the feature-gated
/// modules at all.
fn check_doc_drift(yaml: &str) -> Result<(), String> {
  let step = check_job_step(yaml, "doc")?;
  for command in ci_commands(&step)
    .into_iter()
    .filter(|command| runs_cargo(command, "doc"))
  {
    let tokens: Vec<&str> = command.split_whitespace().collect();
    for need in ["--no-deps", "--all-features"] {
      if !tokens.contains(&need) {
        return Err(format!(
          "ci.yml `{CHECK_JOB}` job: the doc step must carry `{need}`, found: {command}"
        ));
      }
    }
  }
  let flags = ci_body(&step, "env")
    .iter()
    .find_map(|line| line.strip_prefix("RUSTDOCFLAGS:"));
  if !flags.is_some_and(denies_warnings) {
    return Err(format!(
      "ci.yml `{CHECK_JOB}` job: the doc step's `env:` must set `RUSTDOCFLAGS: -D warnings`, so \
       a broken intra-doc link fails the job instead of passing it, found: {flags:?}"
    ));
  }
  Ok(())
}

/// What, if anything, has drifted in the toolchain the `check` job installs,
/// named. In order: exactly one step runs `cargo fmt` and a step ahead of it
/// installs `components: rustfmt`; exactly one step runs `cargo clippy` and a
/// step ahead of it installs `components: clippy`.
///
/// The job runs both tools, and a toolchain step that does not ask for their
/// components leaves each to whatever the runner image carries, so the job can
/// pass on one image and fail on the next. The `features` and `parity` jobs'
/// clippy steps are held to the same rule by [`clippy_step_drift`].
fn check_components_drift(yaml: &str) -> Result<(), String> {
  let steps = ci_steps(&ci_job_lines(yaml, CHECK_JOB));
  for (subcommand, component) in [("fmt", "rustfmt"), ("clippy", "clippy")] {
    let at = check_job_step_at(&steps, subcommand)?;
    if !installs_component(&steps[..at], component) {
      return Err(format!(
        "ci.yml `{CHECK_JOB}` job: no step ahead of the `cargo {subcommand}` step installs \
         `components: {component}`, so the job relies on the runner image for it"
      ));
    }
  }
  Ok(())
}

/// The `model-tests` job: one shard per model kit, each staging its kit's models
/// and then running independent checks over them.
const MODEL_TESTS_JOB: &str = "model-tests";

/// What every check after the staging step carries.
///
/// GitHub's default step condition is `success()`, so one red step marks every
/// step after it `skipped`, and `skipped` is silent: this job once ran for weeks
/// with gate steps that never executed. `!cancelled()` makes a check independent
/// of the ones above it; `steps.download.outcome != 'failure'` keeps the one real
/// dependency, since nothing can run without the staged artifacts.
const GATE_CONDITION: &str = "${{ !cancelled() && steps.download.outcome != 'failure' }}";

/// What the closing `Gate ledger` carries: `!cancelled()` alone, so it reports
/// even when the download died and took every check with it.
const LEDGER_CONDITION: &str = "${{ !cancelled() }}";

/// The checks every `model-tests` shard runs between its staging step and the
/// closing ledger, each by the start of its `name:`.
///
/// Every shard runs the same list, which is what lets one ledger and one set of
/// pins cover them all. A check taken out of it, or renamed, leaves a shard that
/// gates less while every guard above it still holds.
const MODEL_TESTS_CHECKS: [&str; 5] = [
  "Verify staged overlay ordering",
  "Verify staged artifact checksums",
  "fp16 graph sweep",
  "fp16 sweep inventory",
  "Model gates",
];

/// A step as a drift message names it: its `name:`, else its `uses:`, else the
/// first line of its `run`.
fn step_label(step: &[CiEntry]) -> String {
  ["name", "uses", "run"]
    .iter()
    .find_map(|key| ci_body(step, key).first())
    .cloned()
    .unwrap_or_else(|| "(an unnamed step)".to_string())
}

/// What, if anything, has drifted in the guards of the `model-tests` job, named.
/// In order: a step has `id: download`, the staging step every guard reads the
/// outcome of; the last step is the `Gate ledger`, carrying `if: ${{ !cancelled()
/// }}` alone; some step between the two runs `cargo test`; EVERY step between
/// them carries `if: ${{ !cancelled() && steps.download.outcome != 'failure' }}`;
/// each of them has an `id:` that the ledger reads as `=${{ steps.<id>.outcome
/// }}`; and the checks in [`MODEL_TESTS_CHECKS`] are among them.
///
/// Every step after staging is covered, not only those whose script says `cargo
/// test`: the checks are independent of each other, and a step that lost its
/// condition is skipped by a failure above it whatever it runs. The `cargo test`
/// clause is the vacuum guard — a job reshaped until no kit test is left in it
/// would satisfy every other clause.
///
/// The ledger is what reports a check that never ran, so a check whose id it does
/// not read is one nothing watches: ci.yml's own note calls forgetting the entry
/// the benign direction, and this reds it at authoring time. Only the ledger
/// step's own lines are read, so an entry that is commented out does not count.
fn model_tests_guard_drift(yaml: &str) -> Result<(), String> {
  let job = MODEL_TESTS_JOB;
  let steps = ci_steps(&ci_job_lines(yaml, job));
  let Some(staged) = steps
    .iter()
    .position(|step| ci_body(step, "id").first().map(String::as_str) == Some("download"))
  else {
    return Err(format!(
      "ci.yml `{job}` job: no step has `id: download` among {} parsed step(s) — every guard \
       reads the staging outcome through that id, and without it nothing short-circuits; the \
       staging step was renamed, dropped or commented out, or the job's `steps:` list changed \
       shape",
      steps.len()
    ));
  };
  let Some((ledger, checks)) = steps[staged + 1..].split_last() else {
    return Err(format!(
      "ci.yml `{job}` job: no step follows the staging step, so there is no `Gate ledger` and \
       no check to guard"
    ));
  };
  let is_the_ledger = ci_body(ledger, "name")
    .first()
    .is_some_and(|name| name.starts_with("Gate ledger"));
  if !is_the_ledger {
    return Err(format!(
      "ci.yml `{job}` job: the last step must be the `Gate ledger`, which is what reports a \
       check that never ran, found `{}`",
      step_label(ledger)
    ));
  }
  let condition = ci_body(ledger, "if");
  if condition.first().map(String::as_str) != Some(LEDGER_CONDITION) {
    return Err(format!(
      "ci.yml `{job}` job: the `Gate ledger` step must carry `if: {LEDGER_CONDITION}` alone, so \
       it reports even when the download died and took every check with it, found {condition:?}"
    ));
  }
  let runs_a_kit_test = checks.iter().any(|step| {
    ci_commands(step)
      .iter()
      .any(|command| runs_cargo(command, "test"))
  });
  if !runs_a_kit_test {
    return Err(format!(
      "ci.yml `{job}` job: no step between the staging step and the `Gate ledger` runs `cargo \
       test`, so the guards pin nothing about the kit's tests — they were dropped or commented \
       out, or the job's `steps:` list changed shape"
    ));
  }
  for step in checks {
    let condition = ci_body(step, "if");
    if condition.first().map(String::as_str) != Some(GATE_CONDITION) {
      return Err(format!(
        "ci.yml `{job}` job: the step `{}` must carry `if: {GATE_CONDITION}` — GitHub's default \
         condition is `success()`, so without it one red step above marks this check `skipped`, \
         and `skipped` is silent — found {condition:?}",
        step_label(step)
      ));
    }
  }
  let ledger_lines: Vec<&String> = ledger.iter().flat_map(|(_, body)| body).collect();
  for step in checks {
    let Some(id) = ci_body(step, "id").first() else {
      return Err(format!(
        "ci.yml `{job}` job: the step `{}` has no `id:`, so the `Gate ledger` step cannot report \
         whether it ran",
        step_label(step)
      ));
    };
    let entry = format!("=${{{{ steps.{id}.outcome }}}}");
    if !ledger_lines.iter().any(|line| line.contains(&entry)) {
      return Err(format!(
        "ci.yml `{job}` job: the `Gate ledger` step never reads step `{id}` (`{entry}`), so the \
         check `{}` could be skipped without the shard saying so",
        step_label(step)
      ));
    }
  }
  for name in MODEL_TESTS_CHECKS {
    let present = checks.iter().any(|step| {
      ci_body(step, "name")
        .first()
        .is_some_and(|label| label.starts_with(name))
    });
    if !present {
      return Err(format!(
        "ci.yml `{job}` job: no step between the staging step and the `Gate ledger` is named \
         `{name}`, which every shard runs — it was removed, renamed or commented out, or the \
         job's `steps:` list changed shape"
      ));
    }
  }
  Ok(())
}

/// Panic with a pin's drift message, if it has one, at the calling pin's line.
#[track_caller]
fn assert_no_drift(drift: Result<(), String>) {
  if let Err(drift) = drift {
    panic!("{drift}");
  }
}

/// Parse ONLY the "## Rename table" section of `FEATURE_MAP.md` into rows of
/// trimmed cells. Scoped to that section, so the separate curated-CI-combo table
/// lower in the doc cannot satisfy a rename-row assertion, and a bare token
/// elsewhere in the prose cannot stand in for a removed row.
fn rename_table_rows(doc: &str) -> Vec<Vec<String>> {
  let mut rows = Vec::new();
  let mut in_table = false;
  for line in doc.lines() {
    if let Some(heading) = line.strip_prefix("## ") {
      in_table = heading.contains("Rename table");
      continue;
    }
    if !in_table {
      continue;
    }
    let line = line.trim();
    if !line.starts_with('|') {
      continue;
    }
    let cells: Vec<String> = line
      .trim_matches('|')
      .split('|')
      .map(|c| c.trim().to_string())
      .collect();
    // Skip the header row and the `|---|---|` separator row.
    if cells.iter().any(|c| c == "Old crate") {
      continue;
    }
    if cells
      .iter()
      .all(|c| !c.is_empty() && c.chars().all(|ch| ch == '-'))
    {
      continue;
    }
    rows.push(cells);
  }
  rows
}

fn unbacktick(cell: &str) -> &str {
  cell.trim_matches('`')
}

/// `Cargo.toml` `[features]` names match the pinned flat set exactly — no
/// renamed, added, or dropped feature.
#[test]
fn feature_names_match_the_pinned_set() {
  let actual = feature_names(&features_block(&manifest()));
  let expected: BTreeSet<String> = expected_features()
    .iter()
    .map(|(name, _)| (*name).to_string())
    .collect();
  assert_eq!(
    actual, expected,
    "Cargo.toml [features] names drifted from the pinned flat feature set (FEATURE_MAP.md)"
  );
}

/// The oracle features are GONE from this crate — they belong to
/// `coremlit-parity` now. A re-added `speaker-oracle` here would drag the
/// unpublished `dia` git source back into the publishable manifest, which is the
/// exact thing this split exists to prevent, so it is pinned as an absence.
#[test]
fn oracle_features_are_not_this_crates() {
  let names = feature_names(&features_block(&manifest()));
  for moved in ["speaker-oracle", "clap-oracle", "vad-bundled"] {
    assert!(
      !names.contains(moved),
      "`{moved}` is declared on coremlit again — it belongs to coremlit-parity, whose \
       oracle deps (`dia`, `textclap`) are unpublished git sources cargo publish rejects"
    );
  }
  assert!(
    names.contains("align-oracle"),
    "`align-oracle` must STAY on coremlit: it only enables a feature of `asry`, a dependency \
     this crate has either way, so moving it would relocate code without removing a git dep"
  );
}

/// `coremlit-parity`'s `[features]` names match its pinned set exactly.
#[test]
fn parity_feature_names_match_the_pinned_set() {
  let Some(parity) = parity_manifest() else {
    return;
  };
  let actual = feature_names(&features_block(&parity));
  let expected: BTreeSet<String> = expected_parity_features()
    .iter()
    .map(|(name, _)| (*name).to_string())
    .collect();
  assert_eq!(
    actual, expected,
    "coremlit-parity [features] names drifted from the pinned oracle feature set"
  );
}

/// Each `coremlit-parity` feature's dependency set matches its pinned set —
/// same exact-set discipline, so an oracle that quietly stops enabling its
/// `coremlit` module feature (or starts enabling another kit's) reds.
#[test]
fn parity_feature_deps_are_pinned_with_no_cross_oracle_leakage() {
  let Some(parity) = parity_manifest() else {
    return;
  };
  let block = features_block(&parity);
  for (name, deps) in expected_parity_features() {
    let actual = feature_deps(&block, name);
    let expected: BTreeSet<String> = deps.iter().map(|d| (*d).to_string()).collect();
    assert_eq!(
      actual, expected,
      "coremlit-parity feature `{name}` dependency set drifted"
    );
  }
}

/// `coremlit-parity` must never be publishable: it exists to hold the two
/// unpublished git oracles, so `publish = false` is load-bearing, not tidiness.
#[test]
fn parity_crate_is_never_published() {
  let Some(parity) = parity_manifest() else {
    return;
  };
  assert!(
    parity.lines().any(|l| l.trim() == "publish = false"),
    "coremlit-parity must declare `publish = false` — it depends on unpublished git sources"
  );
}

/// Every former per-crate feature name is gone (renamed away by the table).
#[test]
fn old_per_crate_feature_names_are_gone() {
  let names = feature_names(&features_block(&manifest()));
  for old in ["dia", "dia-oracle", "parity-oracle", "vadkit", "bundled"] {
    assert!(
      !names.contains(old),
      "old per-crate feature `{old}` is still declared — the rename table maps it away"
    );
  }
}

/// Each feature's dependency set matches its pinned set exactly. Exact-set
/// equality catches cross-kit LEAKAGE (e.g. adding `"vad"` to `whisper` adds an
/// entry the pinned `whisper` set does not have) as well as a dropped
/// composition edge (e.g. `nl-recognizer` losing `whisper`).
#[test]
fn feature_deps_are_pinned_with_no_cross_kit_leakage() {
  let block = features_block(&manifest());
  for (name, deps) in expected_features() {
    let actual = feature_deps(&block, name);
    let expected: BTreeSet<String> = deps.iter().map(|d| (*d).to_string()).collect();
    assert_eq!(
      actual, expected,
      "feature `{name}` dependency set drifted (cross-kit leakage or a dropped/added dep)"
    );
  }
}

/// The `FEATURE_MAP.md` rename table maps every former kit's BARE crate to its
/// module feature. Parsed structurally (crate cell + `(crate)` cell + feature
/// cell), so removing or altering a bare-crate row reds even when the feature
/// token still appears elsewhere in the doc (the flat-feature list, a `dia`-style
/// feature row, or the prose).
#[test]
fn rename_table_pins_every_bare_crate_row() {
  let rows = rename_table_rows(&read_rel("FEATURE_MAP.md"));
  assert!(
    rows.len() >= BARE_CRATE_MAP.len(),
    "rename-table parse found only {} row(s) — the parser or the table shape broke",
    rows.len()
  );
  for (kit, feature) in BARE_CRATE_MAP {
    let found = rows
      .iter()
      .any(|r| r.len() >= 3 && r[0] == *kit && r[1] == "(crate)" && unbacktick(&r[2]) == *feature);
    assert!(
      found,
      "FEATURE_MAP.md rename table must map bare crate `{kit}` | (crate) | `{feature}` \
       (a removed or altered bare-crate row)"
    );
  }
}

/// Assert two combo sets are equal, naming the symmetric difference so a drift
/// reports exactly what moved (a `""` member prints as an empty string).
fn assert_combo_sets_eq(actual: &BTreeSet<String>, expected: &BTreeSet<String>, what: &str) {
  let missing: Vec<&String> = expected.difference(actual).collect();
  let unexpected: Vec<&String> = actual.difference(expected).collect();
  assert!(
    missing.is_empty() && unexpected.is_empty(),
    "{what} drifted from the pinned curated set — missing (pinned but not in the \
     active matrix): {missing:?}; unexpected (in the active matrix but not pinned): {unexpected:?}"
  );
}

/// `ci.yml`'s ACTIVE feature matrix is EXACTLY the curated combo set. The parsed
/// `matrix.features` set is compared for exact equality with `INTENDED_CI_COMBOS`
/// (not substring containment), so removing a combo, commenting one out, or
/// adding an unexpected one all red — the bare-core `""` included.
#[test]
fn ci_pins_the_curated_feature_combos() {
  let Some(ci) = ci_yml() else {
    return;
  };
  assert_combo_sets_eq(
    &ci_feature_combos(&ci, "features"),
    &intended_ci_combos(),
    "ci.yml `features` job matrix",
  );
}

/// The same exact-set pin for ci.yml's `parity` job — the only place the three
/// third-party oracles are still built in CI now that they are not on this
/// crate's feature matrix. Dropping a row here would silently stop building an
/// oracle; the job scoping is what keeps this set distinct from the one above.
#[test]
fn ci_pins_the_curated_parity_combos() {
  let Some(ci) = ci_yml() else {
    return;
  };
  assert_combo_sets_eq(
    &ci_feature_combos(&ci, "parity"),
    &owned(INTENDED_PARITY_CI_COMBOS),
    "ci.yml `parity` job matrix",
  );
}

/// The pins above say which combos run; this one says what each row's lint step
/// runs. `ci.yml`'s `features` job must keep the clippy step that lints every
/// combo as itself over every target (#158): `cargo clippy -p coremlit --no-deps
/// --all-targets [--features <combo>] -- -D warnings` on both of its arms,
/// guarded by `if: ${{ !cancelled() }}`, with the `clippy` component installed
/// ahead of it. A dropped step, `--all-targets` or `-D warnings` leaves every row
/// green while it lints less, so each reds here, naming the job and what dropped.
#[test]
fn ci_pins_the_features_job_clippy_step() {
  let Some(ci) = ci_yml() else {
    return;
  };
  assert_clippy_step(&ci, &FEATURES_CLIPPY);
}

/// The same pin for the `parity` job's step, which lints `coremlit-parity` under
/// each oracle feature. `--all-targets` is what makes it lint anything: that
/// package's library is an empty stub and all of its code is test targets.
#[test]
fn ci_pins_the_parity_job_clippy_step() {
  let Some(ci) = ci_yml() else {
    return;
  };
  assert_clippy_step(&ci, &PARITY_CLIPPY);
}

/// The `check` job's clippy step is the one lint pass over the whole workspace
/// with every feature on, and the one that compiles the `harness = false`
/// benches: `cargo clippy --all-targets --all-features -- -D warnings`, with no
/// `-p`. A dropped flag or a narrowed package leaves the job green while it
/// lints less, so each reds here, naming the job and what dropped.
#[test]
fn ci_pins_the_check_job_clippy_step() {
  let Some(ci) = ci_yml() else {
    return;
  };
  assert_no_drift(check_clippy_drift(&ci));
}

/// The `check` job's doc step: `cargo doc --no-deps --all-features` under
/// `RUSTDOCFLAGS: -D warnings`, the one place a broken intra-doc link turns the
/// job red.
#[test]
fn ci_pins_the_check_job_doc_step() {
  let Some(ci) = ci_yml() else {
    return;
  };
  assert_no_drift(check_doc_drift(&ci));
}

/// The `check` job installs the components of the two tools it runs, `rustfmt`
/// for `cargo fmt` and `clippy` for `cargo clippy`, ahead of the steps that run
/// them, instead of leaving both to the runner image.
#[test]
fn ci_pins_the_check_job_components() {
  let Some(ci) = ci_yml() else {
    return;
  };
  assert_no_drift(check_components_drift(&ci));
}

/// Every `model-tests` step after staging carries [`GATE_CONDITION`] and the
/// closing `Gate ledger` carries [`LEDGER_CONDITION`] alone. Without them one
/// red step marks every check after it `skipped`, which the job's own comment
/// records as a month in which four gate steps never ran. This is the structural
/// read of the job: a condition commented out of a step, or an unguarded step
/// added after the staging step, reds here by that step's name.
///
/// It also holds the ledger to its job: every check has an `id:` the ledger
/// reads, and the checks every shard runs are all there. All of it is read from
/// the parsed steps, so a commented-out `if:` or ledger entry does not count.
#[test]
fn ci_pins_the_model_tests_step_guards() {
  let Some(ci) = ci_yml() else {
    return;
  };
  assert_no_drift(model_tests_guard_drift(&ci));
}

/// Outside the repository a pin reads nothing and says so by name.
#[test]
fn a_repo_file_is_skipped_by_name_outside_the_source_tree() {
  assert_eq!(repo_file(None, ".github/workflows/ci.yml"), None);
  assert_eq!(
    skip_notice(".github/workflows/ci.yml"),
    "feature_map: skipped — .github/workflows/ci.yml is not in this source tree (a published \
     tarball?)"
  );
}

/// With a root to address it from, a repository file is read.
#[test]
fn a_repo_file_is_read_from_the_root_it_is_given() {
  let root = Path::new(env!("CARGO_MANIFEST_DIR"));
  let text = repo_file(Some(root), "Cargo.toml").expect("a root yields the file");
  assert!(text.contains("[package]"), "read the wrong file: {text}");
}

/// A root that IS there makes a missing file a failure, not a skip: the tarball
/// is the one place these files may be absent, and a deleted workflow must red
/// its pins instead of quietly ending them.
#[test]
#[should_panic(expected = "no-such-file.yml")]
fn a_repo_file_missing_inside_the_source_tree_is_a_failure() {
  let root = Path::new(env!("CARGO_MANIFEST_DIR"));
  let _ = repo_file(Some(root), "no-such-file.yml");
}

/// A well-formed TWO-JOB snippet whose active `features:` lists are exactly the
/// intended sets — the fixture the mutation cases below perturb. Its surrounding
/// keys (a preceding job carrying a `cargo build --features` step, the `features`
/// JOB name above `matrix:`, the `steps:` dedent below, and a SECOND job with its
/// own `matrix.features`) prove the parser enters at the right `features:`, stops
/// at the dedent, and does not fold one job's rows into the other's.
const DOCTORED_MATRIX: &str = r#"
jobs:
  check:
    runs-on: macos-15
    steps:
      - run: cargo build --features whisper --examples
  features:
    runs-on: macos-15
    strategy:
      fail-fast: false
      matrix:
        features:
          - ""
          - "whisper"
          - "align"
          - "speaker"
          - "speaker,serde"
          - "vad"
          - "whisper,vad"
          - "align-oracle"
          - "clap"
          - "granite"
          - "siglip"
          - "ced"
          - "lid"
          - "identity"
          - "identity,speaker"
          - "face"
          - "commercial-face-arcface"
          - "whisper,align,speaker,vad,clap,granite,siglip,ced,lid,identity,face,serde,tracing,nl-recognizer"
          - "whisper,align-oracle,speaker,vad,clap,granite,siglip,ced,lid,identity,face,serde,tracing,nl-recognizer"
    steps:
      - uses: actions/checkout@v7
  parity:
    runs-on: macos-15
    strategy:
      fail-fast: false
      matrix:
        features:
          - "speaker-oracle"
          - "clap-oracle"
          - "vad-bundled"
          - "speaker-oracle,clap-oracle,vad-bundled"
    steps:
      - uses: actions/checkout@v7
"#;

/// The parser reads the intended set from the well-formed fixture — a guard on
/// the mutation cases below (each perturbs this same fixture).
#[test]
fn ci_combo_parser_reads_the_wellformed_matrix() {
  assert_combo_sets_eq(
    &ci_feature_combos(DOCTORED_MATRIX, "features"),
    &intended_ci_combos(),
    "well-formed doctored matrix",
  );
}

/// Job scoping is real, not incidental: the same fixture yields the PARITY set
/// for the `parity` job, and neither job's rows appear in the other's set. A
/// parser that merged every `matrix.features` in the file would fail both.
#[test]
fn ci_combo_parser_scopes_each_job_separately() {
  let features = ci_feature_combos(DOCTORED_MATRIX, "features");
  let parity = ci_feature_combos(DOCTORED_MATRIX, "parity");
  assert_combo_sets_eq(
    &parity,
    &owned(INTENDED_PARITY_CI_COMBOS),
    "well-formed doctored parity matrix",
  );
  assert!(
    features.is_disjoint(&parity),
    "the two jobs' parsed combo sets bled into each other: {features:?} vs {parity:?}"
  );
}

/// An unknown job name parses to the EMPTY set rather than silently falling
/// back to the first matrix in the file — otherwise a renamed job would keep
/// passing its pin against another job's rows.
#[test]
fn ci_combo_parser_returns_empty_for_an_absent_job() {
  assert!(
    ci_feature_combos(DOCTORED_MATRIX, "no-such-job").is_empty(),
    "an absent job must parse to no combos at all"
  );
}

/// Deleting the bare-core `- ""` entry drops it from the parsed set — the
/// set-equality check must red (the R2 gap: the bare-core run was unpinned).
#[test]
fn ci_combo_check_reds_when_bare_core_is_deleted() {
  let doctored = DOCTORED_MATRIX.replace("          - \"\"\n", "");
  assert_ne!(
    ci_feature_combos(&doctored, "features"),
    intended_ci_combos(),
    "deleting the bare-core `- \"\"` entry must make the parsed set differ from the pinned set"
  );
}

/// Commenting out the bare-core `- ""` entry (`# - ""`) must red: comment lines
/// are skipped (not counted), so the parsed set loses `""` — this is what the
/// old substring `.contains()` check silently ACCEPTED.
#[test]
fn ci_combo_check_reds_when_bare_core_is_commented_out() {
  let doctored = DOCTORED_MATRIX.replace("          - \"\"\n", "          # - \"\"\n");
  assert_ne!(
    ci_feature_combos(&doctored, "features"),
    intended_ci_combos(),
    "commenting out the bare-core `- \"\"` entry must make the parsed set differ from the pinned set"
  );
}

/// Dropping any other curated combo (`whisper,vad`) must red as well.
#[test]
fn ci_combo_check_reds_when_a_combo_is_dropped() {
  let doctored = DOCTORED_MATRIX.replace("          - \"whisper,vad\"\n", "");
  assert_ne!(
    ci_feature_combos(&doctored, "features"),
    intended_ci_combos(),
    "dropping the `whisper,vad` combo must make the parsed set differ from the pinned set"
  );
}

/// A well-formed three-job snippet carrying the clippy steps as ci.yml spells
/// them — the fixture the clippy mutation cases below perturb. Around the two
/// real steps sit what a loose reader would take for them: a `check` job whose
/// own `cargo clippy` line must not stand in for a `features` job that lost its
/// step, and step names that spell the flags the `run` lines might drop.
const DOCTORED_CLIPPY_JOBS: &str = r#"
jobs:
  check:
    runs-on: macos-15
    steps:
      - uses: actions/checkout@v7
      - run: cargo clippy --all-targets --all-features -- -D warnings
  features:
    runs-on: macos-15
    steps:
      - uses: actions/checkout@v7
      - uses: dtolnay/rust-toolchain@stable
        with:
          components: clippy
      - name: "cargo test -p coremlit (features: ${{ matrix.features }})"
        run: |
          if [ -z "${{ matrix.features }}" ]; then
            cargo test -p coremlit
          else
            cargo test -p coremlit --features "${{ matrix.features }}"
          fi
      - name: "cargo clippy -p coremlit --all-targets (features: ${{ matrix.features }})"
        if: ${{ !cancelled() }}
        run: |
          if [ -z "${{ matrix.features }}" ]; then
            cargo clippy -p coremlit --no-deps --all-targets -- -D warnings
          else
            cargo clippy -p coremlit --no-deps --all-targets --features "${{ matrix.features }}" -- -D warnings
          fi
  parity:
    runs-on: macos-15
    steps:
      - uses: actions/checkout@v7
      - uses: dtolnay/rust-toolchain@stable
        with:
          components: clippy
      - name: "cargo test -p coremlit-parity (features: ${{ matrix.features }})"
        run: cargo test -p coremlit-parity --features "${{ matrix.features }}"
      - name: "cargo clippy -p coremlit-parity --all-targets (features: ${{ matrix.features }})"
        if: ${{ !cancelled() }}
        run: cargo clippy -p coremlit-parity --no-deps --all-targets --features "${{ matrix.features }}" -- -D warnings
"#;

/// The clippy pin reads the well-formed fixture clean — a guard on the mutation
/// cases below, each of which perturbs this same fixture.
#[test]
fn ci_clippy_check_reads_the_wellformed_steps() {
  assert_clippy_step(DOCTORED_CLIPPY_JOBS, &FEATURES_CLIPPY);
  assert_clippy_step(DOCTORED_CLIPPY_JOBS, &PARITY_CLIPPY);
}

/// Each perturbation of the fixture reds the pin of the job it lands in, and the
/// message names that job and what dropped. A case replaces the FIRST match, so a
/// `features`-job case that landed in `parity` instead leaves its pin green and
/// fails here. Every `--all-targets` case leaves the step's `name:` spelling the
/// flag: the `run` lines are what must carry it.
#[test]
fn ci_clippy_check_reds_on_each_dropped_piece() {
  struct Mutation {
    what: &'static str,
    from: &'static str,
    to: &'static str,
    pin: &'static IntendedClippy,
    names: &'static [&'static str],
  }
  let mutations = [
    Mutation {
      what: "the empty arm drops --all-targets",
      from: "cargo clippy -p coremlit --no-deps --all-targets -- -D warnings",
      to: "cargo clippy -p coremlit --no-deps -- -D warnings",
      pin: &FEATURES_CLIPPY,
      names: &["`features`", "the empty-features arm", "`--all-targets`"],
    },
    Mutation {
      what: "the features arm drops --all-targets",
      from: "cargo clippy -p coremlit --no-deps --all-targets --features",
      to: "cargo clippy -p coremlit --no-deps --features",
      pin: &FEATURES_CLIPPY,
      names: &["`features`", "the `--features` arm", "`--all-targets`"],
    },
    Mutation {
      what: "the features arm drops -D warnings",
      from: "--all-targets --features \"${{ matrix.features }}\" -- -D warnings",
      to: "--all-targets --features \"${{ matrix.features }}\"",
      pin: &FEATURES_CLIPPY,
      names: &["`features`", "the `--features` arm", "`-- -D warnings`"],
    },
    Mutation {
      what: "the empty arm drops --no-deps",
      from: "cargo clippy -p coremlit --no-deps --all-targets -- -D warnings",
      to: "cargo clippy -p coremlit --all-targets -- -D warnings",
      pin: &FEATURES_CLIPPY,
      names: &["`features`", "the empty-features arm", "`--no-deps`"],
    },
    Mutation {
      what: "a trailing comment stands in for --all-targets",
      from: "cargo clippy -p coremlit --no-deps --all-targets -- -D warnings",
      to: "cargo clippy -p coremlit --no-deps -- -D warnings # --all-targets",
      pin: &FEATURES_CLIPPY,
      names: &["`features`", "`--all-targets`"],
    },
    Mutation {
      what: "the features arm stops reading the matrix row",
      from: "--all-targets --features \"${{ matrix.features }}\" -- -D warnings",
      to: "--all-targets -- -D warnings",
      pin: &FEATURES_CLIPPY,
      names: &["`features`", "--features \"${{ matrix.features }}\""],
    },
    Mutation {
      what: "the empty arm is gone",
      from: "            cargo clippy -p coremlit --no-deps --all-targets -- -D warnings\n",
      to: "",
      pin: &FEATURES_CLIPPY,
      names: &["`features`", "without `--features`"],
    },
    Mutation {
      what: "the parity step drops --all-targets",
      from: "-p coremlit-parity --no-deps --all-targets",
      to: "-p coremlit-parity --no-deps",
      pin: &PARITY_CLIPPY,
      names: &["`parity`", "`--all-targets`"],
    },
    Mutation {
      what: "the parity step lints the wrong package",
      from: "run: cargo clippy -p coremlit-parity",
      to: "run: cargo clippy -p coremlit",
      pin: &PARITY_CLIPPY,
      names: &["`parity`", "`cargo clippy -p coremlit-parity`"],
    },
    Mutation {
      what: "the features step loses its condition",
      from: "        if: ${{ !cancelled() }}\n",
      to: "",
      pin: &FEATURES_CLIPPY,
      names: &["`features`", "`if: ${{ !cancelled() }}`"],
    },
    Mutation {
      what: "the parity step's condition becomes always()",
      from: "        if: ${{ !cancelled() }}\n        run: cargo clippy",
      to: "        if: ${{ always() }}\n        run: cargo clippy",
      pin: &PARITY_CLIPPY,
      names: &["`parity`", "`if: ${{ !cancelled() }}`"],
    },
    Mutation {
      what: "the features job stops installing clippy",
      from: "components: clippy",
      to: "components: rustfmt",
      pin: &FEATURES_CLIPPY,
      names: &["`features`", "`components: clippy`"],
    },
  ];
  for case in mutations {
    let doctored = DOCTORED_CLIPPY_JOBS.replacen(case.from, case.to, 1);
    assert_ne!(
      doctored, DOCTORED_CLIPPY_JOBS,
      "{}: the case matched nothing in the fixture",
      case.what
    );
    let Err(drift) = clippy_step_drift(&doctored, case.pin) else {
      panic!(
        "{}: the pin still reads the doctored ci.yml as intended",
        case.what
      );
    };
    for name in case.names {
      assert!(
        drift.contains(name),
        "{}: the message must name {name}, got: {drift}",
        case.what
      );
    }
  }
}

/// A job that lost its clippy step reds although the `check` job above it and the
/// `parity` job below it still run `cargo clippy`: the pin is scoped to its own
/// job, as the matrix pins are, and the other job's pin does not notice.
#[test]
fn ci_clippy_check_reds_when_a_job_loses_its_step() {
  let (head, tail) = DOCTORED_CLIPPY_JOBS
    .split_once("      - name: \"cargo clippy -p coremlit --all-targets")
    .expect("the fixture's `features` clippy step");
  let (_, parity) = tail
    .split_once("  parity:")
    .expect("the fixture's `parity` job");
  let doctored = format!("{head}  parity:{parity}");
  let Err(drift) = clippy_step_drift(&doctored, &FEATURES_CLIPPY) else {
    panic!("a `features` job without a clippy step still passed its pin");
  };
  assert!(
    drift.contains("`features`") && drift.contains("exactly one step running `cargo clippy`"),
    "the message must name the job and the missing step, got: {drift}"
  );
  assert_clippy_step(&doctored, &PARITY_CLIPPY);
}

/// Commenting the clippy commands out removes them: a comment line is not a
/// command, as a commented-out matrix entry is not a combo.
#[test]
fn ci_clippy_check_reds_when_the_commands_are_commented_out() {
  let doctored = DOCTORED_CLIPPY_JOBS
    .lines()
    .map(|line| {
      if line.contains("cargo clippy -p coremlit --no-deps") {
        format!("# {line}")
      } else {
        line.to_string()
      }
    })
    .collect::<Vec<_>>()
    .join("\n");
  assert_ne!(doctored, DOCTORED_CLIPPY_JOBS);
  let Err(drift) = clippy_step_drift(&doctored, &FEATURES_CLIPPY) else {
    panic!("a `features` job with its clippy commands commented out still passed its pin");
  };
  assert!(
    drift.contains("`features`") && drift.contains("exactly one step running `cargo clippy`"),
    "the message must name the job and the missing step, got: {drift}"
  );
}

/// A re-wrapped command is the same command: a backslash-continued `cargo clippy`
/// reads as one line, so reformatting the step alone does not red the pin.
#[test]
fn ci_clippy_check_reads_a_rewrapped_command() {
  let doctored = DOCTORED_CLIPPY_JOBS.replacen(
    "--no-deps --all-targets --features \"${{ matrix.features }}\" -- -D warnings",
    "--no-deps --all-targets \\\n              --features \"${{ matrix.features }}\" -- -D warnings",
    1,
  );
  assert_ne!(doctored, DOCTORED_CLIPPY_JOBS);
  assert_clippy_step(&doctored, &FEATURES_CLIPPY);
}

/// `fixture` with the first `from` at or after the `  <job>:` key replaced by
/// `to`, so a case lands in the job it names and not in a lookalike ahead of it.
fn doctored_in(fixture: &str, job: &str, from: &str, to: &str) -> String {
  let key = format!("\n  {job}:\n");
  let at = fixture
    .find(&key)
    .unwrap_or_else(|| panic!("the fixture has no `{job}` job"));
  let (head, tail) = fixture.split_at(at);
  format!("{head}{}", tail.replacen(from, to, 1))
}

/// A perturbed fixture and the pieces the message of the pin that reds it must
/// name.
struct Doctored {
  what: &'static str,
  yaml: String,
  names: &'static [&'static str],
}

/// Require every doctored fixture to red `pin`, its message naming each piece.
/// A case that changed nothing in the fixture, or that landed in a lookalike job
/// and left the pin green, fails here instead of passing for a pin that works.
fn assert_each_drift_reds(fixture: &str, cases: &[Doctored], pin: fn(&str) -> Result<(), String>) {
  for case in cases {
    assert_ne!(
      case.yaml, fixture,
      "{}: the case matched nothing in the fixture",
      case.what
    );
    let Err(drift) = pin(&case.yaml) else {
      panic!(
        "{}: the pin still reads the doctored ci.yml as intended",
        case.what
      );
    };
    for name in case.names {
      assert!(
        drift.contains(name),
        "{}: the message must name {name}, got: {drift}",
        case.what
      );
    }
  }
}

/// A well-formed `check` job as ci.yml spells it, between two jobs that carry
/// the same toolchain, format, lint and doc steps: the fixture the mutation
/// cases below perturb. A `check` job that lost a step must not read as intact
/// for what a neighbour still runs, so each case lands in `check` and the pin
/// has to red there.
const DOCTORED_CHECK_JOBS: &str = r#"
jobs:
  docs:
    runs-on: macos-15
    steps:
      - uses: dtolnay/rust-toolchain@stable
        with:
          components: clippy, rustfmt
      - run: cargo fmt --all --check
      - run: cargo clippy --all-targets --all-features -- -D warnings
      - run: cargo doc --no-deps --all-features
        env:
          RUSTDOCFLAGS: -D warnings
  check:
    runs-on: macos-15
    steps:
      - uses: actions/checkout@v7
      - uses: dtolnay/rust-toolchain@stable
        with:
          components: clippy, rustfmt
      - run: cargo fmt --all --check
      - run: cargo clippy --all-targets --all-features -- -D warnings
      - run: cargo doc --no-deps --all-features
        env:
          RUSTDOCFLAGS: -D warnings
      - run: cargo build --features whisper --examples
  features:
    runs-on: macos-15
    steps:
      - run: cargo clippy --all-targets --all-features -- -D warnings
"#;

/// The `check` fixture's clippy step as one `run` line.
const CHECK_CLIPPY_STEP: &str =
  "      - run: cargo clippy --all-targets --all-features -- -D warnings\n";

/// The `check` fixture's doc step, its `env:` included.
const CHECK_DOC_STEP: &str = "      - run: cargo doc --no-deps --all-features\n        env:\n          RUSTDOCFLAGS: -D warnings\n";

/// The pins read the well-formed fixture clean — a guard on the mutation cases
/// below, each of which perturbs this same fixture.
#[test]
fn ci_check_job_pins_read_the_wellformed_steps() {
  assert_no_drift(check_clippy_drift(DOCTORED_CHECK_JOBS));
  assert_no_drift(check_doc_drift(DOCTORED_CHECK_JOBS));
  assert_no_drift(check_components_drift(DOCTORED_CHECK_JOBS));
}

/// Each perturbation of the `check` job's clippy step reds its pin, and the
/// message names the job and what dropped.
#[test]
fn ci_check_clippy_pin_reds_on_each_dropped_piece() {
  let in_check = |from: &str, to: &str| doctored_in(DOCTORED_CHECK_JOBS, "check", from, to);
  let cases = [
    Doctored {
      what: "the clippy step is gone",
      yaml: in_check(CHECK_CLIPPY_STEP, ""),
      names: &["`check`", "exactly one step running `cargo clippy`"],
    },
    Doctored {
      what: "the clippy step is commented out",
      yaml: in_check(
        CHECK_CLIPPY_STEP,
        "      # - run: cargo clippy --all-targets --all-features -- -D warnings\n",
      ),
      names: &["`check`", "exactly one step running `cargo clippy`"],
    },
    Doctored {
      what: "a second clippy step is added",
      yaml: in_check(
        "      - run: cargo build",
        "      - run: cargo clippy --all-targets --all-features -- -D warnings\n      - run: cargo build",
      ),
      names: &[
        "`check`",
        "exactly one step running `cargo clippy`",
        "found 2",
      ],
    },
    Doctored {
      what: "the step drops --all-targets",
      yaml: in_check(
        "cargo clippy --all-targets --all-features",
        "cargo clippy --all-features",
      ),
      names: &["`check`", "`--all-targets`"],
    },
    Doctored {
      what: "the step drops --all-features",
      yaml: in_check(
        "cargo clippy --all-targets --all-features",
        "cargo clippy --all-targets",
      ),
      names: &["`check`", "`--all-features`"],
    },
    Doctored {
      what: "the step drops -D warnings",
      yaml: in_check("--all-features -- -D warnings", "--all-features"),
      names: &["`check`", "`-- -D warnings`"],
    },
    Doctored {
      what: "the step narrows to one package",
      yaml: in_check(
        "cargo clippy --all-targets",
        "cargo clippy -p coremlit --all-targets",
      ),
      names: &["`check`", "the whole workspace", "names a package"],
    },
    Doctored {
      what: "a trailing comment stands in for --all-features",
      yaml: in_check(
        "cargo clippy --all-targets --all-features -- -D warnings",
        "cargo clippy --all-targets -- -D warnings # --all-features",
      ),
      names: &["`check`", "`--all-features`"],
    },
    Doctored {
      what: "the step's name spells the flag its run line dropped",
      yaml: in_check(
        CHECK_CLIPPY_STEP,
        "      - name: cargo clippy --all-targets --all-features -- -D warnings\n        run: cargo clippy --all-targets -- -D warnings\n",
      ),
      names: &["`check`", "`--all-features`"],
    },
  ];
  assert_each_drift_reds(DOCTORED_CHECK_JOBS, &cases, check_clippy_drift);
}

/// Each perturbation of the `check` job's doc step reds its pin, and the
/// message names the job and what dropped.
#[test]
fn ci_check_doc_pin_reds_on_each_dropped_piece() {
  let in_check = |from: &str, to: &str| doctored_in(DOCTORED_CHECK_JOBS, "check", from, to);
  let cases = [
    Doctored {
      what: "the doc step is gone",
      yaml: in_check(CHECK_DOC_STEP, ""),
      names: &["`check`", "exactly one step running `cargo doc`"],
    },
    Doctored {
      what: "the doc step is commented out",
      yaml: in_check(
        "      - run: cargo doc --no-deps --all-features\n",
        "      # - run: cargo doc --no-deps --all-features\n",
      ),
      names: &["`check`", "exactly one step running `cargo doc`"],
    },
    Doctored {
      what: "a second doc step is added",
      yaml: in_check(
        "      - run: cargo build",
        &format!("{CHECK_DOC_STEP}      - run: cargo build"),
      ),
      names: &["`check`", "exactly one step running `cargo doc`", "found 2"],
    },
    Doctored {
      what: "the step drops --no-deps",
      yaml: in_check(
        "cargo doc --no-deps --all-features",
        "cargo doc --all-features",
      ),
      names: &["`check`", "`--no-deps`"],
    },
    Doctored {
      what: "the step drops --all-features",
      yaml: in_check("cargo doc --no-deps --all-features", "cargo doc --no-deps"),
      names: &["`check`", "`--all-features`"],
    },
    Doctored {
      what: "RUSTDOCFLAGS is gone",
      yaml: in_check("        env:\n          RUSTDOCFLAGS: -D warnings\n", ""),
      names: &["`check`", "`RUSTDOCFLAGS: -D warnings`"],
    },
    Doctored {
      what: "RUSTDOCFLAGS stops denying warnings",
      yaml: in_check("RUSTDOCFLAGS: -D warnings", "RUSTDOCFLAGS: --cfg docsrs"),
      names: &["`check`", "`RUSTDOCFLAGS: -D warnings`"],
    },
    Doctored {
      what: "a trailing comment stands in for -D warnings",
      yaml: in_check(
        "RUSTDOCFLAGS: -D warnings",
        "RUSTDOCFLAGS: --cfg docsrs # -D warnings",
      ),
      names: &["`check`", "`RUSTDOCFLAGS: -D warnings`"],
    },
  ];
  assert_each_drift_reds(DOCTORED_CHECK_JOBS, &cases, check_doc_drift);
}

/// A re-spelled flag is the same flag: `RUSTDOCFLAGS` quoted, carrying other
/// flags, and denying warnings as `-Dwarnings` still reads as the pin intends.
#[test]
fn ci_check_doc_pin_reads_a_respelled_flag() {
  let doctored = doctored_in(
    DOCTORED_CHECK_JOBS,
    "check",
    "RUSTDOCFLAGS: -D warnings",
    "RUSTDOCFLAGS: \"--cfg docsrs -Dwarnings\"",
  );
  assert_ne!(doctored, DOCTORED_CHECK_JOBS);
  assert_no_drift(check_doc_drift(&doctored));
}

/// The `check` job's toolchain step, `with:` map included, as the fixture
/// spells it.
const CHECK_TOOLCHAIN_STEP: &str = "      - uses: dtolnay/rust-toolchain@stable\n        with:\n          components: clippy, rustfmt\n";

/// Each perturbation of the `check` job's toolchain install reds its pin, and the
/// message names the job and the component no step ahead of the tool installs.
#[test]
fn ci_check_components_pin_reds_on_each_dropped_piece() {
  let in_check = |from: &str, to: &str| doctored_in(DOCTORED_CHECK_JOBS, "check", from, to);
  let cases = [
    Doctored {
      what: "the toolchain step asks for no components",
      yaml: in_check("        with:\n          components: clippy, rustfmt\n", ""),
      names: &["`check`", "`cargo fmt`", "`components: rustfmt`"],
    },
    Doctored {
      what: "the toolchain step stops asking for rustfmt",
      yaml: in_check("components: clippy, rustfmt", "components: clippy"),
      names: &["`check`", "`cargo fmt`", "`components: rustfmt`"],
    },
    Doctored {
      what: "the toolchain step stops asking for clippy",
      yaml: in_check("components: clippy, rustfmt", "components: rustfmt"),
      names: &["`check`", "`cargo clippy`", "`components: clippy`"],
    },
    Doctored {
      what: "the components line is only a comment",
      yaml: in_check(
        "          components: clippy, rustfmt\n",
        "          # components: clippy, rustfmt\n",
      ),
      names: &["`check`", "`components: rustfmt`"],
    },
    Doctored {
      what: "a trailing comment stands in for a component",
      yaml: in_check(
        "components: clippy, rustfmt",
        "components: clippy # rustfmt",
      ),
      names: &["`check`", "`components: rustfmt`"],
    },
    Doctored {
      what: "another input names the component",
      yaml: in_check(
        "components: clippy, rustfmt",
        "toolchain: rustfmt\n          components: clippy",
      ),
      names: &["`check`", "`components: rustfmt`"],
    },
    Doctored {
      what: "the toolchain step comes after the steps that need it",
      yaml: in_check(
        &format!("{CHECK_TOOLCHAIN_STEP}      - run: cargo fmt --all --check\n"),
        &format!("      - run: cargo fmt --all --check\n{CHECK_TOOLCHAIN_STEP}"),
      ),
      names: &["`check`", "no step ahead of", "`components: rustfmt`"],
    },
    Doctored {
      what: "the toolchain step is commented out",
      yaml: in_check(
        CHECK_TOOLCHAIN_STEP,
        "      # - uses: dtolnay/rust-toolchain@stable\n      #   with:\n      #     components: clippy, rustfmt\n",
      ),
      names: &["`check`", "`components: rustfmt`"],
    },
    Doctored {
      what: "the fmt step is gone",
      yaml: in_check("      - run: cargo fmt --all --check\n", ""),
      names: &["`check`", "exactly one step running `cargo fmt`"],
    },
    Doctored {
      what: "the clippy step is gone",
      yaml: in_check(CHECK_CLIPPY_STEP, ""),
      names: &["`check`", "exactly one step running `cargo clippy`"],
    },
  ];
  assert_each_drift_reds(DOCTORED_CHECK_JOBS, &cases, check_components_drift);
}

/// A re-spelled list is the same list: a different order, no space after the
/// comma, and quoted entries all still read as the components the pin intends.
#[test]
fn ci_check_components_pin_reads_a_respelled_list() {
  for list in [
    "rustfmt, clippy",
    "clippy,rustfmt",
    "\"clippy, rustfmt\"",
    "'clippy', 'rustfmt'",
  ] {
    let doctored = doctored_in(
      DOCTORED_CHECK_JOBS,
      "check",
      "components: clippy, rustfmt",
      &format!("components: {list}"),
    );
    assert_ne!(doctored, DOCTORED_CHECK_JOBS);
    assert_no_drift(check_components_drift(&doctored));
  }
}

/// A well-formed `model-tests` job as ci.yml spells its shape, with a lookalike
/// job that is itself compliant ahead of it and a job with an unguarded test
/// step after it: the fixture the mutation cases below perturb. A step before the
/// staging step carries no condition, as in ci.yml.
const DOCTORED_MODEL_TESTS_JOBS: &str = r#"
jobs:
  staging:
    runs-on: macos-15
    steps:
      - name: Stage this kit's models
        id: download
        uses: ./.github/actions/stage-models
      - name: Model gates
        id: gates
        if: ${{ !cancelled() && steps.download.outcome != 'failure' }}
        run: cargo test -p coremlit -- --ignored
      - name: Gate ledger (a gate that never ran proves nothing)
        if: ${{ !cancelled() }}
        run: echo ledger
  model-tests:
    name: model-tests (${{ matrix.kit }})
    runs-on: macos-15
    strategy:
      fail-fast: false
      matrix:
        include:
          - kit: whisper
    steps:
      - uses: actions/checkout@v7
      - uses: dtolnay/rust-toolchain@stable
      - name: Stage this kit's models
        id: download
        uses: ./.github/actions/stage-models
        with:
          kit: ${{ matrix.kit }}
      - name: Verify staged overlay ordering
        id: overlay
        if: ${{ !cancelled() && steps.download.outcome != 'failure' }}
        run: ./check-overlay.sh
      - name: Verify staged artifact checksums
        id: checksums
        if: ${{ !cancelled() && steps.download.outcome != 'failure' }}
        run: |
          set -euo pipefail
          shasum -a 256 -c CHECKSUMS.sha256
      - name: fp16 graph sweep
        id: fp16_guards
        if: ${{ !cancelled() && steps.download.outcome != 'failure' }}
        run: cargo test -p coremlit --test fp16_guards
      - name: fp16 sweep inventory
        id: fp16_inventory
        if: ${{ !cancelled() && steps.download.outcome != 'failure' }}
        run: tests/fp16_sweep_inventory.sh
      - name: Model gates
        id: gates
        if: ${{ !cancelled() && steps.download.outcome != 'failure' }}
        env:
          KIT: ${{ matrix.kit }}
        run: |
          set -euo pipefail
          cargo test -p coremlit --features "$features" -- --ignored
      - name: Gate ledger (a gate that never ran proves nothing)
        if: ${{ !cancelled() }}
        env:
          LEDGER: |
            overlay-ordering=${{ steps.overlay.outcome }}
            artifact-checksums=${{ steps.checksums.outcome }}
            fp16-graph-sweep=${{ steps.fp16_guards.outcome }}
            fp16-sweep-inventory=${{ steps.fp16_inventory.outcome }}
            model-gates=${{ steps.gates.outcome }}
        run: |
          set -euo pipefail
          printf '%s\n' "$LEDGER"
  parity:
    runs-on: macos-15
    steps:
      - run: cargo test -p coremlit-parity
"#;

/// The condition line every `model-tests` check carries in the fixture.
const MODEL_TESTS_GUARD_LINE: &str =
  "        if: ${{ !cancelled() && steps.download.outcome != 'failure' }}\n";

/// The pin reads the well-formed fixture clean — a guard on the mutation cases
/// below, each of which perturbs this same fixture.
#[test]
fn ci_model_tests_pin_reads_the_wellformed_job() {
  assert_no_drift(model_tests_guard_drift(DOCTORED_MODEL_TESTS_JOBS));
}

/// Each perturbation of the `model-tests` job reds its pin, and the message names
/// the job and the step or piece that drifted. A case lands in `model-tests` and
/// not in the compliant job ahead of it, so a renamed staging id or a dropped
/// ledger reds although that job still has both.
#[test]
fn ci_model_tests_pin_reds_on_each_dropped_piece() {
  let in_job =
    |from: &str, to: &str| doctored_in(DOCTORED_MODEL_TESTS_JOBS, "model-tests", from, to);
  let guarded = |id: &str| format!("        id: {id}\n{MODEL_TESTS_GUARD_LINE}");
  let without_tests = doctored_in(
    &in_job(
      "run: cargo test -p coremlit --test fp16_guards",
      "run: echo fp16",
    ),
    "model-tests",
    "          cargo test -p coremlit --features \"$features\" -- --ignored\n",
    "          echo gates\n",
  );
  let inventory = format!(
    "      - name: fp16 sweep inventory\n        id: fp16_inventory\n{MODEL_TESTS_GUARD_LINE}        \
     run: tests/fp16_sweep_inventory.sh\n"
  );
  let ledger_entry = "            model-gates=${{ steps.gates.outcome }}\n";
  let cases = [
    Doctored {
      what: "the staging step's id is renamed",
      yaml: in_job("id: download", "id: staged"),
      names: &["`model-tests`", "`id: download`"],
    },
    Doctored {
      what: "a check loses its condition",
      yaml: in_job(&guarded("gates"), "        id: gates\n"),
      names: &[
        "`model-tests`",
        "`Model gates`",
        "steps.download.outcome != 'failure'",
      ],
    },
    Doctored {
      what: "a check's condition loses the staging clause",
      yaml: in_job(
        &guarded("fp16_guards"),
        "        id: fp16_guards\n        if: ${{ !cancelled() }}\n",
      ),
      names: &["`model-tests`", "`fp16 graph sweep`"],
    },
    Doctored {
      what: "a check's condition becomes always()",
      yaml: in_job(
        &guarded("checksums"),
        "        id: checksums\n        if: ${{ always() && steps.download.outcome != 'failure' }}\n",
      ),
      names: &["`model-tests`", "`Verify staged artifact checksums`"],
    },
    Doctored {
      what: "a check's condition is only a comment",
      yaml: in_job(
        &guarded("gates"),
        "        id: gates\n        # if: ${{ !cancelled() && steps.download.outcome != 'failure' }}\n",
      ),
      names: &["`model-tests`", "`Model gates`"],
    },
    Doctored {
      what: "a check is added without a condition",
      yaml: in_job(
        "      - name: Model gates\n",
        "      - run: cargo test -p coremlit --test newcheck\n      - name: Model gates\n",
      ),
      names: &["`model-tests`", "`cargo test -p coremlit --test newcheck`"],
    },
    Doctored {
      what: "the ledger loses its condition",
      yaml: in_job(
        "      - name: Gate ledger (a gate that never ran proves nothing)\n        if: ${{ !cancelled() }}\n",
        "      - name: Gate ledger (a gate that never ran proves nothing)\n",
      ),
      names: &["`model-tests`", "`Gate ledger`", "alone"],
    },
    Doctored {
      what: "the ledger waits on the download",
      yaml: in_job(
        "        if: ${{ !cancelled() }}\n        env:\n          LEDGER",
        "        if: ${{ !cancelled() && steps.download.outcome != 'failure' }}\n        env:\n          LEDGER",
      ),
      names: &["`model-tests`", "`Gate ledger`", "alone"],
    },
    Doctored {
      what: "a step is added after the ledger",
      yaml: in_job(
        "  parity:\n",
        "      - run: cargo test -p coremlit --test late\n  parity:\n",
      ),
      names: &["`model-tests`", "the last step", "`Gate ledger`"],
    },
    Doctored {
      what: "the ledger is no longer the Gate ledger",
      yaml: in_job(
        "Gate ledger (a gate that never ran proves nothing)",
        "Verdict summary",
      ),
      names: &["`model-tests`", "the last step", "`Gate ledger`"],
    },
    Doctored {
      what: "no kit test is left between staging and the ledger",
      yaml: without_tests,
      names: &["`model-tests`", "`cargo test`"],
    },
    Doctored {
      what: "a check loses its id",
      yaml: in_job(
        "      - name: Model gates\n        id: gates\n",
        "      - name: Model gates\n",
      ),
      names: &["`model-tests`", "`Model gates`", "`id:`"],
    },
    Doctored {
      what: "the ledger no longer reads a check",
      yaml: in_job(ledger_entry, ""),
      names: &["`model-tests`", "`Gate ledger`", "`gates`", "`Model gates`"],
    },
    Doctored {
      what: "the ledger's entry for a check is only a comment",
      yaml: in_job(
        ledger_entry,
        "            # model-gates=${{ steps.gates.outcome }}\n",
      ),
      names: &["`model-tests`", "`Gate ledger`", "`gates`"],
    },
    Doctored {
      what: "the ledger reads the check under another id",
      yaml: in_job(
        ledger_entry,
        "            model-gates=${{ steps.gate.outcome }}\n",
      ),
      names: &["`model-tests`", "`Gate ledger`", "`gates`"],
    },
    Doctored {
      what: "a check is dropped",
      yaml: in_job(&inventory, ""),
      names: &[
        "`model-tests`",
        "`fp16 sweep inventory`",
        "every shard runs",
      ],
    },
    Doctored {
      what: "a check is renamed",
      yaml: in_job("name: fp16 sweep inventory", "name: fp16 inventory sweep"),
      names: &["`model-tests`", "`fp16 sweep inventory`"],
    },
    Doctored {
      what: "a check is commented out",
      yaml: in_job(
        &inventory,
        "      # - name: fp16 sweep inventory\n      #   id: fp16_inventory\n      #   if: ${{ !cancelled() && steps.download.outcome != 'failure' }}\n      #   run: tests/fp16_sweep_inventory.sh\n",
      ),
      names: &["`model-tests`", "`fp16 sweep inventory`"],
    },
  ];
  assert_each_drift_reds(DOCTORED_MODEL_TESTS_JOBS, &cases, model_tests_guard_drift);
}

/// A job with no `model-tests` key parses to no steps and reds, rather than the
/// pin reading another job's steps in its place.
#[test]
fn ci_model_tests_pin_reds_when_the_job_is_absent() {
  let renamed = DOCTORED_MODEL_TESTS_JOBS.replacen("  model-tests:\n", "  model-shards:\n", 1);
  assert_ne!(renamed, DOCTORED_MODEL_TESTS_JOBS);
  let Err(drift) = model_tests_guard_drift(&renamed) else {
    panic!("a ci.yml with no `model-tests` job still passed its pin");
  };
  assert!(
    drift.contains("`model-tests`") && drift.contains("`id: download`"),
    "the message must name the job and the missing staging step, got: {drift}"
  );
}
