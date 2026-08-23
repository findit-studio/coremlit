//! Host-class scoping for MEASURED-then-pinned numeric bands — the sibling of
//! `tests/support/host_class.rs`, for numbers rather than goldens.
//!
//! # The two kinds of number a parity suite asserts
//!
//! A **spec contract** — `siglip`'s `PARITY_FLOOR = 0.99917` (spec §3) — is a
//! promise the port makes to its callers. It is portable by construction: it
//! says "the embedding is close enough to the reference to be useful", and that
//! must hold on every machine that ships this crate. It is asserted
//! unconditionally, everywhere, and it is never widened.
//!
//! A **characterized measurement** — "this machine's `CpuAndGpu` worst cosine
//! is 0.99999, so pin 0.9999" or "this machine's vision ANE collapses to
//! 0.31–0.41" — is a *description of one host*. Its whole content is an
//! observation of particular CPU kernels, a particular GPU driver and a
//! particular Neural Engine generation under a particular macOS build. CoreML
//! contracts none of those to be reproducible across hosts (#36), so asserting
//! such a number on a machine that did not produce it measures the machine, not
//! the port.
//!
//! [`BandGate`] binds each measured band to the host class it was characterized
//! on, and asserts it only there.
//!
//! # The three-way contract, and why it differs from a golden's
//!
//! | recorded [`CharacterizedHost`] | verdict | what the gate does |
//! | --- | --- | --- |
//! | present, equal | [`BandVerdict::Armed`] | asserts the band EXACTLY as before — full strictness on the machine the number describes |
//! | absent | [`BandVerdict::Unrecorded`] | measures, PRINTS, does not assert |
//! | present, different | [`BandVerdict::Foreign`] | measures, PRINTS with both host classes named, does not assert |
//!
//! `host_class.rs` makes the OPPOSITE call for its two non-`Match` cases, and
//! the asymmetry is deliberate:
//!
//! - An unstamped **golden** keeps its tight bounds (`HostVerdict::LegacyUnknown`)
//!   because a golden is committed data: a token mismatch against it is
//!   evidence of *something*, ambiguous between a port defect and host drift,
//!   and worth a red light plus an ambiguity note. Crucially it is also the
//!   ONLY gate on that comparison — dropping it leaves nothing.
//! - An unrecorded **band** is not committed data; it is a summary of a
//!   distribution observed on one machine. Off that machine it predicts
//!   nothing, and its failure direction is not even ordered: the CI runner
//!   measured vision ANE at 0.999664 — *better* than the characterized
//!   0.31–0.41 — and reds a band that was written to describe a broken ANE.
//!   That red is not evidence of a defect, it is evidence that the band
//!   describes a different computer.
//! - And unlike the golden case, dropping the band leaves the suite's portable
//!   spec floor still asserting on every host, so the ship contract is never
//!   the thing being skipped.
//!
//! A recorded-but-different **golden** panics before any CoreML number is
//! produced, because the whole test is that comparison. A [`BandVerdict::Foreign`]
//! band must NOT panic: panicking would hard-red every non-characterizing host,
//! which is the same defect in the other direction. It reports instead.
//!
//! # Nothing skips silently
//!
//! [`BandGate::open`] prints a banner naming the verdict, the running host
//! class, the reason, and the exact command that re-characterizes on this host;
//! every band is then COMPUTED and PRINTED on every host, pass or fail, whether
//! or not it is asserted. A CI log therefore always carries the numbers needed
//! to characterize that runner later — which is how an unrecorded band gets
//! recorded, rather than by guessing a plausible-looking host.
//!
//! # One host per band, for now
//!
//! [`CharacterizedHost`] is a single host class, not a set: exactly one machine
//! can be armed at a time, and arming a second is a source edit that moves
//! `CHARACTERIZED_ON`. That is enough while no band records a host at all, and
//! it keeps the recorded provenance unambiguous — a set would need per-host
//! bands anyway, since two machines that disagree do not share one band.
//!
//! # Widening is not the fix
//!
//! A band wide enough to admit two hosts' measurements asserts nothing. If a
//! band reds on the machine it was characterized on, that is a real finding. If
//! it reds elsewhere, re-measure there and record THAT host with ITS numbers —
//! never stretch one band over both.

use super::HostClass;

/// The host class a measured band was characterized on, as recorded in source
/// beside the band.
///
/// A `&'static str` mirror of [`HostClass`] (whose `String` fields cannot appear
/// in a `const`), so a suite records provenance as a plain constant:
///
/// ```ignore
/// const CHARACTERIZED_ON: Option<CharacterizedHost> = Some(CharacterizedHost {
///   os_build: "24F74",
///   os_product_version: "15.5",
///   chip: "Apple M4 Max",
///   arch: "arm64",
/// });
/// ```
///
/// `None` means the characterization host was never recorded. That is an
/// honest, load-bearing state, NOT a placeholder to fill with a guess: a
/// fabricated host is worse than an absent one, because it silently claims a
/// provenance nobody measured and hard-reds every machine that does not match
/// the fiction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CharacterizedHost {
  /// `kern.osversion` — the OS binary set every CPU kernel and ANE firmware
  /// revision ships in.
  pub os_build: &'static str,
  /// `kern.osproductversion` — human-readable, fully determined by the build.
  pub os_product_version: &'static str,
  /// `machdep.cpu.brand_string` — kernel dispatch and ANE fp16 units vary by
  /// microarchitecture.
  pub chip: &'static str,
  /// Process arch (`arm64` / `x86_64`), which governs the in-process `CpuOnly`
  /// kernels.
  pub arch: &'static str,
}

impl CharacterizedHost {
  /// Whether the RUNNING host is the class this band was characterized on. The
  /// same four-field identity `check_host_class` compares for goldens.
  pub fn matches(&self, running: &HostClass) -> bool {
    self.os_build == running.os_build
      && self.os_product_version == running.os_product_version
      && self.chip == running.chip
      && self.arch == running.arch
  }
}

impl std::fmt::Display for CharacterizedHost {
  /// Byte-identical formatting to [`HostClass`], so a mismatch diagnosis lines
  /// the two hosts up column-for-column.
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(
      f,
      "macOS {} (build {}), {}, {}",
      self.os_product_version, self.os_build, self.chip, self.arch
    )
  }
}

/// Verdict of the band host-class gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BandVerdict {
  /// Recorded and equal to the running host: assert every band exactly as
  /// written. This is the only verdict under which a band can red.
  Armed,
  /// No characterization host recorded in source: measure and report only.
  Unrecorded,
  /// Recorded, but a DIFFERENT host class: measure and report only, naming both
  /// hosts. Deliberately not a panic — see the module docs.
  Foreign,
}

impl BandVerdict {
  /// Whether bands under this verdict are asserted.
  pub fn asserts(self) -> bool {
    self == BandVerdict::Armed
  }
}

/// THE band host-class predicate. Pure — no I/O — so hermetic tests drive it
/// with synthetic host classes.
pub fn band_verdict(recorded: Option<&CharacterizedHost>, running: &HostClass) -> BandVerdict {
  match recorded {
    None => BandVerdict::Unrecorded,
    Some(r) if r.matches(running) => BandVerdict::Armed,
    Some(_) => BandVerdict::Foreign,
  }
}

/// One suite's measured-band gate: the verdict, the hosts involved, and the
/// re-characterization command quoted into every message.
///
/// Open it ONCE at the top of a test — before any measurement — then route
/// every measured band through [`check_floor`](Self::check_floor),
/// [`check_ceiling`](Self::check_ceiling) or [`check_band`](Self::check_band).
/// Portable spec contracts do NOT go through here; they stay bare `assert!`s so
/// no verdict can ever silence them.
pub struct BandGate {
  suite: String,
  verdict: BandVerdict,
  running: HostClass,
  recorded: Option<CharacterizedHost>,
  recharacterize: String,
}

impl BandGate {
  /// Opens the gate against the RUNNING host and prints the banner.
  ///
  /// `suite` names the gate in output; `recharacterize` is the exact command
  /// that re-measures these bands on this machine, quoted into the banner so a
  /// log says how to arm itself.
  pub fn open(
    suite: impl Into<String>,
    recorded: Option<CharacterizedHost>,
    recharacterize: impl Into<String>,
  ) -> Self {
    Self::open_with(suite, recorded, HostClass::running(), recharacterize)
  }

  /// [`open`](Self::open) against a caller-supplied running host — the hermetic
  /// seam, so the provenance tests exercise all three verdicts without owning
  /// three machines.
  pub fn open_with(
    suite: impl Into<String>,
    recorded: Option<CharacterizedHost>,
    running: HostClass,
    recharacterize: impl Into<String>,
  ) -> Self {
    let gate = BandGate {
      suite: suite.into(),
      verdict: band_verdict(recorded.as_ref(), &running),
      running,
      recorded,
      recharacterize: recharacterize.into(),
    };
    println!("{}", gate.banner());
    gate
  }

  /// This gate's verdict.
  pub fn verdict(&self) -> BandVerdict {
    self.verdict
  }

  /// Whether measured bands are asserted on this host.
  pub fn armed(&self) -> bool {
    self.verdict.asserts()
  }

  /// The banner [`open`](Self::open) prints: verdict, hosts, what still gates,
  /// and how to arm. Returned rather than only printed so a hermetic test can
  /// assert the loudness contract instead of trusting stdout.
  pub fn banner(&self) -> String {
    match self.verdict {
      BandVerdict::Armed => format!(
        "\n[band-gate] {}: measured bands ARE ASSERTED on this host.\n  \
         this host   : {}\n  \
         provenance  : the bands below were characterized on exactly this host class, so a\n                \
         band failure IS a finding — re-measure, do not widen.",
        self.suite, self.running
      ),
      BandVerdict::Unrecorded => format!(
        "\n[band-gate] {}: measured bands are NOT ASSERTED on this host.\n  \
         reason      : no characterization host is recorded in source (CHARACTERIZED_ON = None),\n                \
         so it is unknown which machine produced these numbers. A band is a\n                \
         description of ONE host; asserting it on an unrelated machine measures\n                \
         the machine, not the port.\n  \
         this host   : {}\n  \
         still gated : this suite's portable spec floors, asserted on EVERY host — the ship\n                \
         contract is not what is being skipped here.\n  \
         consequence : every band below is computed and printed, then SKIPPED. Read the\n                \
         `[band]` lines for the numbers this host actually produced.\n  \
         to arm      : re-measure on this machine, pin the printed numbers, and record\n                \
         CHARACTERIZED_ON = Some(this host class):\n                \
         {}",
        self.suite, self.running, self.recharacterize
      ),
      BandVerdict::Foreign => format!(
        "\n[band-gate] {}: measured bands are NOT ASSERTED on this host.\n  \
         reason      : the bands were characterized on a DIFFERENT host class. CoreML floats\n                \
         are not contracted portable across macOS builds or chips (#36) — neither\n                \
         the `CpuOnly` kernels that ship with the OS nor the Neural Engine's fp16\n                \
         arithmetic — so these numbers describe that machine, not this one.\n  \
         band host   : {}\n  \
         this host   : {}\n  \
         still gated : this suite's portable spec floors, asserted on EVERY host — the ship\n                \
         contract is not what is being skipped here.\n  \
         consequence : every band below is computed and printed, then SKIPPED. A number\n                \
         outside the band here is NOT evidence of a port defect.\n  \
         to arm here : re-measure on this machine, pin the printed numbers, and point\n                \
         CHARACTERIZED_ON at THIS host class instead — never widen one band to span\n                \
         both hosts, which would make it assert nothing:\n                \
         {}",
        self.suite,
        match self.recorded {
          Some(r) => r.to_string(),
          // Unreachable by `band_verdict`: `Foreign` is only produced from a
          // `Some`. Rendered rather than panicked so a banner can never be the
          // thing that fails a test.
          None => "<unrecorded>".to_string(),
        },
        self.running,
        self.recharacterize
      ),
    }
  }

  /// Measured LOWER bound: asserts `value >= floor` when armed.
  ///
  /// Returns the `[band]` line it printed, so the provenance tests can assert
  /// the reporting contract on the very call the real gates make.
  ///
  /// # Panics
  /// When armed and `value < floor`.
  pub fn check_floor(&self, what: &str, value: f32, floor: f32) -> String {
    self.check(
      what,
      value,
      value >= floor,
      &format!("measured floor {floor}"),
      &format!("{what} {value:.8} below the measured floor {floor}"),
    )
  }

  /// Measured UPPER bound: asserts `value < ceiling` when armed. Returns the
  /// printed `[band]` line, as [`check_floor`](Self::check_floor) does.
  ///
  /// # Panics
  /// When armed and `value >= ceiling`.
  pub fn check_ceiling(&self, what: &str, value: f32, ceiling: f32) -> String {
    self.check(
      what,
      value,
      value < ceiling,
      &format!("measured ceiling {ceiling}"),
      &format!("{what} {value:.8} not below the measured ceiling {ceiling}"),
    )
  }

  /// Measured two-sided band: asserts `lo <= value <= hi` when armed. Returns
  /// the printed `[band]` line, as [`check_floor`](Self::check_floor) does.
  ///
  /// # Panics
  /// When armed and `value` falls outside `[lo, hi]`.
  pub fn check_band(&self, what: &str, value: f32, lo: f32, hi: f32) -> String {
    self.check(
      what,
      value,
      (lo..=hi).contains(&value),
      &format!("characterized band [{lo}, {hi}]"),
      &format!("{what} {value:.8} outside the characterized band [{lo}, {hi}]"),
    )
  }

  /// The one enforcement path: ALWAYS print the measurement and whether it fits,
  /// on every host and under every verdict; assert only when armed.
  ///
  /// The print happens BEFORE the assert, so even an armed failure leaves the
  /// number in the log.
  fn check(&self, what: &str, value: f32, holds: bool, band: &str, failure: &str) -> String {
    let status = if holds { "ok     " } else { "OUTSIDE" };
    let line = if self.armed() {
      format!("[band] {status} {what} = {value:.8}  vs {band}  [ASSERTED]")
    } else {
      format!(
        "[band] {status} {what} = {value:.8}  vs {band}  [BAND NOT ASSERTED — {}]",
        self.skip_reason()
      )
    };
    println!("{line}");
    if self.armed() {
      assert!(holds, "{failure}\n{}", self.attribution());
    }
    line
  }

  /// One line naming why an unarmed band was skipped, appended to every
  /// `[band]` line so a single grep for `BAND NOT ASSERTED` explains itself.
  fn skip_reason(&self) -> &'static str {
    match self.verdict {
      BandVerdict::Armed => "armed",
      BandVerdict::Unrecorded => "no characterization host recorded",
      BandVerdict::Foreign => "characterized on a different host class",
    }
  }

  /// The provenance footer on an ARMED band's failure: this host IS the
  /// characterization host, so the failure is the port's, and the fix is a
  /// re-measurement rather than a wider tolerance.
  fn attribution(&self) -> String {
    format!(
      "  this host   : {}\n  \
       provenance  : this IS the host class the band was characterized on, so the failure is\n                \
       attributable — either the port moved or the platform did. Re-measure and\n                \
       re-pin deliberately; do NOT widen the band to make this pass:\n                \
       {}",
      self.running, self.recharacterize
    )
  }
}
