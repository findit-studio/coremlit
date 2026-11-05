use super::{
  Receiver::{Configuration, OptimizationHints},
  *,
};

/// A responder that records every question it is asked and answers each from
/// `answers`, refusing any question not listed there.
struct Stub {
  answers: Vec<(Receiver, Sel, bool)>,
  asked: Vec<(Receiver, Sel)>,
}

impl Stub {
  fn answering(answers: &[(Receiver, Sel, bool)]) -> Self {
    Self {
      answers: answers.to_vec(),
      asked: Vec::new(),
    }
  }

  fn responds(&mut self, receiver: Receiver, selector: Sel) -> bool {
    self.asked.push((receiver, selector));
    self
      .answers
      .iter()
      .find(|(r, s, _)| *r == receiver && *s == selector)
      .is_some_and(|(_, _, yes)| *yes)
  }

  /// `plan_sends` against this stub.
  fn plan<'a>(
    &mut self,
    function: Option<FunctionName<'a>>,
    options: &LoadOptions,
  ) -> Result<Sends<'a>, LoadError> {
    plan_sends(function, options, |receiver, selector| {
      self.responds(receiver, selector)
    })
  }
}

fn set_function_name() -> Sel {
  objc2::sel!(setFunctionName:)
}

fn set_low_precision() -> Sel {
  objc2::sel!(setAllowLowPrecisionAccumulationOnGPU:)
}

fn set_optimization_hints() -> Sel {
  objc2::sel!(setOptimizationHints:)
}

fn set_specialization_strategy() -> Sel {
  objc2::sel!(setSpecializationStrategy:)
}

/// Every selector a load can send, each answered yes: a macOS 15 host.
fn everything_answers() -> Stub {
  Stub::answering(&[
    (Configuration, set_function_name(), true),
    (Configuration, set_low_precision(), true),
    (Configuration, set_optimization_hints(), true),
    (OptimizationHints, set_specialization_strategy(), true),
  ])
}

fn every_unit() -> [ComputeUnits; 4] {
  [
    ComputeUnits::CpuOnly,
    ComputeUnits::CpuAndGpu,
    ComputeUnits::CpuAndNeuralEngine,
    ComputeUnits::All,
  ]
}

fn non_default(compute: ComputeUnits) -> LoadOptions {
  LoadOptions::new(compute)
    .with_low_precision_gpu_accumulation(true)
    .with_specialization(Specialization::FastPrediction)
}

// ── FunctionName ───────────────────────────────────────────────────────────

#[test]
fn an_empty_function_name_names_no_function() {
  assert_eq!(FunctionName::new(""), None);
}

#[test]
fn a_function_name_is_sent_exactly_as_written() {
  for name in ["embed_128", "vision_70", " padded ", "ünïcode", "a"] {
    assert_eq!(FunctionName::new(name).map(|f| f.as_str()), Some(name));
  }
}

// ── The decision, against a stubbed responder ──────────────────────────────

/// **A default asks nothing and sends nothing.** No function name, no
/// low-precision accumulation and `Specialization::Default`, under every
/// compute unit: not one probe runs, so not even an OS without any of these
/// selectors can refuse the load every existing door makes.
#[test]
fn defaults_probe_nothing_and_send_nothing() {
  for compute in every_unit() {
    let mut stub = Stub::answering(&[]);
    let sends = stub.plan(None, &LoadOptions::new(compute)).unwrap();
    assert_eq!(sends.function(), None);
    assert!(!sends.low_precision_gpu_accumulation());
    assert_eq!(sends.specialization(), None);
    assert!(stub.asked.is_empty(), "{compute}: asked {:?}", stub.asked);
  }
}

/// A named function on an OS whose `MLModelConfiguration` lacks
/// `setFunctionName:` (before macOS 15) is refused by name, after asking
/// exactly that one question.
#[test]
fn a_function_is_refused_by_name_when_the_configuration_cannot_select_one() {
  let mut stub = Stub::answering(&[(Configuration, set_function_name(), false)]);
  let refusal = stub
    .plan(
      FunctionName::new("double"),
      &LoadOptions::new(ComputeUnits::CpuOnly),
    )
    .unwrap_err();
  assert_eq!(refusal, LoadError::FunctionsUnsupported);
  assert_eq!(stub.asked, [(Configuration, set_function_name())]);
}

/// Low-precision accumulation asked for on an OS whose configuration lacks its
/// setter is refused by name, never dropped.
#[test]
fn low_precision_accumulation_is_refused_by_name_when_the_configuration_lacks_it() {
  let mut stub = Stub::answering(&[(Configuration, set_low_precision(), false)]);
  let options = LoadOptions::new(ComputeUnits::CpuAndGpu).with_low_precision_gpu_accumulation(true);
  let refusal = stub.plan(None, &options).unwrap_err();
  assert_eq!(
    refusal,
    LoadError::OptionUnsupported(LoadOption::LowPrecisionGpuAccumulation)
  );
  assert_eq!(stub.asked, [(Configuration, set_low_precision())]);
}

/// **Before macOS 14.4 the `MLOptimizationHints` class is never asked about.**
/// The configuration answers `setOptimizationHints:` first; on a no, the load
/// is refused by name and the class is not touched, because looking up a
/// class the runtime lacks panics. This is the order the production
/// responder's class lookup relies on.
#[test]
fn without_optimization_hints_the_hints_class_is_never_asked_about() {
  let mut stub = Stub::answering(&[(Configuration, set_optimization_hints(), false)]);
  let options =
    LoadOptions::new(ComputeUnits::CpuAndGpu).with_specialization(Specialization::FastPrediction);
  let refusal = stub.plan(None, &options).unwrap_err();
  assert_eq!(
    refusal,
    LoadError::OptionUnsupported(LoadOption::Specialization)
  );
  assert_eq!(stub.asked, [(Configuration, set_optimization_hints())]);
}

/// **macOS 14.4 to 14.x: the hints exist, their strategy does not.**
/// `optimizationHints` arrived in 14.4 and its `specializationStrategy` in
/// 15.0, so a yes from the configuration is not enough: the class is asked
/// next, and a no there is refused by name too.
#[test]
fn hints_without_a_specialization_strategy_are_refused_by_name() {
  let mut stub = Stub::answering(&[
    (Configuration, set_optimization_hints(), true),
    (OptimizationHints, set_specialization_strategy(), false),
  ]);
  let options =
    LoadOptions::new(ComputeUnits::CpuAndGpu).with_specialization(Specialization::FastPrediction);
  let refusal = stub.plan(None, &options).unwrap_err();
  assert_eq!(
    refusal,
    LoadError::OptionUnsupported(LoadOption::Specialization)
  );
  assert_eq!(
    stub.asked,
    [
      (Configuration, set_optimization_hints()),
      (OptimizationHints, set_specialization_strategy()),
    ]
  );
}

/// Where everything answers, every non-default choice is planned, each asked
/// about exactly once and in the order the sends depend on.
#[test]
fn every_non_default_choice_is_planned_where_everything_answers() {
  for compute in every_unit() {
    let mut stub = everything_answers();
    let function = FunctionName::new("double");
    let sends = stub.plan(function, &non_default(compute)).unwrap();
    assert_eq!(sends.function(), function);
    assert!(sends.low_precision_gpu_accumulation());
    assert_eq!(sends.specialization(), Some(Specialization::FastPrediction));
    assert_eq!(
      stub.asked,
      [
        (Configuration, set_function_name()),
        (Configuration, set_low_precision()),
        (Configuration, set_optimization_hints()),
        (OptimizationHints, set_specialization_strategy()),
      ],
      "{compute}"
    );
  }
}

/// Each choice asks only its own questions: a non-default value of one never
/// probes, or sends, another.
#[test]
fn each_choice_asks_only_its_own_questions() {
  let mut stub = everything_answers();
  let options = LoadOptions::new(ComputeUnits::All).with_low_precision_gpu_accumulation(true);
  let sends = stub.plan(None, &options).unwrap();
  assert!(sends.low_precision_gpu_accumulation());
  assert_eq!(sends.specialization(), None);
  assert_eq!(stub.asked, [(Configuration, set_low_precision())]);

  let mut stub = everything_answers();
  let options =
    LoadOptions::new(ComputeUnits::All).with_specialization(Specialization::FastPrediction);
  let sends = stub.plan(None, &options).unwrap();
  assert!(!sends.low_precision_gpu_accumulation());
  assert_eq!(sends.specialization(), Some(Specialization::FastPrediction));
  assert_eq!(
    stub.asked,
    [
      (Configuration, set_optimization_hints()),
      (OptimizationHints, set_specialization_strategy()),
    ]
  );
}

// ── LoadOptions ────────────────────────────────────────────────────────────

#[test]
fn new_is_core_mls_defaults_on_the_given_units() {
  for compute in every_unit() {
    let options = LoadOptions::new(compute);
    assert_eq!(options.compute(), compute);
    assert!(!options.low_precision_gpu_accumulation());
    assert_eq!(options.specialization(), Specialization::Default);
  }
  assert_eq!(Specialization::default(), Specialization::Default);
}

#[test]
fn builders_and_setters_agree() {
  let built = LoadOptions::new(ComputeUnits::CpuOnly)
    .with_compute(ComputeUnits::CpuAndGpu)
    .with_low_precision_gpu_accumulation(true)
    .with_specialization(Specialization::FastPrediction);
  let mut set = LoadOptions::new(ComputeUnits::CpuOnly);
  set
    .set_compute(ComputeUnits::CpuAndGpu)
    .set_low_precision_gpu_accumulation(true)
    .set_specialization(Specialization::FastPrediction);
  assert_eq!(built, set);
  assert_eq!(built, non_default(ComputeUnits::CpuAndGpu));
}

/// **The `Display` spelling, pinned.** It is persisted by downstream
/// derivation fingerprints, so a change here is a change to every fingerprint
/// that nests it.
#[test]
fn load_options_display_is_pinned() {
  assert_eq!(
    LoadOptions::new(ComputeUnits::CpuAndNeuralEngine).to_string(),
    "compute=cpu_and_neural_engine,low_precision_gpu_accumulation=false,specialization=default"
  );
  assert_eq!(
    non_default(ComputeUnits::CpuAndGpu).to_string(),
    "compute=cpu_and_gpu,low_precision_gpu_accumulation=true,specialization=fast_prediction"
  );
}

/// The vocabularies print their wire spelling. Wildcard-free on purpose: a new
/// variant fails to compile here until its spelling is pinned.
#[test]
fn vocabulary_spellings_are_pinned() {
  for specialization in [Specialization::Default, Specialization::FastPrediction] {
    let spelled = match specialization {
      Specialization::Default => "default",
      Specialization::FastPrediction => "fast_prediction",
    };
    assert_eq!(specialization.as_str(), spelled);
    assert_eq!(specialization.to_string(), spelled);
  }
  for option in [
    LoadOption::LowPrecisionGpuAccumulation,
    LoadOption::Specialization,
  ] {
    let spelled = match option {
      LoadOption::LowPrecisionGpuAccumulation => "low_precision_gpu_accumulation",
      LoadOption::Specialization => "specialization",
    };
    assert_eq!(option.as_str(), spelled);
    assert_eq!(option.to_string(), spelled);
  }
}

/// The raw values the strategy is sent as are Core ML's own.
#[test]
fn specialization_maps_to_core_mls_strategies() {
  assert_eq!(
    Specialization::Default.to_raw(),
    MLSpecializationStrategy::Default
  );
  assert_eq!(
    Specialization::FastPrediction.to_raw(),
    MLSpecializationStrategy::FastPrediction
  );
}

#[cfg(feature = "serde")]
mod wire {
  use super::*;

  /// Both directions, byte for byte.
  #[test]
  fn the_json_document_is_pinned() {
    let doc = r#"{"compute":"cpu_and_gpu","low_precision_gpu_accumulation":true,"specialization":"fast_prediction"}"#;
    let options = non_default(ComputeUnits::CpuAndGpu);
    assert_eq!(serde_json::to_string(&options).unwrap(), doc);
    assert_eq!(serde_json::from_str::<LoadOptions>(doc).unwrap(), options);
  }

  /// The two defaulted keys may be left out; `compute` may not.
  #[test]
  fn compute_is_required_and_the_rest_default() {
    assert_eq!(
      serde_json::from_str::<LoadOptions>(r#"{"compute":"cpu_only"}"#).unwrap(),
      LoadOptions::new(ComputeUnits::CpuOnly)
    );
    let missing = serde_json::from_str::<LoadOptions>(r#"{"specialization":"default"}"#)
      .unwrap_err()
      .to_string();
    assert!(missing.contains("compute"), "{missing}");
  }

  /// **A misspelled option is an error naming it, never a silent default.**
  #[test]
  fn an_unknown_key_is_refused_by_name() {
    let refused = serde_json::from_str::<LoadOptions>(
      r#"{"compute":"cpu_only","specialisation":"fast_prediction"}"#,
    )
    .unwrap_err()
    .to_string();
    assert!(refused.contains("specialisation"), "{refused}");
  }

  #[test]
  fn an_unknown_strategy_is_refused() {
    assert!(
      serde_json::from_str::<LoadOptions>(r#"{"compute":"cpu_only","specialization":"fast"}"#)
        .is_err()
    );
  }

  /// TOML, the deployment documents' format, nests the same keys.
  #[test]
  fn a_toml_table_reads_the_same_options() {
    let options: LoadOptions =
      toml::from_str("compute = \"cpu_and_neural_engine\"\nspecialization = \"fast_prediction\"\n")
        .unwrap();
    assert_eq!(
      options,
      LoadOptions::new(ComputeUnits::CpuAndNeuralEngine)
        .with_specialization(Specialization::FastPrediction)
    );
  }

  /// A format that spells no field names carries the shape itself.
  #[test]
  fn every_combination_round_trips_through_postcard() {
    for compute in every_unit() {
      for low in [false, true] {
        for specialization in [Specialization::Default, Specialization::FastPrediction] {
          let options = LoadOptions::new(compute)
            .with_low_precision_gpu_accumulation(low)
            .with_specialization(specialization);
          let bytes = postcard::to_allocvec(&options).unwrap();
          assert_eq!(
            postcard::from_bytes::<LoadOptions>(&bytes).unwrap(),
            options
          );
        }
      }
    }
  }
}
