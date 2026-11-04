use super::*;

/// A responder that records every selector it is asked about and answers each
/// from `answers`, refusing any selector not listed there.
struct Stub {
  answers: Vec<(Sel, bool)>,
  asked: Vec<Sel>,
}

impl Stub {
  fn answering(answers: &[(Sel, bool)]) -> Self {
    Self {
      answers: answers.to_vec(),
      asked: Vec::new(),
    }
  }

  fn responds(&mut self, selector: Sel) -> bool {
    self.asked.push(selector);
    self
      .answers
      .iter()
      .find(|(answered, _)| *answered == selector)
      .is_some_and(|(_, yes)| *yes)
  }
}

fn set_function_name() -> Sel {
  objc2::sel!(setFunctionName:)
}

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

/// A named function on an OS whose `MLModelConfiguration` lacks
/// `setFunctionName:` (before macOS 15) is refused by name, after asking
/// exactly that one selector.
#[test]
fn a_function_is_refused_by_name_when_the_configuration_cannot_select_one() {
  let mut stub = Stub::answering(&[(set_function_name(), false)]);
  let function = FunctionName::new("double");
  let refusal = plan_sends(function, |selector| stub.responds(selector)).unwrap_err();
  assert_eq!(refusal, LoadError::FunctionsUnsupported);
  assert_eq!(stub.asked, [set_function_name()]);
}

/// Where the configuration answers, the plan sends the name it was given.
#[test]
fn a_function_is_planned_when_the_configuration_answers() {
  let mut stub = Stub::answering(&[(set_function_name(), true)]);
  let function = FunctionName::new("double");
  let sends = plan_sends(function, |selector| stub.responds(selector)).unwrap();
  assert_eq!(sends.function(), function);
  assert_eq!(stub.asked, [set_function_name()]);
}

/// **A load without a name asks nothing and sends nothing.** That is what
/// keeps every existing door's `Model::load` path byte-for-byte what it was:
/// no probe runs, so not even an OS without the selector can refuse it.
#[test]
fn a_load_without_a_function_name_probes_nothing_and_sends_nothing() {
  let mut stub = Stub::answering(&[]);
  let sends = plan_sends(None, |selector| stub.responds(selector)).unwrap();
  assert_eq!(sends.function(), None);
  assert!(stub.asked.is_empty(), "asked {:?}", stub.asked);
}
