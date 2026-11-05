use windit::windowed::Vector;

use super::*;

/// A deterministic, non-degenerate 768-d direction, not yet unit length.
fn raw() -> Vec<f64> {
  (0..EMBEDDING_DIM)
    .map(|i| ((i as f64 * 0.37).sin() + 0.05) * (1.0 + (i % 7) as f64))
    .collect()
}

fn unit() -> Embedding {
  Embedding::from_unnormalized(&raw()).expect("a finite, non-zero vector")
}

fn norm(v: &[f32]) -> f64 {
  v.iter()
    .map(|x| f64::from(*x) * f64::from(*x))
    .sum::<f64>()
    .sqrt()
}

/// K3: every Matryoshka truncation is unit length, and is the first `M`
/// coordinates re-normalized.
#[test]
fn a_truncation_is_the_renormalized_prefix_at_unit_length() {
  let full = unit();
  assert!((norm(full.as_slice()) - 1.0).abs() < 1e-6);

  let short: Embedding<256> = full.truncate();
  let prefix = &full.as_slice()[..256];
  let prefix_norm = norm(prefix);
  assert!((norm(short.as_slice()) - 1.0).abs() < 1e-6, "unit length");
  for (i, (got, base)) in short.as_slice().iter().zip(prefix).enumerate() {
    let want = f64::from(*base) / prefix_norm;
    assert!((f64::from(*got) - want).abs() < 1e-6, "component {i}");
  }

  let widths = [
    full.truncate::<768>().dim(),
    full.truncate::<512>().dim(),
    full.truncate::<256>().dim(),
    full.truncate::<128>().dim(),
  ];
  assert_eq!(widths, [768, 512, 256, 128]);
  for v in [
    norm(full.truncate::<512>().as_slice()),
    norm(full.truncate::<128>().as_slice()),
  ] {
    assert!((v - 1.0).abs() < 1e-6);
  }
}

/// The dot product of unit vectors is their cosine; a vector with itself is 1.
#[test]
fn the_cosine_of_a_vector_with_itself_is_one() {
  let v = unit();
  assert!((f64::from(v.cosine(&v)) - 1.0).abs() < 1e-6);
  let w: Embedding = Embedding::from_unnormalized(&raw().iter().rev().copied().collect::<Vec<_>>())
    .expect("non-zero");
  assert!(v.cosine(&w) < 0.99);
  assert_eq!(v.dot(&w), v.cosine(&w));
}

/// windit's fold hands back a vector of the wrong width, a zero vector or a
/// non-finite one only as its own refusal.
#[test]
fn the_fold_constructor_refuses_by_windits_names() {
  assert!(matches!(
    Embedding::<768>::from_unnormalized(&[1.0; 767]),
    Err(windit::WinditError::DimMismatch {
      got: 767,
      expected: 768
    })
  ));
  assert!(matches!(
    Embedding::<128>::from_unnormalized(&[0.0; 128]),
    Err(windit::WinditError::NonFinite)
  ));
  let mut nan = vec![1.0; 128];
  nan[5] = f64::NAN;
  assert!(matches!(
    Embedding::<128>::from_unnormalized(&nan),
    Err(windit::WinditError::NonFinite)
  ));
  let v = Embedding::<128>::from_unnormalized(&[2.0; 128]).unwrap();
  assert_eq!(Vector::as_slice(&v), v.as_slice());
}

/// `Debug` shows the width and the head, never the whole vector.
#[test]
fn the_debug_form_is_the_width_and_the_head() {
  let v = Embedding::<128>::from_unnormalized(&[1.0; 128]).unwrap();
  assert_eq!(
    format!("{v:?}"),
    "Embedding { dim: 128, head: [0.0884, 0.0884, 0.0884, ..] }"
  );
}
