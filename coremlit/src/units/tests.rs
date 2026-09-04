use super::*;

#[test]
fn as_str_round_trips_from_str() {
  for u in [
    ComputeUnits::CpuOnly,
    ComputeUnits::CpuAndGpu,
    ComputeUnits::CpuAndNeuralEngine,
    ComputeUnits::All,
  ] {
    assert_eq!(u.as_str().parse::<ComputeUnits>().unwrap(), u);
  }
}

#[test]
fn display_matches_as_str() {
  assert_eq!(
    ComputeUnits::CpuAndNeuralEngine.to_string(),
    "cpu_and_neural_engine"
  );
}

#[test]
fn unknown_name_is_opaque_error() {
  assert!("tpu".parse::<ComputeUnits>().is_err());
}

#[test]
fn default_is_all() {
  assert_eq!(ComputeUnits::default(), ComputeUnits::All);
}

/// Every variant's document, pinned byte-exactly in BOTH directions.
///
/// Wildcard-free: `#[non_exhaustive]` binds DOWNSTREAM crates only, so this
/// in-crate match stays exhaustive and a new variant fails to compile here
/// until its spelling is pinned.
#[cfg(feature = "serde")]
#[test]
fn serde_document_form_is_pinned() {
  for units in [
    ComputeUnits::CpuOnly,
    ComputeUnits::CpuAndGpu,
    ComputeUnits::CpuAndNeuralEngine,
    ComputeUnits::All,
  ] {
    let doc = match units {
      ComputeUnits::CpuOnly => r#""cpu_only""#,
      ComputeUnits::CpuAndGpu => r#""cpu_and_gpu""#,
      ComputeUnits::CpuAndNeuralEngine => r#""cpu_and_neural_engine""#,
      ComputeUnits::All => r#""all""#,
    };
    assert_eq!(serde_json::to_string(&units).unwrap(), doc);
    assert_eq!(serde_json::from_str::<ComputeUnits>(doc).unwrap(), units);
  }
}

/// The impls are `as_str`/[`FromStr`](core::str::FromStr) and nothing else —
/// the document is the plain string, for every variant.
///
/// The retired per-door `serde(with)` bridges wrote exactly this, so this is
/// also the pin that the deletion changed no document: serializing the type
/// and serializing `as_str()` must produce the same bytes, and what `FromStr`
/// parses must be what `Deserialize` accepts.
#[cfg(feature = "serde")]
#[test]
fn the_document_is_the_as_str_string_for_every_variant() {
  for units in [
    ComputeUnits::CpuOnly,
    ComputeUnits::CpuAndGpu,
    ComputeUnits::CpuAndNeuralEngine,
    ComputeUnits::All,
  ] {
    let direct = serde_json::to_string(&units).unwrap();
    let as_str = serde_json::to_string(units.as_str()).unwrap();
    assert_eq!(direct, as_str, "document drift for {units}");
    assert_eq!(
      serde_json::from_str::<ComputeUnits>(&as_str).unwrap(),
      units.as_str().parse::<ComputeUnits>().unwrap()
    );
  }
}

/// The consumer's actual use: the knob as a FLAT KEY in their own config
/// struct, with no `serde(with)` bridge of their own.
///
/// This is what the public impls buy — before them a downstream crate could
/// not name this type in a `#[derive(Deserialize)]` struct at all.
#[cfg(feature = "serde")]
#[test]
fn a_consumer_struct_reads_the_knob_as_a_flat_key() {
  #[derive(Debug, PartialEq, serde::Serialize, serde::Deserialize)]
  struct NodeConfig {
    coreml_compute: ComputeUnits,
    #[serde(default)]
    fallback_compute: Option<ComputeUnits>,
  }

  let json = r#"{"coreml_compute":"cpu_only"}"#;
  let cfg: NodeConfig = serde_json::from_str(json).unwrap();
  assert_eq!(cfg.coreml_compute, ComputeUnits::CpuOnly);
  assert_eq!(cfg.fallback_compute, None);

  // TOML is the shape a node-options file actually carries.
  let cfg: NodeConfig =
    toml::from_str("coreml_compute = \"cpu_only\"\nfallback_compute = \"cpu_and_neural_engine\"\n")
      .unwrap();
  assert_eq!(
    cfg,
    NodeConfig {
      coreml_compute: ComputeUnits::CpuOnly,
      fallback_compute: Some(ComputeUnits::CpuAndNeuralEngine),
    }
  );

  // `Option` needs no bridge either: absent is `None`, present is the string.
  assert_eq!(
    serde_json::to_string(&NodeConfig {
      coreml_compute: ComputeUnits::All,
      fallback_compute: None,
    })
    .unwrap(),
    r#"{"coreml_compute":"all","fallback_compute":null}"#
  );
}

/// ONE protocol, not two: the direct impls and a door option struct's
/// `ComputeUnits` field reach the wire IDENTICALLY — checked in a format that
/// can tell an enum discriminant from a string.
///
/// `serde_json` and TOML cannot tell them apart: serde's enum protocol
/// (`serialize_unit_variant`) and the string protocol (`serialize_str`) both
/// render `"cpu_only"` in a self-describing format, so a derive and a
/// `serialize_str` bridge look equal there while a non-self-describing format
/// writes a one-byte variant index for one and a length-prefixed string for the
/// other — a value written by one route unreadable by the other. postcard is
/// that format, and `whisper::options::ComputeOptions` is a door struct whose
/// three fields are each a `ComputeUnits` and nothing else, so its encoding IS
/// the concatenation of three field encodings.
#[cfg(all(feature = "serde", feature = "whisper"))]
#[test]
fn one_binary_protocol_across_the_direct_impls_and_a_door_field() {
  use crate::audio::whisper::options::ComputeOptions;

  for units in [
    ComputeUnits::CpuOnly,
    ComputeUnits::CpuAndGpu,
    ComputeUnits::CpuAndNeuralEngine,
    ComputeUnits::All,
  ] {
    // The string protocol, spelled out: postcard writes a varint length (one
    // byte for every name here) then the `as_str` bytes.
    let mut expected = vec![u8::try_from(units.as_str().len()).unwrap()];
    expected.extend_from_slice(units.as_str().as_bytes());
    let direct = postcard::to_allocvec(&units).unwrap();
    assert_eq!(
      direct, expected,
      "{units} must reach a binary format as its `as_str` STRING, not as a \
       variant index: {direct:?} vs {expected:?}"
    );

    let opts = ComputeOptions::new()
      .with_mel(units)
      .with_encoder(units)
      .with_decoder(units);
    let door = postcard::to_allocvec(&opts).unwrap();
    assert_eq!(
      door,
      direct.repeat(3),
      "the door's field and the direct impl disagree for {units}"
    );

    // …and each route READS the other's bytes.
    assert_eq!(
      postcard::from_bytes::<ComputeUnits>(&direct).unwrap(),
      units
    );
    assert_eq!(postcard::from_bytes::<ComputeOptions>(&door).unwrap(), opts);
  }
}
