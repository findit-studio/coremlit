//! The vision part's instances: each loads the configured budgets' functions
//! of the vision bundle, under the kit's contract, and encodes prepared
//! images into image tokens.

use core::{num::NonZeroU16, time::Duration};
use std::time::Instant;

use super::{
  super::{
    config::HIDDEN_SIZE,
    error::{
      Artifact, BudgetNotLoaded, ContractRefusal, Error, Function, LoadRefusal, PredictionFailure,
      Result,
    },
    media::ImageTokens,
    options::{DerivationTerms, ImageOptions},
    prewarm::{PrewarmReport, Warmed},
  },
  VisionBudget,
  prepare::{PATCH_VALUES, POSITION_WIDTH, PreparedImage, VisionArtifacts},
};
use crate::{
  DataType, FunctionName, Model,
  embeddings::image::{ImageDimensions, Rgb8Image},
  model::contract::{Checked, Dim, FeatureContract, LoadContract, StateContract},
};

/// A budget's function contract: the five planes in, `image_tokens` F16
/// `[1, T, 512]` out.
fn contract(budget: VisionBudget) -> LoadContract {
  let (n, t) = (budget.patches(), budget.tokens());
  let f16 = |name, dims: &[usize]| {
    FeatureContract::new(
      name,
      DataType::F16,
      dims.iter().map(|&d| Dim::Exactly(d)).collect(),
    )
  };
  LoadContract::new(
    vec![
      f16("patches", &[1, n, PATCH_VALUES]),
      f16("pool", &[t, n]),
      f16("position_embeddings", &[1, n, POSITION_WIDTH]),
      f16("positions", &[1, n, 2]),
      f16("valid", &[1, n]),
    ],
    vec![f16("image_tokens", &[1, t, HIDDEN_SIZE])],
    StateContract::None,
  )
}

/// The slot of `budget` in an instance's function array.
const fn slot(budget: VisionBudget) -> usize {
  match budget {
    VisionBudget::Tokens70 => 0,
    VisionBudget::Tokens140 => 1,
    VisionBudget::Tokens280 => 2,
  }
}

/// One vision instance: the configured budgets' functions, loaded at
/// construction under the kit's contract. A budget it did not load is
/// refused by name and never loaded on the side.
///
/// `Send` and not `Sync`: encoding takes `&mut self`. Preparing an image
/// needs no instance: [`VisionArtifacts::prepare_into`] runs on any thread.
#[derive(Debug)]
pub struct ImageEncoder {
  artifacts: VisionArtifacts,
  options: ImageOptions,
  functions: [Option<Checked>; 3],
  loads: [Duration; 3],
}

impl ImageEncoder {
  /// An instance over `artifacts` under `options`: each of `options`'
  /// budgets' functions loaded with its Core ML choices, under the kit's
  /// contract.
  ///
  /// # Errors
  /// [`Error::BudgetNotLoaded`] when the default budget is outside the
  /// budgets; [`Error::ArtifactIdentity`] when `options` pins another label;
  /// [`Error::Load`] or [`Error::Contract`] naming the function.
  pub fn new(artifacts: &VisionArtifacts, options: &ImageOptions) -> Result<Self> {
    options.validate()?;
    let view = &artifacts.shared.view;
    view.expect(options.expect_artifacts())?;
    let mut functions: [Option<Checked>; 3] = Default::default();
    let mut loads = [Duration::ZERO; 3];
    for budget in VisionBudget::ALL {
      if !options.budgets().contains(budget) {
        continue;
      }
      let function = Some(Function::Vision(budget));
      let started = Instant::now();
      let name = FunctionName::new(budget.as_str()).expect("a function name is not empty");
      let model = Model::load_function_with(view.bundle(), name, &options.coreml())
        .map_err(|e| Error::Load(LoadRefusal::new(Artifact::VisionBundle, function, e)))?;
      let model = Checked::new(model, &contract(budget))
        .map_err(|violation| ContractRefusal::of(Artifact::VisionBundle, function, violation))?;
      functions[slot(budget)] = Some(model);
      loads[slot(budget)] = started.elapsed();
    }
    Ok(Self {
      artifacts: artifacts.clone(),
      options: options.clone(),
      functions,
      loads,
    })
  }

  /// Runs one throwaway prediction on each loaded budget, so the first image
  /// pays no device specialization. The report holds each budget's load and
  /// that prediction.
  ///
  /// # Errors
  /// As [`Self::encode`].
  pub fn prewarm(&mut self) -> Result<PrewarmReport> {
    let gray = [128u8; 48 * 48 * 3];
    let image = Rgb8Image::new(&gray, 48, 48).expect("a 48 × 48 RGB image");
    let mut prepared = PreparedImage::new();
    let mut report = PrewarmReport::default();
    for budget in VisionBudget::ALL {
      if self.functions[slot(budget)].is_none() {
        continue;
      }
      self
        .artifacts
        .prepare_into(image, Some(budget), &self.options, &mut prepared)?;
      let started = Instant::now();
      self.encode(&prepared)?;
      report.push(Warmed::new(
        Some(Function::Vision(budget)),
        self.loads[slot(budget)],
        started.elapsed(),
      ));
    }
    Ok(report)
  }

  /// Encodes a prepared image through its budget's function: the model's own
  /// `[1, T, 512]` output, and its real tokens, the patches over nine.
  ///
  /// # Errors
  /// [`Error::BudgetNotLoaded`] for a budget the instance did not load;
  /// [`Error::ImageDimensions`] for an image never prepared;
  /// [`Error::Prediction`] naming the function.
  pub fn encode(&mut self, prepared: &PreparedImage) -> Result<ImageTokens> {
    let (Some(budget), Some(inputs)) = (prepared.budget(), prepared.inputs()) else {
      return Err(Error::ImageDimensions(ImageDimensions::new(0, 0)));
    };
    let Some(model) = self.functions[slot(budget)].as_mut() else {
      return Err(Error::BudgetNotLoaded(BudgetNotLoaded::new(
        budget,
        self.options.budgets(),
      )));
    };
    let function = Some(Function::Vision(budget));
    let failure =
      |e| Error::Prediction(PredictionFailure::new(Artifact::VisionBundle, function, e));
    let mut outputs = model.predict_with(&inputs).map_err(failure)?;
    let tokens = outputs.take("image_tokens").ok_or_else(|| {
      failure(crate::PredictionError::MissingOutput(
        "image_tokens".to_owned(),
      ))
    })?;
    let valid = u16::try_from(prepared.tokens())
      .ok()
      .and_then(NonZeroU16::new)
      .expect("a prepared image holds at least one pooling group");
    Ok(ImageTokens::new(tokens, budget, valid))
  }

  /// The options the instance runs under.
  #[inline]
  pub const fn options(&self) -> &ImageOptions {
    &self.options
  }

  /// What the instance's tokens derive from: the vision artifacts' label and
  /// every option that changes them.
  #[inline]
  pub fn derivation_terms(&self) -> DerivationTerms<'_> {
    DerivationTerms::vision(self.artifacts.artifact_label(), &self.options)
  }
}

#[cfg(test)]
mod tests;
