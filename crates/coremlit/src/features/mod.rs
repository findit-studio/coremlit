//! Named feature dictionaries — model inputs and outputs.

use objc2::{
  AnyThread,
  rc::Retained,
  runtime::{AnyObject, ProtocolObject},
};
use objc2_core_ml::{MLDictionaryFeatureProvider, MLFeatureProvider, MLFeatureValue};
use objc2_foundation::{NSDictionary, NSString};

use crate::{MultiArray, NsErrorInfo, PredictionError};

/// An insertion-ordered set of named [`MultiArray`]s.
///
/// The input and output currency of [`Model::predict`](crate::Model::predict).
#[derive(Debug)]
pub struct Features {
  entries: Vec<(String, MultiArray)>,
}

impl Default for Features {
  fn default() -> Self {
    Self::new()
  }
}

impl Features {
  /// An empty feature set.
  #[inline(always)]
  pub const fn new() -> Self {
    Self {
      entries: Vec::new(),
    }
  }

  /// Inserts (or replaces) a named array.
  ///
  /// Replacing an existing name moves it to the end of iteration order.
  pub fn insert(&mut self, name: impl Into<String>, array: MultiArray) -> &mut Self {
    let name = name.into();
    self.entries.retain(|(existing, _)| *existing != name);
    self.entries.push((name, array));
    self
  }

  /// Consuming form of [`Self::insert`].
  #[must_use]
  pub fn with(mut self, name: impl Into<String>, array: MultiArray) -> Self {
    self.insert(name, array);
    self
  }

  /// Borrows the array named `name`.
  pub fn get(&self, name: &str) -> Option<&MultiArray> {
    self.entries.iter().find(|(n, _)| n == name).map(|(_, a)| a)
  }

  /// Removes and returns the array named `name`.
  pub fn take(&mut self, name: &str) -> Option<MultiArray> {
    let index = self.entries.iter().position(|(n, _)| n == name)?;
    Some(self.entries.remove(index).1)
  }

  /// Iterates the feature names in insertion order.
  pub fn names(&self) -> impl Iterator<Item = &str> {
    self.entries.iter().map(|(n, _)| n.as_str())
  }

  /// Number of features.
  #[inline(always)]
  pub const fn len(&self) -> usize {
    self.entries.len()
  }

  /// Whether the set is empty.
  #[inline(always)]
  pub const fn is_empty(&self) -> bool {
    self.entries.is_empty()
  }

  // Every array's native buffer identity, in insertion order. Seeds
  // `from_provider`'s aliasing detection: `Model::predict` calls this on
  // its *inputs* before extracting outputs, because an input outlives the
  // call (the caller still owns it) — an output that echoes an input's
  // buffer (an identity/zero-copy model) is exactly the aliasing case
  // `from_provider` must catch, same as one array under two output names.
  pub(crate) fn data_ptrs(&self) -> Vec<*const core::ffi::c_void> {
    self.entries.iter().map(|(_, a)| a.data_ptr()).collect()
  }

  // Bridges to CoreML's `MLDictionaryFeatureProvider`, the concrete
  // `MLFeatureProvider` used to feed `Model::predict`.
  pub(crate) fn to_provider(
    &self,
  ) -> Result<Retained<MLDictionaryFeatureProvider>, PredictionError> {
    let keys: Vec<Retained<NSString>> = self
      .entries
      .iter()
      .map(|(n, _)| NSString::from_str(n))
      .collect();
    let values: Vec<Retained<AnyObject>> = self
      .entries
      .iter()
      .map(|(_, a)| {
        // SAFETY: featureValueWithMultiArray is a plain constructor send;
        // `a.raw()` borrows a live MLMultiArray for the call's duration and
        // the returned MLFeatureValue retains it, so no dangling reference
        // results once this closure returns.
        let value: Retained<MLFeatureValue> =
          unsafe { MLFeatureValue::featureValueWithMultiArray(a.raw()) };
        // MLDictionaryFeatureProvider's initializer is typed over AnyObject
        // (see below); erase the concrete class now.
        value.into()
      })
      .collect();
    let key_refs: Vec<&NSString> = keys.iter().map(|k| k.as_ref()).collect();
    let dict = NSDictionary::from_retained_objects(&key_refs, &values);
    // SAFETY: `dict` maps NSString keys to MLFeatureValue objects (erased
    // to AnyObject), exactly the generic-dictionary-of-feature-values shape
    // `initWithDictionary:error:` documents; `alloc()` supplies a fresh,
    // unaliased receiver.
    unsafe {
      MLDictionaryFeatureProvider::initWithDictionary_error(
        MLDictionaryFeatureProvider::alloc(),
        &dict,
      )
    }
    .map_err(|e| PredictionError::Native(NsErrorInfo::from_ns_error(&e)))
  }

  // Extracts named multi-arrays out of any CoreML feature provider (e.g. a
  // prediction's output provider).
  //
  // Each returned `MultiArray` wraps a `Retained<MLMultiArray>` obtained
  // fresh from `MLFeatureValue::multiArrayValue()` — but that handle's
  // *buffer* may still be referenced from inside `provider` (the
  // `MLFeatureValue` this came from, and the dictionary/provider holding
  // it), from a caller-held input an identity/zero-copy model echoed back
  // as this same output, or from another output name in this same
  // `provider` pointing at the same array. `MultiArray::from_raw`'s
  // sole-ownership invariant is therefore not met by `provider` alone.
  //
  // `known_ptrs` closes every one of those gaps: callers seed it with the
  // `data_ptr` of every array that could be aliased and outlives this call
  // (`Model::predict` seeds it with every input's, via `Features::data_ptrs`
  // — inputs are exactly the case a duplicate-output-provider fixture can't
  // reach on its own). Each extracted array whose `data_ptr` is already
  // present is deep-copied into a freshly allocated, uniquely owned buffer
  // before being inserted; either way, its (possibly new) `data_ptr` is
  // then pushed, so a *third* name aliasing the same original buffer is
  // caught too. With that, dropping the output provider immediately after
  // calling this function (as `Model::predict` does) restores effective
  // sole ownership of every array extracted here unconditionally — any
  // surviving alias was already copied, not just the ones the provider
  // itself would release.
  //
  // Extracted arrays may also be non-contiguous (row-padded, as pixel-
  // buffer-backed arrays can be): `MultiArray::as_slice`/`as_slice_mut`
  // already refuse those with `TensorError::NonContiguous` rather than
  // misreading the padding, so nothing extra is needed here.
  pub(crate) fn from_provider(
    provider: &ProtocolObject<dyn MLFeatureProvider>,
    known_ptrs: &mut Vec<*const core::ffi::c_void>,
  ) -> Result<Self, PredictionError> {
    let mut features = Self::new();
    // SAFETY: protocol getter message send on a live provider.
    let names = unsafe { provider.featureNames() };
    for name in names.iter() {
      let name_str = name.to_string();
      // SAFETY: `name` was just yielded by `provider.featureNames()`, so it
      // names a member of this same provider.
      let value = unsafe { provider.featureValueForName(&name) }.ok_or_else(|| {
        PredictionError::MissingOutput {
          name: name_str.clone(),
        }
      })?;
      // SAFETY: plain accessor message send on a live MLFeatureValue; `None`
      // means the feature holds a non-multi-array value, not invalid state.
      let array =
        unsafe { value.multiArrayValue() }.ok_or_else(|| PredictionError::NotMultiArray {
          name: name_str.clone(),
        })?;
      let mut array = MultiArray::from_raw(array);
      if known_ptrs.contains(&array.data_ptr()) {
        array = array
          .deep_copy()
          .map_err(PredictionError::AliasCopyFailed)?;
      }
      known_ptrs.push(array.data_ptr());
      features.insert(name_str, array);
    }
    Ok(features)
  }
}

#[cfg(test)]
mod tests;
