//! The kit's fallible reservations (condition 4): one door for every buffer
//! an input or an artifact sizes, so an allocation that cannot be made is
//! answered as [`Error::Allocation`] naming its site, never as an abort.
//!
//! A buffer grows through [`reserve`], [`reserve_exact`], [`reserve_heap`],
//! [`reserve_str`], [`with_capacity`] or [`collect`]. Tests reach every site
//! through [`inject`]: a site set to refuse answers as a failed reservation
//! would, so each road's refusal is a law.

use std::collections::BinaryHeap;

use super::error::{Allocation, AllocationSite, Error, Result};

/// The refusal of `additional` more elements of `T` at `site`.
fn refused<T>(site: AllocationSite, additional: usize) -> Error {
  Error::Allocation(Allocation::new(
    site,
    additional.saturating_mul(size_of::<T>()),
  ))
}

/// Room for `additional` more elements in `buffer`, amortized.
pub(crate) fn reserve<T>(
  buffer: &mut Vec<T>,
  additional: usize,
  site: AllocationSite,
) -> Result<()> {
  if inject::refuses(site) {
    return Err(refused::<T>(site, additional));
  }
  buffer
    .try_reserve(additional)
    .map_err(|_| refused::<T>(site, additional))
}

/// Room for exactly `additional` more elements in `buffer`.
pub(crate) fn reserve_exact<T>(
  buffer: &mut Vec<T>,
  additional: usize,
  site: AllocationSite,
) -> Result<()> {
  if inject::refuses(site) {
    return Err(refused::<T>(site, additional));
  }
  buffer
    .try_reserve_exact(additional)
    .map_err(|_| refused::<T>(site, additional))
}

/// Room for `additional` more elements in `heap`.
pub(crate) fn reserve_heap<T: Ord>(
  heap: &mut BinaryHeap<T>,
  additional: usize,
  site: AllocationSite,
) -> Result<()> {
  if inject::refuses(site) {
    return Err(refused::<T>(site, additional));
  }
  heap
    .try_reserve(additional)
    .map_err(|_| refused::<T>(site, additional))
}

/// Room for `additional` more bytes in `text`.
pub(crate) fn reserve_str(
  text: &mut String,
  additional: usize,
  site: AllocationSite,
) -> Result<()> {
  if inject::refuses(site) {
    return Err(refused::<u8>(site, additional));
  }
  text
    .try_reserve(additional)
    .map_err(|_| refused::<u8>(site, additional))
}

/// An empty list with room for exactly `len` elements.
pub(crate) fn with_capacity<T>(len: usize, site: AllocationSite) -> Result<Vec<T>> {
  let mut list = Vec::new();
  reserve_exact(&mut list, len, site)?;
  Ok(list)
}

/// `items`, whose count is known, collected into a list reserved exactly.
pub(crate) fn collect<T>(
  items: impl ExactSizeIterator<Item = T>,
  site: AllocationSite,
) -> Result<Vec<T>> {
  let mut list = with_capacity(items.len(), site)?;
  list.extend(items);
  Ok(list)
}

/// The test seam: while a [`inject::Guard`] lives, the reservation at its site
/// that follows the given number of successful ones there refuses, as do all
/// later ones at that site, on this thread.
#[cfg(test)]
pub(crate) mod inject {
  use core::cell::Cell;

  use super::AllocationSite;

  std::thread_local! {
    static REFUSE: Cell<Option<(AllocationSite, usize)>> = const { Cell::new(None) };
  }

  /// Refuses `site` after `skip` successful reservations there, until the
  /// guard drops.
  #[must_use = "the seam clears when the guard drops"]
  pub(crate) fn refuse(site: AllocationSite, skip: usize) -> Guard {
    REFUSE.with(|cell| cell.set(Some((site, skip))));
    Guard(())
  }

  /// Clears the seam when dropped.
  pub(crate) struct Guard(());

  impl Drop for Guard {
    fn drop(&mut self) {
      REFUSE.with(|cell| cell.set(None));
    }
  }

  /// Whether the reservation at `site` refuses now.
  pub(crate) fn refuses(site: AllocationSite) -> bool {
    REFUSE.with(|cell| match cell.get() {
      Some((at, 0)) if at == site => true,
      Some((at, skip)) if at == site => {
        cell.set(Some((at, skip - 1)));
        false
      }
      _ => false,
    })
  }
}

#[cfg(not(test))]
mod inject {
  use super::AllocationSite;

  #[inline(always)]
  pub(super) const fn refuses(_: AllocationSite) -> bool {
    false
  }
}

#[cfg(test)]
mod tests;
