//! An image prepared for a vision function, on the CPU and on any thread:
//! the resize to the budget's geometry, then the five planes the function
//! reads.
//!
//! Hugging Face's `Gemma4ImageProcessor`, step for step:
//!
//! 1. **Geometry.** `get_aspect_ratio_preserving_size`, ported exactly: f64
//!    arithmetic, `floor` to multiples of 48 pixels (three patches), the
//!    zero-side edge cases, and the refusal when both sides round to zero.
//! 2. **Resize.** Antialiased bicubic on 8-bit RGB (PIL's `BICUBIC`, Keys
//!    `a = −0.5`), through pixon's `CatmullRom` q8 resampler; none at an equal
//!    size.
//! 3. **Rescale.** A 256-entry table, `f16(v / 255)`; the graph applies
//!    `2x − 1`.
//! 4. **The planes.** `patches` (patch `row · cols + col`, within it
//!    `(y · 16 + x) · 3 + c`), `positions` (`(col, row)`, −1 for padding),
//!    `valid`, `pool` (`1/9` over each 3 × 3 group, groups row-major) and
//!    `position_embeddings` (`table0[col] + table1[row]` in f32, rounded to
//!    f16; zero for padding).

use std::{path::Path, sync::Arc};

use half::{f16, slice::HalfFloatSliceExt};

use super::{
  super::{
    alloc::reserve_exact,
    artifacts::{
      ArtifactLabel, Part,
      stamp::{PartView, Stamp},
    },
    audio::AudioPrecision,
    config::{self, POOLING_KERNEL},
    error::{AllocationSite, BudgetNotLoaded, Error, ResizeFailure, Result},
    options::ImageOptions,
    table::PositionTable,
  },
  VisionBudget,
};
use crate::{
  DataType, MultiArray,
  embeddings::image::{ImageDimensions, Rgb8Image},
};

/// A source side past this is refused: pixon's q8 resampler is exact to PIL
/// below it, siglip's cap for the same reason.
pub(crate) const MAX_IMAGE_AXIS: usize = 1 << 20;

/// A patch's side in pixels.
const PATCH: usize = config::PATCH_SIZE;
/// A patch's values: 16 × 16 pixels of RGB.
pub(crate) const PATCH_VALUES: usize = PATCH * PATCH * 3;
/// A resized side is a multiple of this: one pooling group of patches.
const SIDE: usize = POOLING_KERNEL * PATCH;
/// The position tables' width, one row per column or row index.
pub(crate) const POSITION_WIDTH: usize = 768;

/// `f16(v / 255)` for every 8-bit value: the rescale, as a table.
pub(crate) static RESCALE: [f16; 256] = {
  let mut table = [f16::ZERO; 256];
  let mut v = 0;
  while v < 256 {
    table[v] = f16::from_f64_const(v as f64 / 255.0);
    v += 1;
  }
  table
};

/// The size an image of `height` × `width` is resized to for `budget`, as
/// `(height, width)`: Hugging Face's `get_aspect_ratio_preserving_size`,
/// ported exactly.
///
/// # Errors
/// [`Error::ImageDimensions`] when both sides round to zero, or when the
/// target would hold more patches than the budget.
pub(crate) fn target_size(
  height: usize,
  width: usize,
  budget: VisionBudget,
) -> Result<(usize, usize)> {
  let refuse = || Error::ImageDimensions(ImageDimensions::new(width, height));
  let max_patches = budget.patches();
  let target_px = max_patches * PATCH * PATCH;
  let factor = (target_px as f64 / (height as f64 * width as f64)).sqrt();
  let ideal_height = factor * height as f64;
  let ideal_width = factor * width as f64;
  let mut target_height = (ideal_height / SIDE as f64).floor() as usize * SIDE;
  let mut target_width = (ideal_width / SIDE as f64).floor() as usize * SIDE;
  if target_height == 0 && target_width == 0 {
    return Err(refuse());
  }
  let max_side = (max_patches / (POOLING_KERNEL * POOLING_KERNEL)) * SIDE;
  if target_height == 0 {
    target_height = SIDE;
    target_width = ((width as f64 / height as f64).floor() as usize * SIDE).min(max_side);
  } else if target_width == 0 {
    target_width = SIDE;
    target_height = ((height as f64 / width as f64).floor() as usize * SIDE).min(max_side);
  }
  if target_height * target_width > target_px {
    return Err(refuse());
  }
  Ok((target_height, target_width))
}

/// An image prepared for one budget's vision function: its five input planes,
/// allocated once per budget and refilled, and how many of its patches are
/// real.
///
/// A caller keeps one and refills it with
/// [`VisionArtifacts::prepare_into`], which then allocates nothing for an
/// image already at its budget's geometry; [`VisionArtifacts::prepare`] makes
/// a new one each call. `Send`: prepared on one thread, encoded on another.
#[derive(Debug)]
pub struct PreparedImage {
  /// The budget of the last preparation that succeeded; `None` before the
  /// first and after a failed one, so a failed preparation is never encoded.
  budget: Option<VisionBudget>,
  /// The budget the planes are allocated for.
  shaped: Option<VisionBudget>,
  real: usize,
  patches: Option<MultiArray>,
  positions: Option<MultiArray>,
  valid: Option<MultiArray>,
  pool: Option<MultiArray>,
  position_embeddings: Option<MultiArray>,
  resized: Vec<u8>,
}

impl Default for PreparedImage {
  fn default() -> Self {
    Self::new()
  }
}

impl PreparedImage {
  /// An empty image, to refill with [`VisionArtifacts::prepare_into`]; its
  /// planes are allocated at the first budget it is prepared for.
  pub const fn new() -> Self {
    Self {
      budget: None,
      shaped: None,
      real: 0,
      patches: None,
      positions: None,
      valid: None,
      pool: None,
      position_embeddings: None,
      resized: Vec::new(),
    }
  }

  /// The budget of the last preparation, if it succeeded.
  #[inline]
  pub const fn budget(&self) -> Option<VisionBudget> {
    self.budget
  }

  /// The real patches of the last preparation; the rest are padding.
  #[inline]
  pub const fn patches(&self) -> usize {
    self.real
  }

  /// The image tokens the real patches pool into: the patches over nine.
  #[inline]
  pub const fn tokens(&self) -> usize {
    self.real / (POOLING_KERNEL * POOLING_KERNEL)
  }

  /// The five planes, by the function's input names, once prepared.
  pub(crate) fn inputs(&self) -> Option<[(&'static str, &MultiArray); 5]> {
    Some([
      ("patches", self.patches.as_ref()?),
      ("pool", self.pool.as_ref()?),
      ("position_embeddings", self.position_embeddings.as_ref()?),
      ("positions", self.positions.as_ref()?),
      ("valid", self.valid.as_ref()?),
    ])
  }

  /// Allocates the planes for `budget`, unless they are already its. A
  /// failure part way leaves no shape recorded, so the next call allocates
  /// all five again.
  fn shape_for(&mut self, budget: VisionBudget) -> Result<()> {
    if self.shaped == Some(budget) {
      return Ok(());
    }
    self.shaped = None;
    let (n, t) = (budget.patches(), budget.tokens());
    let zeros = |shape: &[usize]| MultiArray::zeros(shape, DataType::F16);
    self.patches = Some(zeros(&[1, n, PATCH_VALUES])?);
    self.positions = Some(zeros(&[1, n, 2])?);
    self.valid = Some(zeros(&[1, n])?);
    self.pool = Some(zeros(&[t, n])?);
    self.position_embeddings = Some(zeros(&[1, n, POSITION_WIDTH])?);
    self.shaped = Some(budget);
    Ok(())
  }
}

/// What every vision instance and every preparing thread shares.
#[derive(Debug)]
pub(crate) struct Shared {
  pub(crate) view: PartView,
  table: PositionTable,
}

/// The vision part's read-only state: the position tables, mapped once, the
/// checked `config.json`, and what the directory's stamp says of the vision
/// bundle. A clone is a reference count, and preparing an image needs no
/// model, so any thread prepares while another encodes.
#[derive(Debug, Clone)]
pub struct VisionArtifacts {
  pub(crate) shared: Arc<Shared>,
}

impl VisionArtifacts {
  /// Opens the vision part of the stamped model directory `dir`: its stamp,
  /// `config.json` against the compiled contract, and the position tables
  /// against the sentinels the stamp recorded. Loads no model and writes
  /// nothing.
  ///
  /// # Errors
  /// [`Error::Unstamped`], [`Error::StaleStamp`] or [`Error::Stamp`] from the
  /// stamp; [`Error::Read`] or [`Error::Config`] for `config.json` (past its
  /// 1 MiB cap included); [`Error::Traversal`] for a stamped file or
  /// `config.json` that is now a broken link; [`Error::Allocation`] if a
  /// document's buffer cannot be reserved; [`Error::TableShape`] or
  /// [`Error::TableIdentity`] for the tables.
  pub fn open(dir: impl AsRef<Path>) -> Result<Self> {
    let dir = dir.as_ref();
    let view = Stamp::read(dir)?.part(dir, Part::Vision, AudioPrecision::Fp16)?;
    config::load(&dir.join("config.json"))?;
    let table = PositionTable::open(&dir.join("position_embeddings.f16"), view.sentinels())?;
    Ok(Self {
      shared: Arc::new(Shared { view, table }),
    })
  }

  /// The label of the vision artifacts, from the stamp.
  #[inline]
  pub fn artifact_label(&self) -> ArtifactLabel {
    self.shared.view.label()
  }

  /// Prepares `image` for `budget` (`None`: the options' default) into a new
  /// [`PreparedImage`]. The convenience form of [`Self::prepare_into`]: about
  /// 9.2 MB of planes at 280, allocated each call.
  ///
  /// # Errors
  /// As [`Self::prepare_into`].
  pub fn prepare(
    &self,
    image: Rgb8Image<'_>,
    budget: Option<VisionBudget>,
    options: &ImageOptions,
  ) -> Result<PreparedImage> {
    let mut out = PreparedImage::new();
    self.prepare_into(image, budget, options, &mut out)?;
    Ok(out)
  }

  /// Prepares `image` for `budget` (`None`: the options' default) into `out`,
  /// refilling its planes. For an image already at its budget's geometry,
  /// once `out` holds that budget's planes, this allocates nothing; a resize
  /// allocates pixon's resample plan and the resized pixels' buffer as it
  /// grows. On an error, `out` holds no preparation: its budget is `None`,
  /// and [`ImageEncoder::encode`](super::ImageEncoder::encode) refuses it.
  ///
  /// # Errors
  /// [`Error::BudgetNotLoaded`] for a budget outside `options`' budgets;
  /// [`Error::ImageDimensions`] for a side past 2²⁰ or a target with both
  /// sides zero; [`Error::Resize`] if pixon refuses the resize;
  /// [`Error::Allocation`] or [`Error::Tensor`] for the planes.
  pub fn prepare_into(
    &self,
    image: Rgb8Image<'_>,
    budget: Option<VisionBudget>,
    options: &ImageOptions,
    out: &mut PreparedImage,
  ) -> Result<()> {
    out.budget = None;
    out.real = 0;
    let budget = budget.unwrap_or(options.default_budget());
    if !options.budgets().contains(budget) {
      return Err(Error::BudgetNotLoaded(BudgetNotLoaded::new(
        budget,
        options.budgets(),
      )));
    }
    let (height, width) = (image.height(), image.width());
    if height > MAX_IMAGE_AXIS || width > MAX_IMAGE_AXIS {
      return Err(Error::ImageDimensions(ImageDimensions::new(width, height)));
    }
    let (target_height, target_width) = target_size(height, width, budget)?;
    out.shape_for(budget)?;
    let PreparedImage {
      real,
      patches,
      positions,
      valid,
      pool,
      position_embeddings,
      resized,
      ..
    } = out;
    let pixels: &[u8] = if (target_height, target_width) == (height, width) {
      image.data()
    } else {
      resize(image, target_height, target_width, resized)?;
      &resized[..]
    };
    let (rows, cols) = (target_height / PATCH, target_width / PATCH);
    *real = rows * cols;
    let planes = Planes {
      patches: patches.as_mut().expect("shaped").as_slice_mut::<f16>()?,
      positions: positions.as_mut().expect("shaped").as_slice_mut::<f16>()?,
      valid: valid.as_mut().expect("shaped").as_slice_mut::<f16>()?,
      pool: pool.as_mut().expect("shaped").as_slice_mut::<f16>()?,
      position_embeddings: position_embeddings
        .as_mut()
        .expect("shaped")
        .as_slice_mut::<f16>()?,
    };
    fill(pixels, rows, cols, &self.shared.table, planes);
    out.budget = Some(budget);
    Ok(())
  }
}

/// Resizes `image` to `height` × `width` into `out`: pixon's `CatmullRom` q8
/// resampler, PIL's antialiased `BICUBIC` on 8-bit RGB.
fn resize(image: Rgb8Image<'_>, height: usize, width: usize, out: &mut Vec<u8>) -> Result<()> {
  use pixon::{Convert, frame::Rgb24Frame, resample::CatmullRom};

  let len = height * width * 3;
  out.clear();
  reserve_exact(out, len, AllocationSite::ImagePlanes)?;
  out.resize(len, 0);
  // Both sides are at most 2²⁰, inside pixon's `u32` geometry.
  let (w, h) = (image.width() as u32, image.height() as u32);
  let frame = Rgb24Frame::try_new(image.data(), w, h, w * 3)
    .map_err(|_| Error::ImageDimensions(ImageDimensions::new(image.width(), image.height())))?;
  Convert::from(&frame)
    .resize_with(CatmullRom, width, height)
    .rgb(out)
    .run()
    .map_err(|e| Error::Resize(ResizeFailure::new(e)))
}

/// The five planes of one budget, borrowed for a refill.
pub(crate) struct Planes<'a> {
  pub(crate) patches: &'a mut [f16],
  pub(crate) positions: &'a mut [f16],
  pub(crate) valid: &'a mut [f16],
  pub(crate) pool: &'a mut [f16],
  pub(crate) position_embeddings: &'a mut [f16],
}

/// Fills the planes from `pixels`, a `rows · 16` × `cols · 16` RGB image, and
/// zeroes the padding. Each plane holds the budget's `n` patches; the first
/// `rows · cols` are real.
pub(crate) fn fill(
  pixels: &[u8],
  rows: usize,
  cols: usize,
  table: &PositionTable,
  planes: Planes<'_>,
) {
  let width = cols * PATCH * 3;
  let real = rows * cols;
  let groups_per_row = cols / POOLING_KERNEL;
  let ninth = f16::from_f64_const(1.0 / 9.0);
  let n = planes.valid.len();
  planes.pool.fill(f16::ZERO);
  planes.patches[real * PATCH_VALUES..].fill(f16::ZERO);
  planes.position_embeddings[real * POSITION_WIDTH..].fill(f16::ZERO);
  planes.positions[real * 2..].fill(f16::from_f32_const(-1.0));
  planes.valid[..real].fill(f16::ONE);
  planes.valid[real..].fill(f16::ZERO);
  let (mut x_row, mut y_row) = ([0f32; POSITION_WIDTH], [0f32; POSITION_WIDTH]);
  let mut halves = [f16::ZERO; POSITION_WIDTH];
  for row in 0..rows {
    decode_row(
      table.row(1, row).expect("a row inside the table"),
      &mut halves,
    );
    halves.convert_to_f32_slice(&mut y_row);
    for col in 0..cols {
      let p = row * cols + col;
      let patch = &mut planes.patches[p * PATCH_VALUES..(p + 1) * PATCH_VALUES];
      for y in 0..PATCH {
        let start = (row * PATCH + y) * width + col * PATCH * 3;
        let source = &pixels[start..start + PATCH * 3];
        for (cell, &value) in patch[y * PATCH * 3..(y + 1) * PATCH * 3]
          .iter_mut()
          .zip(source)
        {
          *cell = RESCALE[usize::from(value)];
        }
      }
      planes.positions[p * 2] = f16::from_f32(col as f32);
      planes.positions[p * 2 + 1] = f16::from_f32(row as f32);
      let group = col / POOLING_KERNEL + groups_per_row * (row / POOLING_KERNEL);
      planes.pool[group * n + p] = ninth;
      decode_row(
        table.row(0, col).expect("a column inside the table"),
        &mut halves,
      );
      halves.convert_to_f32_slice(&mut x_row);
      for (x, y) in x_row.iter_mut().zip(&y_row) {
        *x += *y;
      }
      planes.position_embeddings[p * POSITION_WIDTH..(p + 1) * POSITION_WIDTH]
        .convert_from_f32_slice(&x_row);
    }
  }
}

/// A row of the position tables, little-endian f16 bytes, as halves.
fn decode_row(bytes: &[u8], out: &mut [f16; POSITION_WIDTH]) {
  for (half, pair) in out.iter_mut().zip(bytes.as_chunks::<2>().0) {
    *half = f16::from_bits(u16::from_le_bytes(*pair));
  }
}

#[cfg(test)]
mod tests;
