use super::*;
use crate::window::{DEFAULT_STEP_SAMPLES, chunk_starts};

// =====================================================================
// Hermetic: the geometry — argmax's chunk/window grid
// =====================================================================

/// The constants the whole mapping is derived from, re-proved against the
/// pinned model contract (`tests/argmax_model_io.rs`) rather than trusted.
#[test]
fn derived_geometry_matches_the_model_contract() {
  assert_eq!(ARGMAX_WINDOW_STRIDE_SAMPLES, 16_000, "Seg.swift:110-112");
  assert_eq!(ARGMAX_CHUNK_STRIDE_OFFSET, 144_000, "Seg.swift:114-116");
  assert_eq!(ARGMAX_CHUNK_HOP_SAMPLES, 336_000, "Seg.swift:168 (21 s)");
  assert_eq!(ARGMAX_MASK_FRAMES, 1767, "Emb.swift:56 framesPerChunk");
  // The stride the graph bakes in IS the crate's default step: this is what
  // lets both sources share one chunk grid.
  assert_eq!(ARGMAX_WINDOW_STRIDE_SAMPLES, DEFAULT_STEP_SAMPLES as usize);
  assert_eq!(seconds_per_window(), 10.0);
  assert_eq!(seconds_per_stride(), 1.0);
}

/// The grid theorem's unstated PREMISE, made explicit: this port implements
/// argmax's `useFullRedundancy == true` branch (`Seg.swift:146`), which is its
/// own default (`Seg.swift:26`).
///
/// The hop is a HOST-side choice — nothing in the model I/O declares it — so it
/// cannot be validated against the loaded model. It is pinned in constants
/// instead, and this test says out loud what the module-bottom `const _` is
/// really asserting.
#[test]
fn the_grid_theorem_requires_argmaxs_full_redundancy() {
  // `Seg.swift:146`'s TRUE branch: chunkStrideOffset = modelChunkStrideOffset
  // = windowLength - windowStride (`Seg.swift:110-116`).
  assert_eq!(
    ARGMAX_CHUNK_STRIDE_OFFSET,
    ARGMAX_WINDOW_SAMPLES - ARGMAX_WINDOW_STRIDE_SAMPLES,
    "this port ports the useFullRedundancy == true branch"
  );
  // ...which makes the chunk hop an exact multiple of the window stride — THE
  // premise of `c = k * 21 + w`.
  assert_eq!(
    ARGMAX_CHUNK_HOP_SAMPLES,
    ARGMAX_WINDOWS_PER_CHUNK * ARGMAX_WINDOW_STRIDE_SAMPLES
  );

  // The FALSE branch (chunkStrideOffset = 0) would break it: the hop becomes a
  // whole chunk, no longer a multiple of the 16 000-sample stride, so
  // consecutive chunks' window grids stop abutting. Window starts within chunk
  // k would run to k*480_000 + 20*16_000 = k*480_000 + 320_000, and the next
  // chunk would begin at (k+1)*480_000 — leaving grid points
  // k*480_000 + 336_000 .. (k+1)*480_000 covered by NO window.
  let no_redundancy_hop = ARGMAX_CHUNK_SAMPLES; // offset 0
  assert_ne!(
    no_redundancy_hop,
    ARGMAX_WINDOWS_PER_CHUNK * ARGMAX_WINDOW_STRIDE_SAMPLES,
    "if argmax ever defaulted useFullRedundancy to false, this port's chunk \
     grid would develop holes and `c = k*21 + w` would be false — the \
     module-bottom const-assert is what catches that"
  );
}

/// The `f32` [`window_start_frame`] and the integer form `w * 589 / 10` agree
/// on every window of the used domain — asserted, not assumed.
///
/// This is why a mutation swapping one for the other survives the suite: the
/// two are genuinely EQUIVALENT here (binary32's `58.9` rounds UP, so the
/// truncation never falls short at `w = 10` / `w = 20`, the only two points
/// where the exact product lands on an integer). Pinning it means the day
/// argmax's window geometry changes and they diverge, THIS test says so
/// instead of the difference passing silently.
#[test]
fn window_start_frame_agrees_with_the_integer_form() {
  for w in 0..ARGMAX_WINDOWS_PER_CHUNK {
    assert_eq!(
      window_start_frame(w),
      w * ARGMAX_FRAMES_PER_WINDOW / 10,
      "w={w}: the f32 and integer forms must agree across the used domain"
    );
  }
}

/// `Emb.swift:58-75`'s `chunkIndices` start frames, pinned exactly. Verified
/// independently against the real model's own geometry (the probe recorded
/// this identical sequence). The `f32` reproduction is load-bearing: `58.9`
/// rounds UP in binary32, which is what keeps `trunc(10 * s) == 589` and
/// `trunc(20 * s) == 1178` rather than 588 / 1177.
#[test]
fn window_start_frames_match_argmax_timeline() {
  let got: Vec<usize> = (0..ARGMAX_WINDOWS_PER_CHUNK)
    .map(window_start_frame)
    .collect();
  assert_eq!(
    got,
    vec![
      0, 58, 117, 176, 235, 294, 353, 412, 471, 530, 589, 647, 706, 765, 824, 883, 942, 1001, 1060,
      1119, 1178
    ]
  );
  // The timeline is exactly covered and never overrun.
  let last = got[ARGMAX_WINDOWS_PER_CHUNK - 1] + ARGMAX_FRAMES_PER_WINDOW - 1;
  assert_eq!(last, ARGMAX_MASK_FRAMES - 1, "last frame must be 1766");
  // Strictly increasing, and every window fits.
  for w in 0..ARGMAX_WINDOWS_PER_CHUNK {
    assert!(got[w] + ARGMAX_FRAMES_PER_WINDOW <= ARGMAX_MASK_FRAMES);
    if w > 0 {
      assert!(got[w] > got[w - 1]);
    }
  }
}

/// `Seg.swift:147-153`'s chunk loop: 30 s chunks at a 21 s hop, with the
/// truncated chunk always LAST (so every start is exactly `k * 336_000`).
#[test]
fn argmax_chunk_starts_match_the_swift_loop() {
  // Shorter than one chunk → a single (zero-padded) chunk.
  assert_eq!(argmax_chunk_starts(1), vec![0]);
  assert_eq!(argmax_chunk_starts(160_000), vec![0]);
  assert_eq!(argmax_chunk_starts(479_999), vec![0]);
  // Exactly one chunk.
  assert_eq!(argmax_chunk_starts(480_000), vec![0]);
  // Just past → a second chunk at the 21 s hop.
  assert_eq!(argmax_chunk_starts(480_001), vec![0, 336_000]);
  assert_eq!(argmax_chunk_starts(640_000), vec![0, 336_000]);
  assert_eq!(argmax_chunk_starts(816_000), vec![0, 336_000]);
  assert_eq!(argmax_chunk_starts(816_001), vec![0, 336_000, 672_000]);
  assert_eq!(argmax_chunk_starts(960_000), vec![0, 336_000, 672_000]);

  // Every start is k * hop, for a wide sweep.
  for total in (1..2_000_000).step_by(9_973) {
    for (k, &start) in argmax_chunk_starts(total).iter().enumerate() {
      assert_eq!(start, k * ARGMAX_CHUNK_HOP_SAMPLES, "total={total} k={k}");
    }
  }
}

/// `Emb.swift:120-130`'s `bounded(windowIdx:)`, in exact sample arithmetic.
#[test]
fn window_bounded_matches_argmax_predicate() {
  // A full 30 s chunk: every window is bounded.
  for w in 0..ARGMAX_WINDOWS_PER_CHUNK {
    assert!(window_bounded(w, ARGMAX_CHUNK_SAMPLES), "w={w}");
  }
  // Window 0 is ALWAYS bounded, however short the chunk (Emb.swift:124).
  assert!(window_bounded(0, 1));
  assert!(!window_bounded(1, 1));

  // 18 s chunk: bounded ⟺ w * 16_000 + 144_000 < 288_000 ⟺ w < 9.
  let len = 18 * 16_000;
  for w in 0..ARGMAX_WINDOWS_PER_CHUNK {
    assert_eq!(window_bounded(w, len), w < 9, "w={w}");
  }
  // Exactly 10 s: only window 0 (the boundary is EXCLUSIVE — `<`, not `<=`).
  let len = 10 * 16_000;
  for w in 1..ARGMAX_WINDOWS_PER_CHUNK {
    assert!(!window_bounded(w, len), "w={w}");
  }
  // One sample more, and window 1 becomes bounded.
  assert!(window_bounded(1, 10 * 16_000 + 1));
}

/// **THE theorem** the whole index mapping rests on (module doc): the set of
/// argmax's BOUNDED windows, mapped to absolute sample starts, is EXACTLY
/// dia's own offline chunk grid — so the two sources agree on `num_chunks`,
/// chunk-for-chunk, sample-for-sample. Swept exhaustively over lengths
/// spanning every boundary case (sub-window, sub-chunk, chunk multiples,
/// truncated final chunks, multi-chunk).
#[test]
fn bounded_window_grid_equals_dia_chunk_grid() {
  let options = WindowOptions::new();
  let mut totals: Vec<usize> = vec![
    1, 2, 159_999, 160_000, 160_001, 335_999, 336_000, 336_001, 479_999, 480_000, 480_001, 639_999,
    640_000, 650_000, 656_000, 657_000, 815_999, 816_000, 816_001, 959_999, 960_000, 960_001,
  ];
  // Dense sweep across several chunk periods, plus every ±1 around each
  // 16_000-sample boundary in the first two chunks (where a grid off-by-one
  // would hide).
  totals.extend((1..1_500_000).step_by(7_919));
  for c in 0..64 {
    let base = c * ARGMAX_WINDOW_STRIDE_SAMPLES;
    totals.extend([base.saturating_sub(1), base, base + 1]);
  }
  totals.retain(|&t| t > 0);

  for total in totals {
    // What argmax's own geometry yields, window by window.
    let mut argmax_starts: Vec<usize> = Vec::new();
    for (k, &chunk_start) in argmax_chunk_starts(total).iter().enumerate() {
      let chunk_len = (total - chunk_start).min(ARGMAX_CHUNK_SAMPLES);
      for w in 0..ARGMAX_WINDOWS_PER_CHUNK {
        if window_bounded(w, chunk_len) {
          argmax_starts.push(chunk_start + w * ARGMAX_WINDOW_STRIDE_SAMPLES);
          // The global chunk index and the absolute start agree.
          assert_eq!(
            global_chunk(k, w) * ARGMAX_WINDOW_STRIDE_SAMPLES,
            chunk_start + w * ARGMAX_WINDOW_STRIDE_SAMPLES,
            "total={total} k={k} w={w}: c*stride must be the absolute start"
          );
        }
      }
    }

    let dia_starts = chunk_starts(total, &options);
    assert_eq!(
      argmax_starts, dia_starts,
      "total={total}: argmax's bounded-window grid must equal dia's chunk grid"
    );
    // Consequence: `c = k*21 + w` is a bijection onto `0..num_chunks`.
    assert_eq!(
      argmax_starts.len(),
      dia_starts.len(),
      "total={total}: num_chunks must agree"
    );
  }
}

/// The global chunk index identity, stated on its own: `c = k * 21 + w`, and
/// it is exactly `absolute_start / stride`. Guards the constant `21` in
/// [`global_chunk`] against being "simplified" to anything else.
#[test]
fn global_chunk_index_is_k_times_windows_plus_w() {
  for k in 0..8 {
    for w in 0..ARGMAX_WINDOWS_PER_CHUNK {
      let c = global_chunk(k, w);
      assert_eq!(c, k * ARGMAX_WINDOWS_PER_CHUNK + w);
      let absolute = k * ARGMAX_CHUNK_HOP_SAMPLES + w * ARGMAX_WINDOW_STRIDE_SAMPLES;
      assert_eq!(c * ARGMAX_WINDOW_STRIDE_SAMPLES, absolute);
    }
  }
  // Consecutive chunks abut with no gap and no overlap: the last window of
  // chunk k and the first of chunk k+1 are adjacent grid points.
  for k in 0..8 {
    assert_eq!(
      global_chunk(k, ARGMAX_WINDOWS_PER_CHUNK - 1) + 1,
      global_chunk(k + 1, 0)
    );
  }
}

// =====================================================================
// Hermetic: the activity gate
// =====================================================================

/// `Emb.swift:98-103`: STRICTLY more than 2 active frames. The boundary is
/// the whole point — 2 is out, 3 is in.
#[test]
fn activity_gate_excludes_two_frames_includes_three() {
  assert_eq!(active_speakers(&[0.0, 2.0, 3.0]), [false, false, true]);
  assert_eq!(active_speakers(&[1.0, 2.0, 2.5]), [false, false, true]);
  assert_eq!(active_speakers(&[589.0, 3.0, 0.0]), [true, true, false]);
  // The literal Swift form (`a * spf > 2 * spf`) must agree with `a > 2` on
  // every reachable frame count — `speaker_activity` is an exact integer
  // count in [0, 589].
  for a in 0..=ARGMAX_FRAMES_PER_WINDOW {
    let got = active_speakers(&[a as f32, 0.0, 0.0])[0];
    assert_eq!(got, a > 2, "activity={a}");
  }
}

// =====================================================================
// Hermetic: the mask scatter (Emb.swift:216-253)
// =====================================================================

/// A hand-built argmax-shaped tensor set: `ids[w][f][s] = 1` exactly where
/// `pred(w, f, s)`, `overlapped[w][f] = 1` exactly where `overlap(w, f)`,
/// and `activity` derived from `ids` the way the real model derives it
/// (verified 63/63 on the real model — module doc).
fn build_decoded(
  pred: impl Fn(usize, usize, usize) -> bool,
  overlap: impl Fn(usize, usize) -> bool,
) -> DecodedChunk {
  let mut ids =
    vec![0.0f32; ARGMAX_WINDOWS_PER_CHUNK * ARGMAX_FRAMES_PER_WINDOW * ARGMAX_NUM_SPEAKERS];
  let mut overlapped = vec![0.0f32; ARGMAX_WINDOWS_PER_CHUNK * ARGMAX_FRAMES_PER_WINDOW];
  let mut activity = vec![0.0f32; ARGMAX_WINDOWS_PER_CHUNK * ARGMAX_NUM_SPEAKERS];
  for w in 0..ARGMAX_WINDOWS_PER_CHUNK {
    for f in 0..ARGMAX_FRAMES_PER_WINDOW {
      if overlap(w, f) {
        overlapped[overlapped_index(w, f)] = 1.0;
      }
      for s in 0..ARGMAX_NUM_SPEAKERS {
        if pred(w, f, s) {
          ids[ids_index(w, f, s)] = 1.0;
          activity[w * ARGMAX_NUM_SPEAKERS + s] += 1.0;
        }
      }
    }
  }
  DecodedChunk {
    ids,
    activity,
    overlapped,
  }
}

fn mask_at(masks: &[f16], row: usize, frame: usize) -> f32 {
  f32::from(masks[row * ARGMAX_MASK_FRAMES + frame])
}

/// Row `w*3+s` carries window `w`'s 589 frames at `start_frame(w) + f`, and
/// NOTHING outside that span. This is the test that dies if the window→row
/// or frame→timeline index is broken.
#[test]
fn mask_row_places_window_w_at_its_start_frame() {
  // Speaker 1 active on every frame of every window; no overlap.
  let d = build_decoded(|_, _, s| s == 1, |_, _| false);
  let plans = window_plans(ARGMAX_CHUNK_SAMPLES, &d.activity);
  let masks = build_speaker_masks(&d.ids, &d.overlapped, &plans);

  for w in 0..ARGMAX_WINDOWS_PER_CHUNK {
    let row = w * ARGMAX_NUM_SPEAKERS + 1;
    let start = window_start_frame(w);
    for frame in 0..ARGMAX_MASK_FRAMES {
      let in_span = frame >= start && frame < start + ARGMAX_FRAMES_PER_WINDOW;
      let expected = if in_span { 1.0 } else { 0.0 };
      assert_eq!(
        mask_at(&masks, row, frame),
        expected,
        "row {row} (w={w}, s=1), frame {frame}: span is [{start}, {})",
        start + ARGMAX_FRAMES_PER_WINDOW
      );
    }
  }
  // Rows for the INACTIVE speakers 0 and 2 stay entirely zero.
  for w in 0..ARGMAX_WINDOWS_PER_CHUNK {
    for s in [0usize, 2] {
      assert!(mask_row_is_zero(&masks, w * ARGMAX_NUM_SPEAKERS + s));
    }
  }
  // Slot 63 (the padding row) is zero.
  assert!(mask_row_is_zero(&masks, ARGMAX_MASK_SLOTS - 1));
}

/// The mask VALUE is the overlap-excluded `speaker_ids * (1 - overlapped)`
/// (`Emb.swift:245`): an overlap frame contributes 0 even where the speaker is
/// active. This is the NON-fallback branch — 489 clean frames survive here,
/// far more than `EXCLUDE_OVERLAP_MIN_FRAMES`, so `dia`'s fallback does not
/// fire and argmax's clean mask is used as-is
/// (`too_few_clean_frames_falls_back_to_the_raw_mask` covers the other branch).
#[test]
fn mask_value_is_ids_times_one_minus_overlap() {
  // Speaker 1 active on every frame; frames 100..200 are overlap frames.
  let d = build_decoded(|_, _, s| s == 1, |_, f| (100..200).contains(&f));
  let plans = window_plans(ARGMAX_CHUNK_SAMPLES, &d.activity);
  let masks = build_speaker_masks(&d.ids, &d.overlapped, &plans);

  let w = 3usize;
  let row = w * ARGMAX_NUM_SPEAKERS + 1;
  let start = window_start_frame(w);
  let mut clean_count = 0usize;
  for f in 0..ARGMAX_FRAMES_PER_WINDOW {
    let expected = if (100..200).contains(&f) { 0.0 } else { 1.0 };
    assert_eq!(mask_at(&masks, row, start + f), expected, "w={w} f={f}");
    if expected != 0.0 {
      clean_count += 1;
    }
  }
  assert_eq!(clean_count, ARGMAX_FRAMES_PER_WINDOW - 100);
  assert!(
    clean_count > EXCLUDE_OVERLAP_MIN_FRAMES,
    "precondition: dia's fallback must NOT fire in this test"
  );
}

// =====================================================================
// Hermetic: dia's exclude-overlap FALLBACK on argmax's tensors
// (owned.rs:573-591 — see the module doc)
// =====================================================================

/// The mask row slot `(0, 1)` ends up with, given `active` active frames of
/// which frames `0..overlap_upto` are ALSO overlap frames (so the clean count
/// is `active - overlap_upto`).
fn fallback_case(active: usize, overlap_upto: usize) -> Vec<f32> {
  let d = build_decoded(
    move |_, f, s| s == 1 && f < active,
    move |_, f| f < overlap_upto,
  );
  let plans = window_plans(ARGMAX_CHUNK_SAMPLES, &d.activity);
  assert!(
    plans[0].active[1],
    "active={active}: the slot must clear the `> 2` activity gate"
  );
  let masks = build_speaker_masks(&d.ids, &d.overlapped, &plans);
  let start = window_start_frame(0);
  (0..ARGMAX_FRAMES_PER_WINDOW)
    .map(|f| mask_at(&masks, 1, start + f))
    .collect()
}

/// The RAW `speaker_ids` mask for `active` leading active frames.
fn raw_mask(active: usize) -> Vec<f32> {
  (0..ARGMAX_FRAMES_PER_WINDOW)
    .map(|f| f32::from(u8::from(f < active)))
    .collect()
}

/// **THE fix** (module doc): a slot with too few CLEAN frames pools over its
/// RAW `speaker_ids` mask instead of the sparse overlap-excluded one — `dia`'s
/// per-slot fallback (`owned.rs:589`), not argmax's unconditional clean mask.
///
/// The boundary is `clean_count <= EXCLUDE_OVERLAP_MIN_FRAMES` (= 2): 0, 1 and
/// 2 clean frames fall back; 3 keeps the clean mask. **Flipping `<=` to `<` in
/// `build_speaker_masks` makes the `clean == 2` case below fail** — it would
/// keep a 2-frame clean mask where dia's rule demands the raw one. That is what
/// pins the boundary rather than merely exercising it.
#[test]
fn too_few_clean_frames_falls_back_to_the_raw_mask() {
  // clean_count = 0 — every active frame is an overlap frame. This is the case
  // that used to produce an all-zero mask row (and the degenerate norm-0.5356
  // embedding); it now falls back to all 10 raw frames.
  assert_eq!(
    fallback_case(10, 10),
    raw_mask(10),
    "clean=0 must fall back"
  );
  // clean_count = 1.
  assert_eq!(fallback_case(10, 9), raw_mask(10), "clean=1 must fall back");
  // clean_count = 2 — THE `<=` boundary.
  assert_eq!(
    fallback_case(10, 8),
    raw_mask(10),
    "clean=2 must fall back (`<=`, not `<`)"
  );

  // clean_count = 3 — strictly more than the threshold, so the overlap-excluded
  // mask is KEPT: frames 0..7 are excluded, 7..10 survive.
  let clean3 = fallback_case(10, 7);
  assert_eq!(
    clean3.iter().filter(|&&v| v != 0.0).count(),
    3,
    "clean=3 must keep the 3-frame overlap-excluded mask"
  );
  for (f, &got) in clean3.iter().enumerate() {
    let expected = f32::from(u8::from((7..10).contains(&f)));
    assert_eq!(got, expected, "clean=3 must keep the clean mask, f={f}");
  }
  assert_ne!(clean3, raw_mask(10), "clean=3 must NOT fall back");
}

/// The unreachability property itself, swept: for EVERY (active, overlap)
/// combination that clears the activity gate, the mask row is non-empty — and
/// its nonzero count is exactly what dia's rule prescribes.
///
/// This is the property `place_embeddings` `debug_assert!`s instead of
/// guarding, and it is why the bespoke all-zero-mask DROP guard an earlier
/// revision carried is gone.
#[test]
fn every_gated_in_slot_gets_a_non_empty_mask() {
  for active in 3..=12usize {
    // `active > 2` clears the gate.
    for overlap_upto in 0..=active {
      let got = fallback_case(active, overlap_upto);
      let nonzero = got.iter().filter(|&&v| v != 0.0).count();
      let clean = active - overlap_upto;
      let expected = if clean > EXCLUDE_OVERLAP_MIN_FRAMES {
        clean // the overlap-excluded mask
      } else {
        active // dia's fallback: the raw mask
      };
      assert_eq!(
        nonzero, expected,
        "active={active} overlap_upto={overlap_upto} (clean={clean})"
      );
      assert!(
        nonzero > 0,
        "active={active} overlap_upto={overlap_upto}: an ACTIVE slot's mask \
         row can never be empty"
      );
    }
  }
}

/// `Emb.swift:233-236`: a window with NO active speaker is skipped wholesale
/// — all three of its rows stay zero, even for a speaker with 1-2 frames.
#[test]
fn window_with_no_active_speaker_leaves_every_row_zero() {
  // Window 5's speaker 0 has exactly 2 frames (below the gate); nobody else
  // is ever active.
  let d = build_decoded(|w, f, s| w == 5 && s == 0 && f < 2, |_, _| false);
  let plans = window_plans(ARGMAX_CHUNK_SAMPLES, &d.activity);
  assert_eq!(d.activity[5 * ARGMAX_NUM_SPEAKERS], 2.0);
  assert!(!plans[5].any_active(), "2 frames must not clear the gate");

  let masks = build_speaker_masks(&d.ids, &d.overlapped, &plans);
  for row in 0..ARGMAX_MASK_SLOTS {
    assert!(mask_row_is_zero(&masks, row), "row {row}");
  }
}

/// A window's mask is built even when the window is UNBOUNDED
/// (`Emb.swift:230` has no `bounded` check; `Emb.swift:285` gates only the
/// read-back) — the row is computed and then discarded.
#[test]
fn masks_are_built_for_unbounded_windows_too() {
  let d = build_decoded(|_, _, s| s == 1, |_, _| false);
  // A 12 s chunk: windows >= 3 are unbounded.
  let plans = window_plans(12 * 16_000, &d.activity);
  assert!(plans[2].bounded);
  assert!(!plans[3].bounded);
  let masks = build_speaker_masks(&d.ids, &d.overlapped, &plans);
  // Window 20 is unbounded, yet its mask row is still populated.
  assert!(!mask_row_is_zero(&masks, 20 * ARGMAX_NUM_SPEAKERS + 1));
}

// =====================================================================
// Hermetic: segmentations + embedding placement (the Extraction mapping)
// =====================================================================

/// Allocates the two Extraction buffers for `num_chunks` chunks.
fn buffers(num_chunks: usize) -> (Vec<f64>, Vec<f32>) {
  (
    vec![0.0f64; num_chunks * ARGMAX_FRAMES_PER_WINDOW * SEG_NUM_SLOTS],
    vec![0.0f32; num_chunks * SEG_NUM_SLOTS * EMBEDDING_DIM],
  )
}

/// `segmentations[c][f][s]` carries `speaker_ids[w][f][s]` at `c = k*21 + w`
/// — for ACTIVE slots only. This is the test that dies if the chunk index or
/// the `[c][f][s]` stride is broken.
#[test]
fn segmentations_carry_speaker_ids_at_the_mapped_chunk() {
  // Speaker 1 active on frames [f == w] only... make it richer: speaker 1
  // active on frames f where f % 7 == w % 7, so each window has a DIFFERENT
  // pattern (a broken w index would copy the wrong window's pattern).
  let d = build_decoded(|w, f, s| s == 1 && f % 7 == w % 7, |_, _| false);
  let plans = window_plans(ARGMAX_CHUNK_SAMPLES, &d.activity);

  let num_chunks = 2 * ARGMAX_WINDOWS_PER_CHUNK;
  let (mut segs, _) = buffers(num_chunks);
  let k = 1usize; // NOT chunk 0 — a dropped `k` term would still pass at k=0.
  for (w, plan) in plans.iter().enumerate() {
    write_segmentations(global_chunk(k, w), w, &d.ids, plan, &mut segs);
  }

  for w in 0..ARGMAX_WINDOWS_PER_CHUNK {
    let c = k * ARGMAX_WINDOWS_PER_CHUNK + w;
    for f in 0..ARGMAX_FRAMES_PER_WINDOW {
      for s in 0..SEG_NUM_SLOTS {
        let got = segs[(c * ARGMAX_FRAMES_PER_WINDOW + f) * SEG_NUM_SLOTS + s];
        let expected = if s == 1 && f % 7 == w % 7 { 1.0 } else { 0.0 };
        assert_eq!(got, expected, "c={c} (k={k}, w={w}) f={f} s={s}");
      }
    }
  }
  // Chunk 0's slab (k=0, never written) is untouched.
  assert!(
    segs[..ARGMAX_WINDOWS_PER_CHUNK * ARGMAX_FRAMES_PER_WINDOW * SEG_NUM_SLOTS]
      .iter()
      .all(|&v| v == 0.0)
  );
}

/// An INACTIVE slot's segmentation column is all-zero even though it has 1-2
/// active frames in `speaker_ids` — argmax's activity gate is strictly more
/// aggressive than dia's, and `Extraction`'s invariant demands the column
/// match the (zero) embedding row.
#[test]
fn inactive_slot_segmentation_column_is_zero() {
  // Speaker 0: 2 frames (gated out). Speaker 1: every frame (active).
  let d = build_decoded(|_, f, s| (s == 0 && f < 2) || s == 1, |_, _| false);
  let plans = window_plans(ARGMAX_CHUNK_SAMPLES, &d.activity);
  assert_eq!(plans[0].active, [false, true, false]);

  let (mut segs, _) = buffers(ARGMAX_WINDOWS_PER_CHUNK);
  for (w, plan) in plans.iter().enumerate() {
    write_segmentations(global_chunk(0, w), w, &d.ids, plan, &mut segs);
  }
  // Slot 0's column is zero everywhere despite ids[w][0..2][0] == 1.
  for c in 0..ARGMAX_WINDOWS_PER_CHUNK {
    for f in 0..ARGMAX_FRAMES_PER_WINDOW {
      assert_eq!(
        segs[(c * ARGMAX_FRAMES_PER_WINDOW + f) * SEG_NUM_SLOTS],
        0.0,
        "c={c} f={f}: gated-out slot 0 must stay zero"
      );
    }
  }
}

/// A synthetic `[64, 256]` embedder output whose row `r` is the constant
/// `(r + 1) as f32` — so a mis-indexed row is unmistakable.
fn synthetic_embeddings() -> Vec<f32> {
  let mut e = vec![0.0f32; ARGMAX_MASK_SLOTS * EMBEDDING_DIM];
  for row in 0..ARGMAX_MASK_SLOTS {
    for d in 0..EMBEDDING_DIM {
      e[row * EMBEDDING_DIM + d] = (row + 1) as f32;
    }
  }
  e
}

/// `raw_embeddings[c = k*21+w][s][..] <- speaker_embeddings[0][w*3+s][..]`:
/// the 64→(21×3) un-flattening. This is the test that dies if the embedder
/// row index or the `[c][s][d]` stride is broken.
#[test]
fn embeddings_unflatten_64_slots_to_21_chunks_by_3_slots() {
  let d = build_decoded(|_, _, _| true, |_, _| false); // all 3 speakers active
  let plans = window_plans(ARGMAX_CHUNK_SAMPLES, &d.activity);
  let masks = build_speaker_masks(&d.ids, &d.overlapped, &plans);
  let embeddings = synthetic_embeddings();

  let num_chunks = 2 * ARGMAX_WINDOWS_PER_CHUNK;
  let (mut segs, mut raw) = buffers(num_chunks);
  let k = 1usize;
  for (w, plan) in plans.iter().enumerate() {
    write_segmentations(global_chunk(k, w), w, &d.ids, plan, &mut segs);
    place_embeddings(
      global_chunk(k, w),
      w,
      plan,
      &masks,
      &embeddings,
      &mut raw,
      &mut segs,
    )
    .expect("finite synthetic embeddings");
  }

  for w in 0..ARGMAX_WINDOWS_PER_CHUNK {
    let c = k * ARGMAX_WINDOWS_PER_CHUNK + w;
    for s in 0..SEG_NUM_SLOTS {
      let row = w * ARGMAX_NUM_SPEAKERS + s; // Emb.swift:240,288
      let expected = (row + 1) as f32;
      let got = &raw[(c * SEG_NUM_SLOTS + s) * EMBEDDING_DIM..][..EMBEDDING_DIM];
      assert!(
        got.iter().all(|&v| v == expected),
        "c={c} (k={k}, w={w}) s={s}: expected embedder row {row} (const {expected}), got {:?}",
        &got[..4]
      );
    }
  }
  // Chunk 0 (k=0) was never written: all zero.
  assert!(
    raw[..ARGMAX_WINDOWS_PER_CHUNK * SEG_NUM_SLOTS * EMBEDDING_DIM]
      .iter()
      .all(|&v| v == 0.0)
  );
}

/// An inactive slot's embedding row stays ZERO — it is never copied from the
/// embedder output, whose all-zero-mask rows carry a finite, non-zero
/// DEGENERATE constant (module doc: L2 ≈ 0.5356 on the real model, well above
/// `PLDA_MIN_NORM`). Emitting them would feed dia garbage.
///
/// Gated-out slots are now the ONLY rows whose mask is all-zero — for a slot
/// that cleared the gate, `dia`'s fallback makes that unreachable
/// (`every_gated_in_slot_gets_a_non_empty_mask`) — and their embedder output is
/// never read.
#[test]
fn inactive_slot_embedding_row_stays_zero() {
  // Only speaker 1 is active.
  let d = build_decoded(|_, _, s| s == 1, |_, _| false);
  let plans = window_plans(ARGMAX_CHUNK_SAMPLES, &d.activity);
  let masks = build_speaker_masks(&d.ids, &d.overlapped, &plans);
  let embeddings = synthetic_embeddings();

  let (mut segs, mut raw) = buffers(ARGMAX_WINDOWS_PER_CHUNK);
  for (w, plan) in plans.iter().enumerate() {
    write_segmentations(global_chunk(0, w), w, &d.ids, plan, &mut segs);
    place_embeddings(
      global_chunk(0, w),
      w,
      plan,
      &masks,
      &embeddings,
      &mut raw,
      &mut segs,
    )
    .unwrap();
  }

  for c in 0..ARGMAX_WINDOWS_PER_CHUNK {
    for s in [0usize, 2] {
      let row = &raw[(c * SEG_NUM_SLOTS + s) * EMBEDDING_DIM..][..EMBEDDING_DIM];
      assert!(row.iter().all(|&v| v == 0.0), "c={c} s={s} must stay zero");
    }
    // ...while the active slot 1 IS populated.
    let row = &raw[(c * SEG_NUM_SLOTS + 1) * EMBEDDING_DIM..][..EMBEDDING_DIM];
    assert!(row.iter().all(|&v| v != 0.0), "c={c} s=1 must be written");
  }
}

/// The worst case — a slot that clears the activity gate but whose EVERY
/// active frame is an overlap frame — end to end. Under argmax's unconditional
/// clean mask this is the all-zero mask row that yields the degenerate
/// norm-0.5356 constant; under `dia`'s fallback it gets the raw `speaker_ids`
/// mask, a real embedding, and is KEPT (row written, column intact).
///
/// The slot is not dropped, and that is the point: argmax would not drop it
/// either — it would only withhold it from cluster FORMATION
/// (`VBxClustering.swift:50`), a channel `Extraction` does not have (module
/// doc). Dropping would lose the attribution outright.
#[test]
fn an_all_overlap_slot_falls_back_and_is_kept() {
  // Speaker 1 active on frames 0..10 — and EVERY frame is an overlap frame.
  let d = build_decoded(|_, f, s| s == 1 && f < 10, |_, _| true);
  let plans = window_plans(ARGMAX_CHUNK_SAMPLES, &d.activity);
  assert!(
    plans[0].active[1],
    "10 frames must clear the activity gate — this is an ACTIVE slot"
  );

  let masks = build_speaker_masks(&d.ids, &d.overlapped, &plans);
  for w in 0..ARGMAX_WINDOWS_PER_CHUNK {
    let row = w * ARGMAX_NUM_SPEAKERS + 1;
    // `ids * (1 - overlapped)` is identically zero here; the fallback rescued it.
    assert!(
      !mask_row_is_zero(&masks, row),
      "w={w}: the fallback must leave a NON-empty mask row"
    );
    let start = window_start_frame(w);
    let nonzero = (0..ARGMAX_FRAMES_PER_WINDOW)
      .filter(|&f| mask_at(&masks, row, start + f) != 0.0)
      .count();
    assert_eq!(nonzero, 10, "w={w}: the raw mask's 10 active frames");
  }

  let (mut segs, mut raw) = buffers(ARGMAX_WINDOWS_PER_CHUNK);
  let embeddings = synthetic_embeddings(); // every row finite and non-zero
  for (w, plan) in plans.iter().enumerate() {
    write_segmentations(global_chunk(0, w), w, &d.ids, plan, &mut segs);
    place_embeddings(
      global_chunk(0, w),
      w,
      plan,
      &masks,
      &embeddings,
      &mut raw,
      &mut segs,
    )
    .unwrap();
  }

  // The slot SURVIVES: embedding row written, segmentation column intact.
  for c in 0..ARGMAX_WINDOWS_PER_CHUNK {
    let w = c; // k = 0
    let expected = (w * ARGMAX_NUM_SPEAKERS + 1 + 1) as f32; // synthetic row w*3+1
    let row = &raw[embedding_range(c, 1)];
    assert!(
      row.iter().all(|&v| v == expected),
      "c={c}: the slot must be EMBEDDED, not dropped"
    );
    for f in 0..10 {
      assert_eq!(
        segs[(c * ARGMAX_FRAMES_PER_WINDOW + f) * SEG_NUM_SLOTS + 1],
        1.0,
        "c={c} f={f}: the column must survive with the row"
      );
    }
  }
}

/// dia's PLDA-norm drop (`owned.rs:619-630`), re-applied here: a consumed row
/// below `PLDA_MIN_NORM` is dropped (row zero + column zeroed).
#[test]
fn low_norm_embedding_drops_the_slot() {
  let d = build_decoded(|_, _, s| s == 1, |_, _| false);
  let plans = window_plans(ARGMAX_CHUNK_SAMPLES, &d.activity);
  let masks = build_speaker_masks(&d.ids, &d.overlapped, &plans);

  // Row 1 (w=0, s=1) has a tiny norm: 256 * (1e-5)^2 -> ~1.6e-4 < 0.01.
  let mut embeddings = synthetic_embeddings();
  for d_i in 0..EMBEDDING_DIM {
    embeddings[EMBEDDING_DIM + d_i] = 1e-5;
  }

  let (mut segs, mut raw) = buffers(ARGMAX_WINDOWS_PER_CHUNK);
  for (w, plan) in plans.iter().enumerate() {
    write_segmentations(global_chunk(0, w), w, &d.ids, plan, &mut segs);
  }
  for (w, plan) in plans.iter().enumerate() {
    place_embeddings(
      global_chunk(0, w),
      w,
      plan,
      &masks,
      &embeddings,
      &mut raw,
      &mut segs,
    )
    .unwrap();
  }

  // Chunk 0 (w=0) dropped...
  assert!(raw[embedding_range(0, 1)].iter().all(|&v| v == 0.0));
  assert_eq!(segs[1], 0.0, "the dropped slot's column must be zeroed");
  // ...while chunk 1 (w=1) survives: its embedder row is `w*3 + s = 4`, whose
  // synthetic constant is `row + 1 = 5.0` (norm 5*16 = 80, far above the gate).
  assert!(raw[embedding_range(1, 1)].iter().all(|&v| v == 5.0));
  assert_eq!(segs[ARGMAX_FRAMES_PER_WINDOW * SEG_NUM_SLOTS + 1], 1.0);
}

/// A non-finite value in a CONSUMED embedding row is a HARD error (dia's
/// `owned.rs:611-618`; never a silent drop), reported at the flat index into
/// the model's own `speaker_embeddings` output.
#[test]
fn non_finite_consumed_embedding_row_errors() {
  let d = build_decoded(|_, _, s| s == 1, |_, _| false);
  let plans = window_plans(ARGMAX_CHUNK_SAMPLES, &d.activity);
  let masks = build_speaker_masks(&d.ids, &d.overlapped, &plans);

  let mut embeddings = synthetic_embeddings();
  embeddings[EMBEDDING_DIM + 7] = f32::NAN; // row 1 (w=0, s=1), dim 7

  let (mut segs, mut raw) = buffers(ARGMAX_WINDOWS_PER_CHUNK);
  let got = place_embeddings(
    global_chunk(0, 0),
    0,
    &plans[0],
    &masks,
    &embeddings,
    &mut raw,
    &mut segs,
  );
  assert_eq!(
    got,
    Err(InferError::NonFiniteOutput {
      index: EMBEDDING_DIM + 7
    })
  );
}

/// A non-finite value in a DISCARDED row (an inactive slot, or the unused
/// 64th) does NOT error — those rows are outside the Extraction entirely, and
/// dia computes no analog of them (`crate::extract`'s "NonFinite-output scan
/// scope" draws the same line).
#[test]
fn non_finite_discarded_embedding_row_is_ignored() {
  let d = build_decoded(|_, _, s| s == 1, |_, _| false); // only slot 1 active
  let plans = window_plans(ARGMAX_CHUNK_SAMPLES, &d.activity);
  let masks = build_speaker_masks(&d.ids, &d.overlapped, &plans);

  let mut embeddings = synthetic_embeddings();
  embeddings[0] = f32::NAN; // row 0 = (w=0, s=0) — INACTIVE, never consumed
  embeddings[(ARGMAX_MASK_SLOTS - 1) * EMBEDDING_DIM] = f32::INFINITY; // slot 63

  let (mut segs, mut raw) = buffers(ARGMAX_WINDOWS_PER_CHUNK);
  place_embeddings(
    global_chunk(0, 0),
    0,
    &plans[0],
    &masks,
    &embeddings,
    &mut raw,
    &mut segs,
  )
  .expect("a NaN in a discarded row must not fail the extraction");
  assert!(raw.iter().all(|v| v.is_finite()));
}

// =====================================================================
// Hermetic: Options (rust-options-pattern)
// =====================================================================

/// [`ArgmaxOptions::window`]'s `onset` is INERT for this source — not silently
/// ignored, but provably unable to change any output (module doc's "`onset` is
/// INERT for this source").
///
/// `onset` reaches exactly ONE site in `ArgmaxSource::extract`:
/// `try_count_from_segmentations`'s `v >= onset` threshold. Its
/// `segmentations` are argmax's hard-binary `speaker_ids` (value set exactly
/// `{0.0, 1.0}`, re-asserted on the real model by
/// `argmax_decoded_output_value_semantics`), and every onset the type admits
/// lies in `(0.0, 1.0]` — so `1.0 >= onset` is always true, `0.0 >= onset`
/// always false, and `count` is the SAME for all of them.
///
/// Pinned here so the day argmax emits a soft decode (or the comparison stops
/// being inclusive), this test fails and the knob becomes live rather than the
/// change passing silently.
#[test]
fn onset_is_inert_on_argmaxs_hard_binary_decode() {
  let num_chunks = 3usize;
  // A hard 0/1 `segmentations` buffer of exactly the shape this source builds.
  let mut segs = vec![0.0f64; num_chunks * ARGMAX_FRAMES_PER_WINDOW * SEG_NUM_SLOTS];
  for (i, v) in segs.iter_mut().enumerate() {
    *v = f64::from(u8::from(i % 3 == 0 || i % 11 == 0));
  }
  let w_opts = WindowOptions::new();
  let chunks_sw = crate::window::chunk_sliding_window(&w_opts);
  let frames_sw = crate::window::frame_sliding_window();
  let count_at = |onset: f32| {
    crate::window::try_count_from_segmentations(
      &segs,
      num_chunks,
      ARGMAX_FRAMES_PER_WINDOW,
      SEG_NUM_SLOTS,
      onset,
      chunks_sw,
      frames_sw,
    )
    .expect("count")
  };

  let baseline = count_at(crate::window::DEFAULT_ONSET);
  // The buffer is non-trivial, so this cannot pass vacuously on an all-zero count.
  assert!(baseline.iter().any(|&c| c > 0));
  for onset in [f32::MIN_POSITIVE, 0.01, 0.1, 0.5, 0.9, 1.0] {
    assert!(
      crate::window::check_onset(onset),
      "onset={onset} must be a VALID onset for this to prove inertness"
    );
    assert_eq!(
      count_at(onset),
      baseline,
      "onset={onset} must not change `count` on a hard-binary decode"
    );
  }
}

#[test]
fn argmax_options_defaults() {
  let o = ArgmaxOptions::new();
  assert_eq!(o.variant(), ArgmaxVariant::Baseline);
  assert_eq!(o.variant(), DEFAULT_ARGMAX_VARIANT);
  assert_eq!(o.compute().segmenter(), ComputeUnits::All);
  assert_eq!(o.compute().preprocessor(), ComputeUnits::All);
  assert_eq!(o.compute().embedder(), ComputeUnits::All);
  assert_eq!(o.window(), WindowOptions::new());
  assert_eq!(ArgmaxOptions::default(), o);
  assert_eq!(ArgmaxVariant::default(), ArgmaxVariant::Baseline);
}

#[test]
fn argmax_options_builders_round_trip() {
  let compute = ArgmaxComputeOptions::new()
    .with_segmenter(ComputeUnits::CpuOnly)
    .with_preprocessor(ComputeUnits::CpuAndGpu)
    .with_embedder(ComputeUnits::CpuAndNeuralEngine);
  let o = ArgmaxOptions::new()
    .with_variant(ArgmaxVariant::W8A16)
    .with_compute(compute)
    .with_window(WindowOptions::new().with_onset(0.25));
  assert_eq!(o.variant(), ArgmaxVariant::W8A16);
  assert_eq!(o.compute().segmenter(), ComputeUnits::CpuOnly);
  assert_eq!(o.compute().preprocessor(), ComputeUnits::CpuAndGpu);
  assert_eq!(o.compute().embedder(), ComputeUnits::CpuAndNeuralEngine);
  assert_eq!(o.window().onset(), 0.25);
}

/// The variant→directory mapping, including the trap that the BASELINE tier
/// is spelled differently for the two models (`W32A32` vs `W16A16`).
#[test]
fn variant_directories_match_the_shipped_layout() {
  assert_eq!(ArgmaxVariant::Baseline.segmenter_dir(), "W32A32");
  assert_eq!(ArgmaxVariant::Baseline.embedder_dir(), "W16A16");
  assert_eq!(ArgmaxVariant::W8A16.segmenter_dir(), "W8A16");
  assert_eq!(ArgmaxVariant::W8A16.embedder_dir(), "W8A16");

  let root = Path::new("/models");
  assert_eq!(
    seg_path(root, ArgmaxVariant::Baseline),
    Path::new("/models/speaker_segmenter/pyannote-v3/W32A32/SpeakerSegmenter.mlmodelc")
  );
  assert_eq!(
    preprocessor_path(root, ArgmaxVariant::Baseline),
    Path::new("/models/speaker_embedder/pyannote-v3/W16A16/SpeakerEmbedderPreprocessor.mlmodelc")
  );
  assert_eq!(
    embed_path(root, ArgmaxVariant::W8A16),
    Path::new("/models/speaker_embedder/pyannote-v3/W8A16/SpeakerEmbedder.mlmodelc")
  );
}

#[cfg(feature = "serde")]
#[test]
fn argmax_variant_serde_wire_values_are_snake_case() {
  assert_eq!(
    serde_json::to_string(&ArgmaxVariant::Baseline).unwrap(),
    "\"baseline\""
  );
  assert_eq!(
    serde_json::to_string(&ArgmaxVariant::W8A16).unwrap(),
    "\"w8_a16\""
  );
  for v in [ArgmaxVariant::Baseline, ArgmaxVariant::W8A16] {
    let json = serde_json::to_string(&v).unwrap();
    assert_eq!(serde_json::from_str::<ArgmaxVariant>(&json).unwrap(), v);
  }
}

#[cfg(feature = "serde")]
#[test]
fn argmax_options_deserialize_from_empty_object_uses_defaults() {
  let o: ArgmaxOptions = serde_json::from_str("{}").unwrap();
  assert_eq!(o, ArgmaxOptions::new());
}

// =====================================================================
// Hermetic: the padded-chunk builder
// =====================================================================

/// `Seg.swift:178-182`'s pad-or-trim, plus the unpadded length that drives
/// `bounded()` (`Seg.swift:176`'s `waveformLength`).
#[test]
fn fill_padded_chunk_pads_the_tail_and_reports_the_real_length() {
  let samples: Vec<f32> = (0..500_000).map(|i| (i % 100) as f32 / 100.0).collect();
  let mut padded = vec![f16::ONE; ARGMAX_CHUNK_SAMPLES];

  // A full chunk from 0.
  let n = fill_padded_chunk(&mut padded, &samples, 0);
  assert_eq!(n, ARGMAX_CHUNK_SAMPLES);
  assert_eq!(f32::from(padded[0]), 0.0);
  assert_eq!(
    f32::from(padded[7]),
    f32::from(f16::from_f32(samples[7])),
    "samples are converted to F16, never F32"
  );

  // The final, SHORT chunk: 500_000 - 336_000 = 164_000 real samples, and
  // the rest must be zero (not stale `f16::ONE`).
  let n = fill_padded_chunk(&mut padded, &samples, ARGMAX_CHUNK_HOP_SAMPLES);
  assert_eq!(n, 164_000);
  assert_eq!(
    f32::from(padded[163_999]),
    f32::from(f16::from_f32(samples[499_999]))
  );
  assert!(
    padded[164_000..].iter().all(|&v| v == f16::ZERO),
    "the out-of-range tail must be zero-padded"
  );
}

// =====================================================================
// Model-gated (#[ignore]): requires ARGMAX_TEST_MODELS
// =====================================================================

fn argmax_models_dir() -> PathBuf {
  std::env::var_os("ARGMAX_TEST_MODELS").map_or_else(
    || {
      PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("Models")
        .join("argmax-speakerkit")
    },
    PathBuf::from,
  )
}

fn load_source() -> ArgmaxSource {
  // CpuOnly for determinism, matching every other model-gated loader here.
  ArgmaxSource::from_dir_with(
    argmax_models_dir(),
    ArgmaxOptions::new().with_compute(
      ArgmaxComputeOptions::new()
        .with_segmenter(ComputeUnits::CpuOnly)
        .with_preprocessor(ComputeUnits::CpuOnly)
        .with_embedder(ComputeUnits::CpuOnly),
    ),
  )
  .expect("load the argmax speakerkit models")
}

/// The committed 30.0 s parity fixture (`tests/common`'s `02_pyannote_sample`).
fn load_pyannote_sample() -> Vec<f32> {
  let path =
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/audio/02_pyannote_sample.wav");
  let mut reader = hound::WavReader::open(&path).expect("02_pyannote_sample.wav opens");
  assert_eq!(reader.spec().sample_rate, 16_000);
  assert_eq!(reader.spec().channels, 1);
  reader
    .samples::<i16>()
    .map(|s| f32::from(s.expect("valid sample")) / 32_768.0)
    .collect()
}

/// Asserts every [`Extraction`] invariant `dia`'s `OfflineInput` requires.
fn assert_extraction_invariants(x: &Extraction, expected_chunks: usize) {
  assert_eq!(x.num_chunks(), expected_chunks);
  assert_eq!(x.num_frames_per_chunk(), ARGMAX_FRAMES_PER_WINDOW);
  assert_eq!(x.num_speakers(), SEG_NUM_SLOTS);
  // Lengths ARE the formulas.
  assert_eq!(
    x.segmentations().len(),
    x.num_chunks() * x.num_frames_per_chunk() * x.num_speakers()
  );
  assert_eq!(
    x.raw_embeddings().len(),
    x.num_chunks() * x.num_speakers() * EMBEDDING_DIM
  );
  assert_eq!(x.count().len(), x.num_output_frames());
  // Finiteness.
  assert!(x.segmentations().iter().all(|v| v.is_finite()));
  assert!(x.raw_embeddings().iter().all(|v| v.is_finite()));
  // `count` is an instantaneous speaker count over 3 slots.
  assert!(
    x.count().iter().all(|&c| (c as usize) <= SEG_NUM_SLOTS),
    "count must never exceed {SEG_NUM_SLOTS}"
  );
  // argmax's `speaker_ids` is hard 0/1, so `segmentations` is too.
  assert!(
    x.segmentations().iter().all(|&v| v == 0.0 || v == 1.0),
    "segmentations must be the hard 0/1 in-graph decode"
  );
  // THE cross-tensor invariant: a zero embedding row ⟺ a zero seg column.
  for c in 0..x.num_chunks() {
    for s in 0..SEG_NUM_SLOTS {
      let row = &x.raw_embeddings()[(c * SEG_NUM_SLOTS + s) * EMBEDDING_DIM..][..EMBEDDING_DIM];
      let row_zero = row.iter().all(|&v| v == 0.0);
      let col_zero = (0..x.num_frames_per_chunk())
        .all(|f| x.segmentations()[(c * x.num_frames_per_chunk() + f) * SEG_NUM_SLOTS + s] == 0.0);
      assert_eq!(
        row_zero, col_zero,
        "chunk {c} slot {s}: a dropped slot must have BOTH an all-zero \
         embedding row and an all-zero segmentation column"
      );
    }
  }
}

/// End-to-end on the real 30.0 s fixture: 21 Extraction chunks (one argmax
/// 30 s chunk × 21 in-graph windows), all invariants, all finite.
#[test]
#[ignore = "requires local argmax speakerkit models (ARGMAX_TEST_MODELS)"]
fn argmax_source_extracts_the_pyannote_sample() {
  let samples = load_pyannote_sample();
  assert_eq!(samples.len(), 480_000, "the fixture is exactly 30.0 s");

  let got = load_source().extract(&samples).expect("extract");
  assert_extraction_invariants(&got, ARGMAX_WINDOWS_PER_CHUNK);

  // Real speech: some slot must actually have been embedded.
  assert!(
    got.raw_embeddings().iter().any(|&v| v != 0.0),
    "a 30 s speech clip must produce at least one embedding"
  );
  assert!(
    got.count().iter().any(|&c| c > 0),
    "a 30 s speech clip must have some active frames"
  );
}

/// The multi-chunk stitch: 60 s of audio → 3 argmax 30 s chunks (starts 0 /
/// 336 000 / 672 000, the last one truncated to 18 s so only 9 of its windows
/// are bounded) → 51 Extraction chunks, exactly dia's grid.
#[test]
#[ignore = "requires local argmax speakerkit models (ARGMAX_TEST_MODELS)"]
fn argmax_source_extracts_long_audio_across_chunks() {
  let one = load_pyannote_sample();
  let samples: Vec<f32> = one.iter().chain(one.iter()).copied().collect();
  assert_eq!(samples.len(), 960_000, "60 s");

  // The geometry this exercises, stated up front.
  assert_eq!(argmax_chunk_starts(960_000), vec![0, 336_000, 672_000]);
  let expected_chunks = chunk_starts(960_000, &WindowOptions::new()).len();
  assert_eq!(expected_chunks, 51, "21 + 21 + 9 bounded windows");

  let got = load_source().extract(&samples).expect("extract 60 s");
  assert_extraction_invariants(&got, expected_chunks);
  assert!(got.raw_embeddings().iter().any(|&v| v != 0.0));
}

/// The two sources must agree on GEOMETRY (the grid theorem) even though they
/// legitimately disagree on VALUES (different decode semantics — spec §4).
#[test]
#[ignore = "requires local argmax + speakerkit models (both env vars)"]
fn argmax_and_fluid_audio_agree_on_geometry_not_values() {
  let samples = load_pyannote_sample();
  let argmax = load_source().extract(&samples).expect("argmax extract");

  let models_dir = std::env::var_os("SPEAKERKIT_TEST_MODELS").map_or_else(
    || {
      PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("Models")
        .join("speakerkit")
    },
    PathBuf::from,
  );
  let seg = crate::segment::SegmentModel::from_file_with(
    models_dir.join("pyannote_segmentation.mlmodelc"),
    crate::segment::SegmentModelOptions::new().with_compute(ComputeUnits::CpuOnly),
  )
  .expect("load pyannote_segmentation.mlmodelc");
  let embed = crate::embed::EmbedModel::from_file_with(
    models_dir.join("wespeaker_v2.mlmodelc"),
    crate::embed::EmbedModelOptions::new().with_compute(ComputeUnits::CpuOnly),
  )
  .expect("load wespeaker_v2.mlmodelc");
  let fluid = crate::source::FluidAudioSource::new(seg, embed)
    .extract(&samples)
    .expect("fluid extract");

  // Geometry: identical, by the grid theorem.
  assert_eq!(argmax.num_chunks(), fluid.num_chunks());
  assert_eq!(argmax.num_frames_per_chunk(), fluid.num_frames_per_chunk());
  assert_eq!(argmax.num_output_frames(), fluid.num_output_frames());
  assert_eq!(argmax.num_speakers(), fluid.num_speakers());
  assert_eq!(argmax.chunks_sw(), fluid.chunks_sw());
  assert_eq!(argmax.frames_sw(), fluid.frames_sw());
  assert_eq!(argmax.segmentations().len(), fluid.segmentations().len());
  assert_eq!(argmax.raw_embeddings().len(), fluid.raw_embeddings().len());
  // Values: NOT required to match — two different decodes (spec §4). Assert
  // only that both actually produced something, so this test cannot pass
  // vacuously on two empty extractions.
  assert!(argmax.raw_embeddings().iter().any(|&v| v != 0.0));
  assert!(fluid.raw_embeddings().iter().any(|&v| v != 0.0));
}

/// The value semantics the whole mapping assumes, re-asserted against the
/// REAL models (module doc's "Value semantics" section). The Swift says WHICH
/// tensors; only this says WHAT is in them — if a future argmax revision made
/// `speaker_ids` soft, or `speaker_activity` a duration rather than a count,
/// every downstream index would still "work" while meaning something else.
#[test]
#[ignore = "requires local argmax speakerkit models (ARGMAX_TEST_MODELS)"]
fn argmax_decoded_output_value_semantics() {
  let samples = load_pyannote_sample();
  let source = load_source();

  let mut padded = vec![f16::ZERO; ARGMAX_CHUNK_SAMPLES];
  let n = fill_padded_chunk(&mut padded, &samples, 0);
  assert_eq!(n, ARGMAX_CHUNK_SAMPLES);
  let waveform = MultiArray::from_slice(&[ARGMAX_CHUNK_SAMPLES], &padded).unwrap();
  let d = source.segment_chunk(&waveform).expect("segment");

  // 1. `speaker_ids` is hard binary — the in-graph powerset decode.
  assert!(
    d.ids.iter().all(|&v| v == 0.0 || v == 1.0),
    "speaker_ids must be exactly {{0.0, 1.0}}"
  );
  // 2. `speaker_activity[w][s]` IS the active-frame COUNT of `speaker_ids`.
  for w in 0..ARGMAX_WINDOWS_PER_CHUNK {
    for s in 0..ARGMAX_NUM_SPEAKERS {
      let count = (0..ARGMAX_FRAMES_PER_WINDOW)
        .filter(|&f| d.ids[ids_index(w, f, s)] != 0.0)
        .count();
      assert_eq!(
        d.activity[w * ARGMAX_NUM_SPEAKERS + s],
        count as f32,
        "speaker_activity[{w}][{s}] must equal the frame count of speaker_ids"
      );
    }
  }
  // 3. `overlapped_speaker_activity[w][f]` IS the binary "2+ active" flag.
  for w in 0..ARGMAX_WINDOWS_PER_CHUNK {
    for f in 0..ARGMAX_FRAMES_PER_WINDOW {
      let active = (0..ARGMAX_NUM_SPEAKERS)
        .filter(|&s| d.ids[ids_index(w, f, s)] != 0.0)
        .count();
      let expected = if active >= 2 { 1.0 } else { 0.0 };
      assert_eq!(
        d.overlapped[overlapped_index(w, f)],
        expected,
        "overlapped[{w}][{f}] must be (active_speakers >= 2)"
      );
    }
  }
}

/// The all-zero-mask row returns a FINITE, non-zero, CONSTANT embedding whose
/// norm is far above `PLDA_MIN_NORM` — the model behavior that makes an
/// unconditional clean mask dangerous, and hence the reason this port applies
/// `dia`'s exclude-overlap fallback (module doc).
///
/// It is NOT a NaN (which FluidAudio's WeSpeaker returns from an empty mask and
/// which the finiteness scan would catch), and its norm is ~54× the PLDA guard,
/// so nothing downstream would catch it either. The fallback makes it
/// unreachable for a consumed slot
/// (`no_consumed_slot_yields_the_degenerate_constant`); this test keeps the
/// underlying model fact honest — if a future argmax revision made it NaN or
/// zero, the rationale changes and must be re-read, not assumed.
#[test]
#[ignore = "requires local argmax speakerkit models (ARGMAX_TEST_MODELS)"]
fn all_zero_mask_row_yields_a_finite_degenerate_constant() {
  let samples = load_pyannote_sample();
  let source = load_source();
  let mut padded = vec![f16::ZERO; ARGMAX_CHUNK_SAMPLES];
  fill_padded_chunk(&mut padded, &samples, 0);

  // Every one of the 64 mask rows all-zero.
  let masks = vec![f16::ZERO; ARGMAX_MASK_SLOTS * ARGMAX_MASK_FRAMES];
  let embeddings = source.embed_chunk(&padded, &masks).expect("embed");

  assert!(
    embeddings.iter().all(|v| v.is_finite()),
    "an all-zero mask must NOT produce NaN/Inf (unlike FluidAudio's WeSpeaker)"
  );
  let norm = |row: usize| -> f64 {
    embeddings[row * EMBEDDING_DIM..(row + 1) * EMBEDDING_DIM]
      .iter()
      .map(|v| f64::from(*v) * f64::from(*v))
      .sum::<f64>()
      .sqrt()
  };
  let base = norm(0);
  assert!(
    base > PLDA_MIN_NORM,
    "the degenerate embedding's norm ({base}) is ABOVE PLDA_MIN_NORM \
     ({PLDA_MIN_NORM}) — which is exactly why the norm guard cannot catch it \
     and the degenerate-mask guard must exist"
  );
  for row in 0..ARGMAX_MASK_SLOTS {
    assert!(
      (norm(row) - base).abs() < 1e-6,
      "row {row}: every all-zero-mask row must yield the SAME constant"
    );
  }
}

/// **THE fix, on the real models** (module doc): with `dia`'s exclude-overlap
/// fallback in place, NO consumed slot pools over an all-zero mask, so none of
/// them can carry the degenerate constant
/// (`all_zero_mask_row_yields_a_finite_degenerate_constant`) — every consumed
/// embedding is finite, non-constant, and distinct from it.
///
/// It also measures the two populations the module doc distinguishes, on real
/// audio, so neither claim is taken on faith — and so that the gulf between
/// their thresholds stays visible (measured on `02_pyannote_sample`:
/// `consumed=41 fell_back=0 sparse=5`):
///
/// - `fell_back` — slots whose clean count is `<= EXCLUDE_OVERLAP_MIN_FRAMES`
///   (2), the ones `dia`'s fallback rewrites. A RARE path: none on this fixture.
/// - `sparse` — slots whose `nonOverlappedFrameRatio` is `<= minActiveRatio`
///   (0.2, i.e. `<= 117` clean frames of 589), the ones argmax's OWN clustering
///   would withhold from cluster FORMATION (`VBxClustering.swift:50`,
///   `SpeakerClustering.swift:23`) and which this port instead carries into
///   `dia`'s clustering, sparse clean mask and all.
///
/// The second set is ~50× the first, and the fallback does NOT rescue it — that
/// is the module doc's declared divergence, and this test exists partly to stop
/// anyone reading the fallback as a fix for it. What the fallback guarantees is
/// narrower and is what the assertions below check: no consumed slot pools over
/// an EMPTY mask, so none carries the meaningless constant.
#[test]
#[ignore = "requires local argmax speakerkit models (ARGMAX_TEST_MODELS)"]
fn no_consumed_slot_yields_the_degenerate_constant() {
  let samples = load_pyannote_sample();
  let source = load_source();

  let mut padded = vec![f16::ZERO; ARGMAX_CHUNK_SAMPLES];
  let n = fill_padded_chunk(&mut padded, &samples, 0);
  assert_eq!(
    n, ARGMAX_CHUNK_SAMPLES,
    "the fixture is exactly one 30 s chunk"
  );

  // 1. The degenerate constant, re-measured on THIS machine's models rather
  //    than hard-coded from the probe.
  let zero_masks = vec![f16::ZERO; ARGMAX_MASK_SLOTS * ARGMAX_MASK_FRAMES];
  let degenerate = source
    .embed_chunk(&padded, &zero_masks)
    .expect("embed an all-zero mask");
  let degenerate = &degenerate[..EMBEDDING_DIM];

  // 2. The decoded tensors, so each slot's clean-frame count — argmax's own
  //    `nonOverlappedFrameRatio` numerator (`Emb.swift:105-118`) — is available.
  let waveform = MultiArray::from_slice(&[ARGMAX_CHUNK_SAMPLES], &padded).unwrap();
  let d = source.segment_chunk(&waveform).expect("segment");
  let plans = window_plans(ARGMAX_CHUNK_SAMPLES, &d.activity);

  let got = source.extract(&samples).expect("extract");
  assert_eq!(got.num_chunks(), ARGMAX_WINDOWS_PER_CHUNK);

  let (mut consumed, mut sparse, mut fell_back) = (0usize, 0usize, 0usize);
  for (w, plan) in plans.iter().enumerate() {
    for (s, &is_active) in plan.active.iter().enumerate() {
      if !is_active {
        continue; // gated out — never masked, never embedded
      }
      consumed += 1;
      let clean = (0..ARGMAX_FRAMES_PER_WINDOW)
        .filter(|&f| {
          d.ids[ids_index(w, f, s)] != 0.0 && d.overlapped[overlapped_index(w, f)] == 0.0
        })
        .count();
      if clean <= EXCLUDE_OVERLAP_MIN_FRAMES {
        fell_back += 1;
      }
      // argmax's clustering-stage exclusion: ratio = clean / 589 <= 0.2.
      if clean as f32 / ARGMAX_FRAMES_PER_WINDOW as f32 <= 0.2 {
        sparse += 1;
      }

      let c = global_chunk(0, w);
      let row = &got.raw_embeddings()[embedding_range(c, s)];
      assert!(
        row.iter().any(|&v| v != 0.0),
        "c={c} s={s} (clean={clean}): a consumed slot must be EMBEDDED, not dropped"
      );
      assert!(row.iter().all(|v| v.is_finite()), "c={c} s={s}");

      // Not the all-zero-mask constant...
      let max_diff = row
        .iter()
        .zip(degenerate)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
      assert!(
        max_diff > 1e-3,
        "c={c} s={s} (clean={clean}): a consumed embedding must not BE the \
         degenerate all-zero-mask constant (max|diff| = {max_diff})"
      );
      // ...nor a constant vector at all.
      let lo = row.iter().copied().fold(f32::INFINITY, f32::min);
      let hi = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
      assert!(
        hi - lo > 1e-3,
        "c={c} s={s} (clean={clean}): a consumed embedding must not be constant"
      );
    }
  }

  assert!(
    consumed > 0,
    "a 30 s speech clip must produce consumed slots"
  );
  eprintln!("consumed={consumed} fell_back(clean<=2)={fell_back} sparse(ratio<=0.2)={sparse}");
  // The divergence the module doc declares is REAL on this fixture: these are
  // the slots argmax would withhold from cluster formation and this port hands
  // to dia — every one of them just proven to carry a real embedding.
  assert!(
    sparse > 0,
    "expected sparse-clean slots on this fixture — if this fires, the module \
     doc's 12-17 % claim needs re-measuring, not deleting"
  );
}

/// `step_samples` is compiled into argmax's graph: anything but 16 000 is
/// REJECTED, never silently ignored (which would return an `Extraction` whose
/// `chunks_sw.step()` lied about its own grid).
#[test]
#[ignore = "requires local argmax speakerkit models (ARGMAX_TEST_MODELS)"]
fn argmax_source_rejects_a_foreign_step_samples() {
  let source = ArgmaxSource::from_dir_with(
    argmax_models_dir(),
    ArgmaxOptions::new().with_window(WindowOptions::new().with_step_samples(8_000)),
  )
  .expect("load");
  assert_eq!(
    source.extract(&[0.0; 16_000]),
    Err(ExtractError::UnsupportedStepSamples {
      step: 8_000,
      required: 16_000,
    })
  );
}

/// Empty input is rejected exactly as `Extractor::extract` rejects it.
#[test]
#[ignore = "requires local argmax speakerkit models (ARGMAX_TEST_MODELS)"]
fn argmax_source_rejects_empty_samples() {
  assert_eq!(load_source().extract(&[]), Err(ExtractError::EmptySamples));
}
