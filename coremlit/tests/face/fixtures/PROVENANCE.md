# Face fixtures — where every image came from, and on what basis

Three families of fixture live under this directory.

- **`align_crop_64x48_rgb8.bin` / `align_expected_112x112_rgb8.bin`** are SYNTHETIC.
  They contain no photograph and no person: `conversion/face/align_oracle.py` generates
  the crop from three arithmetic sequences and warps it through the ArcFace template. They
  are the alignment golden and have no provenance question.
- **`faces/`** holds 18 real photographs of 6 people, plus the 112x112 aligned crop cut
  from each. This file is about those.
- **`onnx_reference.json`** holds the fp32 `onnxruntime` embedding of each of those 18
  crops — 18 x 512 floats, the cross-implementation oracle `tests/face/parity.rs` compares
  the CoreML door against. It carries NO weight bytes: it is a measurement of what the
  pinned `w600k_r50.onnx` computes over six public-domain photographs, and the bytes that
  produced it are named by hash in its own provenance block so a reader can regenerate and
  compare. It is committed because a gate cannot depend on an ONNX runtime — the `face`
  feature pulls none — which is the shape `granite` and `siglip` already use for their
  transformers-fp32 goldens.

## The licence basis, once

Every image in `faces/` is a **work of the U.S. federal government and is in the public
domain**. They come from NASA's image library. NASA's Media Usage Guidelines state:

> NASA content — images, audio, video, and computer files used in the rendition of
> 3-dimensional models, such as texture maps and polygon data in any format — generally are
> not subject to copyright in the United States.

<https://www.nasa.gov/nasa-brand-center/images-and-media/>

Two caveats from the same page, recorded because they are part of the terms even though
neither bears on copyright:

- an image showing an **identifiable person** may not be usable for **commercial** purposes
  without infringing that person's right of privacy or publicity. Every image here shows an
  identifiable person. This is a constraint on advertising and endorsement, not on
  redistributing a test fixture, and it does not make the works copyrighted;
- **NASA insignia** may not be used without permission. Several images contain them.

**There is no research-only corpus anywhere in this directory.** No LFW, no CelebA, no
CFP, no WebFace, nothing scraped. The point is worth stating flatly: `coremlit` cannot
train on a clean corpus — issue #115's census found no accuracy-adequate face model whose
corpus permits a product — but what it *redistributes* is entirely within its own control,
and this is it.

## How the set was chosen — the rule, fixed before any embedding was computed

An image is eligible iff **all** of:

1. its NASA record names a NASA centre and no field in the record or in the asset's
   `metadata.json` names a non-NASA rights holder;
2. its caption names exactly one astronaut and no other person on the sweep's roster;
3. SCRFD-10GF detects **exactly one** face in the `~medium` asset, at score >= 0.7 and
   box width >= 90 px;
4. `|yaw proxy| <= 1.2`, where the yaw proxy is the nose keypoint's offset from the eye
   midpoint in eye-spans. Beyond that the far eye is occluded and its keypoint is
   extrapolated rather than observed, so the fixture would be measuring the DETECTOR;
5. it is **visually confirmed to be the named person**, by comparing the aligned crop
   against that person's official NASA portrait.

Rules 1–4 are mechanical and live in `conversion/face/scripts/build_fixtures.py`. Rule 5
is a human judgement, and it is the reason the source photographs are committed beside the
crops: a claim that a fixture shows Peggy Whitson is checkable by someone other than its
author only if the photograph is there to look at.

**Rule 5 is not ceremony.** NASA record `NHQ201605250017`, captioned "Scott Kelly
Post-Flight Visit to Washington", passes rules 1–4 and its single detected face is **a
different person at the event**. A caption names who a photograph is *about*, not whose
face is in it.

**Rule 5 also removed an identity rather than an image.** Scott Kelly has an identical twin
brother who is also a NASA astronaut. A known-pairs set cannot contain a person whose
different-person pairs are genuinely ambiguous to any embedder, and no visual check can
settle a portrait's caption against that risk. Both Kellys are out.

## What is committed

| file | what |
|---|---|
| `faces/<id>.rgb8` | the **aligned 112x112 RGB8 crop**, 37 632 bytes, row-major HWC. This is the test input, and it is committed so a gate is hermetic. |
| `faces/<id>.jpg` | the source photograph, re-encoded at 640 px on the long side, quality 82. A legible copy for rule 5, not the crop's source. |
| `faces/manifest.json` | per-image provenance, the detection, the 5 landmarks, the solved 2x3 alignment matrix, and both SHA-256s. |
| `onnx_reference.json` | the fp32 `onnxruntime` embedding of every crop, its L2 norm, the known-pairs statistics those vectors give, the pinned source hashes and the observed toolchain. Each face carries its crop's SHA-256, so a reference cut against different bytes reds at load rather than at comparison. |

The crops are cut from the **full `~medium` NASA asset**, whose SHA-256 is pinned per
image in `build_fixtures.py`; the script re-downloads and re-verifies before it cuts, so a
regeneration reproduces these bytes or fails loudly. Alignment is
`conversion/face/align_oracle.py` — the same code the committed alignment golden is
produced by — so a parity or known-pairs number taken over these crops is a statement about
the **embedder** and not about two alignments that happen to be close.

The detector (`det_10g.onnx`, SCRFD-10GF, from the same `buffalo_l` pack) runs **only**
here. It is never converted, never published, and nothing in `src/` imports it. It is used
because the five landmarks the ArcFace template is defined against have to come from
somewhere, and taking them from the detector InsightFace itself pairs with `w600k_r50`
introduces no new alignment.

## The set

6 identities, 3 photographs each, every pair of photographs of one person from a different
session — 18 same-person pairs and 135 different-person pairs.

| fixture | person | date | centre | credit | NASA id | face box | yaw proxy | crop sha256 |
|---|---|---|---|---|---|---|---|---|
| `whitson_JSC2001-03044` | Peggy A. Whitson | 2001-11-28 | JSC | NASA/Bill Stafford | [`JSC2001-03044`](https://images-assets.nasa.gov/image/JSC2001-03044/JSC2001-03044~medium.jpg) | 271 px | +0.04 | `23be26b605829dd1…` |
| `whitson_iss005e07178` | Peggy A. Whitson | 2002-07-09 | JSC | NASA | [`iss005e07178`](https://images-assets.nasa.gov/image/iss005e07178/iss005e07178~medium.jpg) | 109 px | -0.82 | `f0d14ff73b447e82…` |
| `whitson_NHQ201803020004` | Peggy A. Whitson | 2018-03-02 | HQ | NASA/Joel Kowsky | [`NHQ201803020004`](https://images-assets.nasa.gov/image/NHQ201803020004/NHQ201803020004~medium.jpg) | 192 px | +0.19 | `df741cfc8786f8ea…` |
| `hopkins_iss037e021314` | Michael S. Hopkins | 2013-10-26 | JSC | NASA/Karen Nyberg | [`iss037e021314`](https://images-assets.nasa.gov/image/iss037e021314/iss037e021314~medium.jpg) | 270 px | -0.25 | `b9ecab35803375c2…` |
| `hopkins_iss037e013962` | Michael S. Hopkins | 2013-10-15 | JSC | NASA/Karen Nyberg | [`iss037e013962`](https://images-assets.nasa.gov/image/iss037e013962/iss037e013962~medium.jpg) | 118 px | -0.69 | `c8cbf937191c8552…` |
| `hopkins_iss064e025819` | Michael S. Hopkins | 2021-01-26 | JSC | NASA | [`iss064e025819`](https://images-assets.nasa.gov/image/iss064e025819/iss064e025819~medium.jpg) | 179 px | +0.05 | `12ae26201cacba62…` |
| `williams_jsc2005e02663` | Sunita L. Williams | 2004-09-22 | JSC | NASA/Mark Sowa | [`jsc2005e02663`](https://images-assets.nasa.gov/image/jsc2005e02663/jsc2005e02663~medium.jpg) | 158 px | -0.07 | `43dcb26e81e6bdf4…` |
| `williams_iss015e07586` | Sunita L. Williams | 2007-05-13 | JSC | NASA | [`iss015e07586`](https://images-assets.nasa.gov/image/iss015e07586/iss015e07586~medium.jpg) | 118 px | +0.34 | `053a666dd2f6ed48…` |
| `williams_jsc2011e086095` | Sunita L. Williams | 2011-09-08 | JSC | NASA/Robert Markowitz | [`jsc2011e086095`](https://images-assets.nasa.gov/image/jsc2011e086095/jsc2011e086095~medium.jpg) | 206 px | -0.13 | `abdec81d1eed39fa…` |
| `lindgren_NHQ202009160011` | Kjell N. Lindgren | 2020-09-16 | HQ | NASA/Bill Ingalls | [`NHQ202009160011`](https://images-assets.nasa.gov/image/NHQ202009160011/NHQ202009160011~medium.jpg) | 200 px | +0.08 | `78ad6b74713080a2…` |
| `lindgren_iss045e168328` | Kjell N. Lindgren | 2015-12-02 | JSC | NASA/Kjell Lindgren | [`iss045e168328`](https://images-assets.nasa.gov/image/iss045e168328/iss045e168328~medium.jpg) | 233 px | -0.41 | `a2f03159a37d53a4…` |
| `lindgren_NHQ202303310015` | Kjell N. Lindgren | 2023-03-31 | HQ | NASA/Keegan Barber | [`NHQ202303310015`](https://images-assets.nasa.gov/image/NHQ202303310015/NHQ202303310015~medium.jpg) | 125 px | -0.16 | `2faeebb33599e8f5…` |
| `meir_NHQ202009150005` | Jessica U. Meir | 2020-09-15 | HQ | NASA/Bill Ingalls | [`NHQ202009150005`](https://images-assets.nasa.gov/image/NHQ202009150005/NHQ202009150005~medium.jpg) | 183 px | +0.12 | `e642796e2a0b72aa…` |
| `meir_iss062e103558` | Jessica U. Meir | 2020-03-20 | JSC | NASA/Jessica Meir | [`iss062e103558`](https://images-assets.nasa.gov/image/iss062e103558/iss062e103558~medium.jpg) | 122 px | +0.39 | `dd3dc18fda7c0599…` |
| `meir_jsc2025e078605_alt` | Jessica U. Meir | 2025-09-26 | JSC | NASA/Josh Valcarcel | [`jsc2025e078605_alt`](https://images-assets.nasa.gov/image/jsc2025e078605_alt/jsc2025e078605_alt~medium.jpg) | 146 px | -0.02 | `4e3723b111bc940e…` |
| `hague_NHQ202001130002` | Nick Hague | 2020-01-13 | HQ | NASA/Aubrey Gemignani | [`NHQ202001130002`](https://images-assets.nasa.gov/image/NHQ202001130002/NHQ202001130002~medium.jpg) | 256 px | +0.05 | `7464acddd9da5daf…` |
| `hague_iss059e036143` | Nick Hague | 2019-04-29 | JSC | NASA/Nick Hague | [`iss059e036143`](https://images-assets.nasa.gov/image/iss059e036143/iss059e036143~medium.jpg) | 138 px | -0.03 | `8eec5835f20a1f0c…` |
| `hague_NHQ202509120024` | Nick Hague | 2025-09-12 | HQ | NASA/Bill Ingalls | [`NHQ202509120024`](https://images-assets.nasa.gov/image/NHQ202509120024/NHQ202509120024~medium.jpg) | 132 px | -0.06 | `0bb730e7c0ff6927…` |

**The frontal-to-profile case is in the set deliberately.** Issue #115 measured AuraFace
splitting identity on frontal-to-profile pairs 38.55 % of the time against `buffalo_l`'s
2.22 %, so a fixture set with no profile in it would not exercise the regime that decided
the model choice. `whitson_iss005e07178` (yaw proxy −0.82, a full side view) against
`whitson_NHQ201803020004` (frontal, 16 years later) is the hardest pair in the set and
the one that sets `min same-person` in every measurement.

## Rejected, and why

Kept because each names a failure mode a reader would otherwise have to rediscover.


| rejected | why |
|---|---|
| `NHQ201605250017` — Scott Kelly Post-Flight Visit to Washington | passes rules 1-4; the single detected face is a DIFFERENT PERSON at the event. A caption names who a photograph is about, not whose face is in it. |
| `(the whole Scott Kelly identity)` — several official NASA portraits | identical twin brother, also a NASA astronaut. A different-person pair that is genuinely ambiguous to any embedder does not belong in a known-pairs fixture set. |
| `iss062e014339` — ACE-T4 Module Configuration (Jessica Meir) | yaw proxy +1.63 — beyond YAW_MAX; the crop is dominated by a gloved hand and the far-eye keypoint is extrapolated. |
| `iss037e020099` — Hopkins at work in Quest airlock | yaw proxy -2.30 — beyond YAW_MAX. |
| `iss045e089495` — Ham Radio Session in Columbus (Kjell Lindgren) | yaw proxy -2.27 — beyond YAW_MAX. The crop looks usable, which is the point of having a rule rather than an eye: the far eye is not observed. |
| `KSC-20210715-PH-KLS02_*` — Victor Glover Tours VAB / O&C | rule 1: the KSC records' rights line defers to a third party ("For copyright and restrictions refer to ..."). |
| `jsc2013e067459` — Expedition 37 Crew News Conference | rule 2: the caption names three crew members. The face IS Hopkins — confirmed against the two unambiguous Hopkins images — but a fixture should not rest on a caption a reader has to adjudicate. |

## Regenerating

```sh
coremlit/conversion/face/run_arcface.sh fixtures     # the crops and faces/manifest.json
coremlit/conversion/face/run_arcface.sh reference    # onnx_reference.json
```

The first re-downloads each pinned `~medium` asset, refuses any whose SHA-256 has moved,
re-runs detection and alignment, and rewrites the crops, the legible sources and
`faces/manifest.json`. The second re-verifies every crop against that manifest, re-runs the
pinned ONNX over them, refuses a run whose known pairs stop separating at InsightFace's own
operating point, and rewrites `onnx_reference.json`; it needs only numpy and onnxruntime — no
torch and no coremltools — and observes exactly the packages it imports rather than the whole
conversion stack. A change to any committed byte is therefore a deliberate diff and never a
silent re-baseline.

The two must be regenerated TOGETHER and in that order: the reference carries each crop's
SHA-256, and `tests/face/arcface/mod.rs`'s loader refuses a pair that has drifted apart.
