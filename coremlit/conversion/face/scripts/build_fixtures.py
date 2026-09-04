"""Cut the committed known-pairs fixtures from pinned NASA photographs.
Usage: ``python build_fixtures.py``.

**Every image here is a work of the U.S. federal government, in the public domain.** They
come from NASA's own image library, whose Media Usage Guidelines state that NASA content
"generally are not subject to copyright in the United States". No LFW, no CelebA, nothing
research-licensed, nothing scraped — the corpus this crate TESTS on must be as clean as the
corpus it wishes it could train on, because a fixture is redistributed and a training set
is not.

# The selection rule, fixed before a single embedding was computed

An image is eligible iff **all** of:

1. its NASA record names a NASA centre and no field in the record or in the asset's
   ``metadata.json`` names a non-NASA rights holder (the sweep that produced this list
   rejected every Reuters/AP/Boeing/Axiom/GCTC/ESA/JAXA credit, and every KSC record whose
   rights line defers to a third party);
2. its caption names exactly one astronaut and no other person on the sweep's roster;
3. SCRFD-10GF detects **exactly one** face in the ``~medium`` asset, at score
   >= ``DET_SCORE_MIN`` and box width >= ``BOX_WIDTH_MIN`` px;
4. ``|yaw proxy| <= YAW_MAX`` — beyond that the far eye is occluded and its keypoint is
   extrapolated rather than observed, so the fixture would be measuring the DETECTOR;
5. it is **visually confirmed to be the named person**, by comparing the aligned crop
   against that person's official NASA portrait.

Rule 5 is not ceremony. NASA record ``NHQ201605250017``, captioned "Scott Kelly Post-Flight
Visit to Washington", passed rules 1-4 and its single detected face is **a different person
at the event**. A caption names who a photograph is ABOUT, not whose face is in it. That
image is not in this list.

Rule 5 also removed an identity rather than an image: **Scott Kelly has an identical twin
brother who is also a NASA astronaut.** A known-pairs fixture set cannot contain a person
whose different-person pairs are genuinely ambiguous to any embedder, and no visual check
can settle a NASA portrait's caption against that risk. Both Kellys are out.

# What is committed, and what is only pinned

The **aligned 112x112 RGB8 crops** are committed: they are the test inputs, and a gate has
to be hermetic. The **source photographs** are committed too, re-encoded at
``SOURCE_LONG_SIDE`` px so a reader can see whose face this is — that is what makes rule 5
checkable by someone other than its author. The crops are cut from the FULL ``~medium``
asset, whose SHA-256 is pinned per image, so this script replays them exactly; the
committed JPEG is a legible copy, not the crop's source.

Alignment is ``align_oracle.py`` — the same code the committed alignment golden is produced
by — so a parity or known-pairs number here is a statement about the EMBEDDER and not about
two alignments that happen to be close.
"""
import hashlib
import io
import json
import sys
import urllib.request

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from _arcface_common import (INSIGHTFACE_REV, TEMPLATE_SIZE, align_oracle, conv_dir,
                             detector_path, fixtures_dir, observed_toolchain, sha256_file)
from _scrfd import Detector

DET_SCORE_MIN = 0.7
BOX_WIDTH_MIN = 90
YAW_MAX = 1.2
SOURCE_LONG_SIDE = 640
SOURCE_QUALITY = 82

#: NASA's own statement of the terms, quoted rather than paraphrased, and recorded once
#: because it is identical for every row.
LICENCE_BASIS = ("Work of the U.S. federal government, in the public domain. NASA Media "
                 "Usage Guidelines: \"NASA content — images, audio, video, and computer "
                 "files used in the rendition of 3-dimensional models, such as texture "
                 "maps and polygon data in any format — generally are not subject to "
                 "copyright in the United States.\" "
                 "https://www.nasa.gov/nasa-brand-center/images-and-media/")

#: The publicity/privacy caveat NASA attaches to the same page. It bears on a COMMERCIAL
#: use of a recognisable person and not on copyright, so it does not affect these bytes
#: being redistributable in a test suite — but it is part of the terms and is recorded.
LICENCE_CAVEAT = ("NASA's guidelines add that if a NASA image includes an identifiable "
                  "person, using it for commercial purposes may infringe that person's "
                  "right of privacy or publicity, and that NASA insignia may not be used "
                  "without permission. Neither bears on copyright; both are recorded "
                  "because these images show identifiable people.")

#: (slug, full name, nasa_id, ~medium asset sha256, bytes, date, centre, credit, title).
#: The URL is derived from the id — ``images-assets.nasa.gov/image/<id>/<id>~medium.jpg`` —
#: so the id is the only name to keep in step.
FIXTURES = (
    ("whitson", "Peggy A. Whitson", "JSC2001-03044",
     "547ae82dc6b8b28133ae0064ba535fea9262efa636a9fdeb3b240903f69ae90c", 183546,
     "2001-11-28", "JSC", "NASA/Bill Stafford", "Official Portrait of Peggy Whitson"),
    ("whitson", "Peggy A. Whitson", "iss005e07178",
     "7a907c18ef44b0534aea53298ba2b4f8246ea18212ca1ccbeb4e809760302c1f", 165164,
     "2002-07-09", "JSC", "NASA", "Whitson works at the MSG in the U.S. Laboratory during Expedition Five"),
    ("whitson", "Peggy A. Whitson", "NHQ201803020004",
     "e38db34b5a8eea000f81d29d278cec43efd7e5efabdb2a2f937dd58c8d623662", 117074,
     "2018-03-02", "HQ", "NASA/Joel Kowsky", "Astronaut Peggy Whitson at NASM"),

    ("hopkins", "Michael S. Hopkins", "iss037e021314",
     "03509548036ab97122975f724668bd7701174581845d2959d97b1b34ff38a4b4", 79475,
     "2013-10-26", "JSC", "NASA/Karen Nyberg", "Hopkins in Node 2"),
    ("hopkins", "Michael S. Hopkins", "iss037e013962",
     "705406ec65346f15525fb95d80ac76f942e69ae0c1766123bfd71d91ddf185c7", 191953,
     "2013-10-15", "JSC", "NASA/Karen Nyberg", "Hopkins in Node 2"),
    ("hopkins", "Michael S. Hopkins", "iss064e025819",
     "672dde0822329542f770afbfa987884f9f77bfc220f9725fbfb664940e688ea9", 165728,
     "2021-01-26", "JSC", "NASA", "Expedition 64 Flight Engineer Michael Hopkins checks safety tethers"),

    ("williams", "Sunita L. Williams", "jsc2005e02663",
     "f07a5fbbd28521391b4c15aa8bcf466904958c3580bb0ed1a31db493826599a5", 147807,
     "2004-09-22", "JSC", "NASA/Mark Sowa", "Official Portrait of Astronaut Sunita L. Williams"),
    ("williams", "Sunita L. Williams", "iss015e07586",
     "1a4e78ac7c76250abc47d94cc3083ce1c447a3ffe812fc09ebda9506ffafa4cc", 155272,
     "2007-05-13", "JSC", "NASA", "Williams during SWAB experiment in the US Lab during Expedition 15"),
    ("williams", "Sunita L. Williams", "jsc2011e086095",
     "656b5ad7506199802f653c2013067eac4f928e8be88a3cfdd30c5ac0c500472a", 117541,
     "2011-09-08", "JSC", "NASA/Robert Markowitz",
     "Expedition 32 crew member Sunita Williams during her EMU Training and Certification"),

    ("lindgren", "Kjell N. Lindgren", "NHQ202009160011",
     "b69e05880dc5325c191f79c81c959f4fe4907bbe293072961776b311d9e2313d", 91007,
     "2020-09-16", "HQ", "NASA/Bill Ingalls", "Portrait - Astronaut Kjell Lindgren"),
    ("lindgren", "Kjell N. Lindgren", "iss045e168328",
     "4753da94e677820f1144ff5c35e9af6c576473e21eb5add492c8d7a87ee0dd45", 138775,
     "2015-12-02", "JSC", "NASA/Kjell Lindgren", "Lindgren conducts Veg-01 Plant Pillow Refill"),
    ("lindgren", "Kjell N. Lindgren", "NHQ202303310015",
     "c66b0a74b2d1042ddb2ff5203755dc03d46294da7df06d5db7fcc20eddd32351", 82855,
     "2023-03-31", "HQ", "NASA/Keegan Barber",
     "NASA's Crew-4 STEM Event at James W. Robinson Secondary School"),

    ("meir", "Jessica U. Meir", "NHQ202009150005",
     "c99c8f145e5083ecb0b5d47722a510ccfea94f47bb1a524c6b7a16fe4e2308e5", 90186,
     "2020-09-15", "HQ", "NASA/Bill Ingalls", "Portrait - Astronaut Jessica Meir"),
    ("meir", "Jessica U. Meir", "iss062e103558",
     "f3d095e74f2d079f01206b8b6e76274bcc3e75a9c4e0512ffa2696fd49f15fda", 134532,
     "2020-03-20", "JSC", "NASA/Jessica Meir", "IFM N3 MCA Mass Spectrometer Remove and Replace"),
    ("meir", "Jessica U. Meir", "jsc2025e078605_alt",
     "3177035d2ef10190bf921fc223cd2624293c75a6e74d9e32c071d31df1c08473", 151632,
     "2025-09-26", "JSC", "NASA/Josh Valcarcel",
     "Official portrait of NASA astronaut Jessica Meir wearing a spacesuit"),

    ("hague", "Nick Hague", "NHQ202001130002",
     "9e8775425b9bea22bb71ba936b1672c79089e4ffe6423534e1336ba5fca19151", 88284,
     "2020-01-13", "HQ", "NASA/Aubrey Gemignani", "Portrait of Astronaut Nick Hague"),
    ("hague", "Nick Hague", "iss059e036143",
     "d31f8bedfd15d1bc133387f42c54d203849b406d8e077db144499281ed59d4f3", 145430,
     "2019-04-29", "JSC", "NASA/Nick Hague", "NODE 3 Filter Remove and Replace"),
    ("hague", "Nick Hague", "NHQ202509120024",
     "a894b1d01a00df0db8fd634ba4e6c95ebd203daec0dfb459b9ee66699c9112bd", 66133,
     "2025-09-12", "HQ", "NASA/Bill Ingalls", "Astronaut Nick Hague Attends Joint Base Andrews Air Show"),
)

#: Rejected candidates worth keeping, because each names a failure mode a reader would
#: otherwise have to rediscover.
REJECTED = (
    ("NHQ201605250017", "Scott Kelly Post-Flight Visit to Washington",
     "passes rules 1-4; the single detected face is a DIFFERENT PERSON at the event. A "
     "caption names who a photograph is about, not whose face is in it."),
    ("(the whole Scott Kelly identity)", "several official NASA portraits",
     "identical twin brother, also a NASA astronaut. A different-person pair that is "
     "genuinely ambiguous to any embedder does not belong in a known-pairs fixture set."),
    ("iss062e014339", "ACE-T4 Module Configuration (Jessica Meir)",
     "yaw proxy +1.63 — beyond YAW_MAX; the crop is dominated by a gloved hand and the "
     "far-eye keypoint is extrapolated."),
    ("iss037e020099", "Hopkins at work in Quest airlock",
     "yaw proxy -2.30 — beyond YAW_MAX."),
    ("iss045e089495", "Ham Radio Session in Columbus (Kjell Lindgren)",
     "yaw proxy -2.27 — beyond YAW_MAX. The crop looks usable, which is the point of "
     "having a rule rather than an eye: the far eye is not observed."),
    ("KSC-20210715-PH-KLS02_*", "Victor Glover Tours VAB / O&C",
     "rule 1: the KSC records' rights line defers to a third party "
     "(\"For copyright and restrictions refer to ...\")."),
    ("jsc2013e067459", "Expedition 37 Crew News Conference",
     "rule 2: the caption names three crew members. The face IS Hopkins — confirmed "
     "against the two unambiguous Hopkins images — but a fixture should not rest on a "
     "caption a reader has to adjudicate."),
)


def asset_url(nasa_id):
    return f"https://images-assets.nasa.gov/image/{nasa_id}/{nasa_id}~medium.jpg"


def fetch(nasa_id, want_sha, want_bytes):
    cache = conv_dir() / "fixtures-src"
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"{nasa_id}.jpg"
    if not (path.is_file() and sha256_file(path) == want_sha):
        url = asset_url(nasa_id)
        print(f"[..] GET {url}")
        with urllib.request.urlopen(url, timeout=120) as response:
            path.write_bytes(response.read())
    got, size = sha256_file(path), path.stat().st_size
    if got != want_sha or size != want_bytes:
        raise SystemExit(f"{nasa_id}: {size} bytes / sha256 {got}; pinned {want_bytes} / "
                         f"{want_sha}. NASA re-encoded the asset, or the id moved — do not "
                         f"silently re-baseline a fixture's source.")
    return path


def yaw_proxy(kps):
    """Nose offset from the eye midpoint, in eye-spans. Not degrees, and not calibrated to
    degrees: it is a monotone stand-in for yaw whose only job is to place the ``YAW_MAX``
    cut, and it is computed from the same five points the alignment uses."""
    left_eye, right_eye, nose = kps[0], kps[1], kps[2]
    span = float(np.linalg.norm(right_eye - left_eye))
    if span < 1e-6:
        return float("inf")
    return float((nose[0] - 0.5 * (left_eye[0] + right_eye[0])) / span)


def main():
    from PIL import Image

    observed = observed_toolchain()
    oracle = align_oracle()
    detector = Detector(detector_path())
    out = fixtures_dir() / "faces"
    out.mkdir(parents=True, exist_ok=True)

    rows, failures = [], []
    for slug, person, nasa_id, sha, size, date, center, credit, title in FIXTURES:
        path = fetch(nasa_id, sha, size)
        image = np.asarray(Image.open(path).convert("RGB"))
        boxes, scores, kps = detector.detect(image)
        if len(boxes) != 1:
            failures.append(f"{nasa_id}: {len(boxes)} faces detected, expected exactly 1")
            continue
        width = float(boxes[0][2] - boxes[0][0])
        yaw = yaw_proxy(kps[0])
        if scores[0] < DET_SCORE_MIN or width < BOX_WIDTH_MIN:
            failures.append(f"{nasa_id}: score {scores[0]:.3f} / box width {width:.0f}px "
                            f"below ({DET_SCORE_MIN}, {BOX_WIDTH_MIN})")
            continue
        if abs(yaw) > YAW_MAX:
            failures.append(f"{nasa_id}: |yaw proxy| {abs(yaw):.2f} > {YAW_MAX}")
            continue

        landmarks = kps[0].astype(np.float32).astype(np.float64)
        matrix = oracle.similarity_transform(landmarks, oracle.ARCFACE_DST)
        aligned = oracle.warp_inter_linear(image, matrix, TEMPLATE_SIZE)

        stem = f"{slug}_{nasa_id}"
        crop_path = out / f"{stem}.rgb8"
        crop_path.write_bytes(aligned.tobytes())

        legible = Image.open(path).convert("RGB")
        legible.thumbnail((SOURCE_LONG_SIDE, SOURCE_LONG_SIDE), Image.LANCZOS)
        buffer = io.BytesIO()
        legible.save(buffer, "JPEG", quality=SOURCE_QUALITY, optimize=True)
        source_path = out / f"{stem}.jpg"
        source_path.write_bytes(buffer.getvalue())

        rows.append({
            "id": stem, "person": slug, "person_name": person,
            "crop": crop_path.name, "crop_sha256": hashlib.sha256(aligned.tobytes()).hexdigest(),
            "source": source_path.name, "source_sha256": sha256_file(source_path),
            "source_bytes": source_path.stat().st_size,
            "nasa_id": nasa_id, "nasa_url": asset_url(nasa_id),
            "nasa_asset_sha256": sha, "nasa_asset_bytes": size,
            "date_created": date, "center": center, "credit": credit, "title": title,
            "image_size": [int(image.shape[1]), int(image.shape[0])],
            "detection": {"score": float(scores[0]), "box_width_px": width,
                          "yaw_proxy": yaw,
                          "box": [float(v) for v in boxes[0]],
                          "landmarks5": [[float(x), float(y)] for x, y in landmarks]},
            "align_matrix": [[float(v) for v in row] for row in matrix],
            "licence": "public-domain-usgov",
        })
        print(f"[ok] {stem:38s} {date}  {width:5.0f}px  yaw {yaw:+.2f}  "
              f"crop {rows[-1]['crop_sha256'][:16]}…  src {rows[-1]['source_bytes']:6d} B")

    if failures:
        raise SystemExit("FIXTURE BUILD FAILED:\n  " + "\n  ".join(failures))

    people = sorted({r["person"] for r in rows})
    manifest = {
        "revision": 1,
        "template": {"size": TEMPLATE_SIZE, "layout": "hwc", "order": "rgb", "dtype": "uint8",
                     "bytes": TEMPLATE_SIZE * TEMPLATE_SIZE * 3},
        "alignment": {"oracle": "conversion/face/align_oracle.py",
                      "template": "ArcFace 5-point, insightface face_align.norm_crop",
                      "insightface_revision": INSIGHTFACE_REV},
        "detector": {"model": "det_10g.onnx (SCRFD-10GF) from insightface buffalo_l",
                     "role": "fixture cutting only; never converted, never published"},
        "selection_rule": {"det_score_min": DET_SCORE_MIN, "box_width_min_px": BOX_WIDTH_MIN,
                           "yaw_proxy_max": YAW_MAX,
                           "source_long_side_px": SOURCE_LONG_SIDE,
                           "source_jpeg_quality": SOURCE_QUALITY},
        "licence": {"basis": LICENCE_BASIS, "caveat": LICENCE_CAVEAT,
                    "spdx": "PD-USGov (not an SPDX identifier; there is no licence, the "
                            "works are not under copyright in the United States)"},
        "identities": people,
        "counts": {"images": len(rows), "identities": len(people),
                   "same_person_pairs": sum(
                       len([r for r in rows if r["person"] == p]) *
                       (len([r for r in rows if r["person"] == p]) - 1) // 2 for p in people),
                   "different_person_pairs": len(rows) * (len(rows) - 1) // 2 - sum(
                       len([r for r in rows if r["person"] == p]) *
                       (len([r for r in rows if r["person"] == p]) - 1) // 2 for p in people)},
        "rejected": [{"nasa_id": a, "title": b, "reason": c} for a, b, c in REJECTED],
        "toolchain": observed,
        "faces": rows,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    total = sum(r["source_bytes"] for r in rows) + len(rows) * TEMPLATE_SIZE * TEMPLATE_SIZE * 3
    print(f"\n[ok] {len(rows)} crops, {len(people)} identities, "
          f"{manifest['counts']['same_person_pairs']} same-person pairs / "
          f"{manifest['counts']['different_person_pairs']} different-person pairs, "
          f"{total / 1024:.0f} KiB committed")
    print(f"[ok] wrote {out / 'manifest.json'}")

    print("\n--- PROVENANCE.md table ---")
    print("| fixture | person | date | centre | credit | NASA id | box px | yaw proxy |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| `{r['id']}` | {r['person_name']} | {r['date_created']} | {r['center']} | "
              f"{r['credit']} | [`{r['nasa_id']}`]({r['nasa_url']}) | "
              f"{r['detection']['box_width_px']:.0f} | {r['detection']['yaw_proxy']:+.2f} |")


if __name__ == "__main__":
    main()
