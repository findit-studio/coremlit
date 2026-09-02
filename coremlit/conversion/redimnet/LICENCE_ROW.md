# The licence row and the `MODELS_LOCK` table — the analysis behind them, and the coupling that once kept them out

**Status.** The row, the table and the `identity` shard landed together in #136 for
`redimnet_b5.mlmodelc`, once the artifact repository existed (private, HF `80c2d0a`). B5
is the registered artifact. **B2 and B2-ptn are converted, measured and preserved in the
same repository but deliberately NOT registered** — no row, no table entry, no gated test
— for the reason `README.md`'s "B2: converted, measured, not registered" records. If that
decision is ever reversed, each B2 bundle's row is the B5 row below with the asset, the
`weights/weight.bin` SHA-256 and the pin locator changed (its own `const` in
`tests/identity/common/mod.rs`, because three bundles that all carry a
`weights/weight.bin` cannot share a locator), and the `ptn` row must additionally state
that no published metric of any kind exists for that checkpoint. What follows is the
licence ANALYSIS the B5 row cites as its `source`, and the record of why none of the three
artifacts could land alone — kept because the coupling it describes is still how this
repository's checks work.

---

*The original analysis, written before the artifact repository existed:*

Issue #123 asks for both. Both were drafted here **verbatim and ready to paste**, and
neither could be committed to `coremlit/tests/model_licences.rs` or to `MODELS_LOCK` at the
time, because committing either one alone turned a green gate red. The reason was a real
coupling in this repository's own checks, not caution.

## The coupling: three artifacts, none of which can land alone

```
                 licence row  ──requires──▶  MODELS_LOCK table  ──requires──▶  ci.yml shard
                (model_licences.rs)                                          (+ gates to run)
                        ▲                                                            │
                        └───────────── all three require ────────────────────────────┘
                                a PUBLISHED artifact repo + immutable revision
```

**1. A licence row whose `staged_by` names no `MODELS_LOCK` table is a hard failure.**
`coremlit/tests/model_licences.rs:1305-1310`, inside `unmatched_coverage`, reached from the
plain (un-`#[ignore]`d, un-feature-gated) test
`every_staged_file_has_a_licence_row_and_every_row_names_a_staged_file`
(`model_licences.rs:2243-2270`):

```rust
  for repo in claimed.difference(&staged) {
    failures.push(format!(
      "a licence row is staged_by {repo:?}, which no MODELS_LOCK table names. Either the table \
       was removed and the row is describing bytes CI no longer fetches, or the name is a typo."
    ));
  }
```

That is direction 1 of the register's three, and the file has a hermetic falsifier for
exactly this case (`direction_one_reds_when_a_row_names_no_staged_repo`,
`model_licences.rs:3068-3082`), so the behaviour is pinned and cannot be argued with.

**2. A `MODELS_LOCK` table whose `kit` has no `ci.yml` shard is a hard failure.**
`coremlit/tests/whisper/models_lock.rs:521-546`:

```rust
  assert_eq!(
    lock_kits, shard_kits,
    "MODELS_LOCK's kits and ci.yml's model-tests shards have drifted apart. A kit in the lock \
     with no shard is a download nothing performs; a shard with no table downloads nothing and \
     then gates a bare checkout. Add the missing table or the missing matrix row."
  );
```

Today's matrix is `ced, whisper, speaker, granite, clap, siglip, lid`. Adding
`kit = "redimnet"` without a matching shard reds this test.

**3. There is no sanctioned exemption.** `Key::Unpinned` and `Key::Unmanifested` relieve a
row of needing a *byte identity*; both are explicitly scoped to rows that still have a
staging table (`model_licences.rs:424-425` and `438-440`). `CHECKSUMLESS_KITS`
(`models_lock.rs:88-105`) exempts a kit from checksum verification, not from being staged.
Nothing in the register models a "not published yet" artifact.

**4. And the missing precondition is not code.** All three need a published artifact
repository (the analogue of `FinDIT-Studio/cedkit-coreml`) at an immutable commit SHA. That
is a publishing action, not an edit — and inventing a revision here would be precisely the
"record a version nobody ran" defect the rest of this recipe is built to prevent.

`model_licences.rs` is otherwise fully hermetic — it reads `MODELS_LOCK`, `Cargo.toml`, and
source text only, never the downloaded bytes — so once the table and the shard exist, the
row below can be pasted and evaluated with no model present.

## The licence row, as it must read

Paste into the `ARTIFACTS` table in `coremlit/tests/model_licences.rs`. Two fields are
placeholders and are marked; everything else is final.

```rust
  Artifact {
    file: "redimnet/redimnet_b5.mlmodelc/weights/weight.bin",
    // MEASURED on the conversion run recorded in conversion/redimnet/README.md. It must be
    // RE-READ from the published artifact before this row lands: the .mlmodelc is produced
    // by `xcrun coremlcompiler`, whose output the Python toolchain does not pin (the same
    // caveat conversion/granite records), so the publishing run's bytes are the ones a pin
    // may name.
    key: Key::Sha256("1735fc68f4cdf10ad8bb56135da3bd8c0c83f6c3549ee8514f0346046f90a79b"),
    pin: "tests/redimnet/model_io.rs::B5_SHA256",          // does not exist yet — see README
    staged_by: "FinDIT-Studio/redimnetkit-coreml",          // repository not published yet
    loader: "src/audio/mod.rs::identity",                   // door not written yet — see README
    gate: "identity",
    weights: Terms::unresolved(
      "NO WRITTEN GRANT COVERS THESE BYTES, and that is a step DOWN in artifact-level clarity \
       from the incumbent rather than a step across. `IDRnD/redimnet` ships MIT, but the grant \
       is written over \"the Software\" — the model source — and neither that repository nor \
       `PalabraAI/redimnet2` extends it to the released `.pt` assets in writing. Compare the \
       row this would sit beside: WeSpeaker's own model-licence document places its \
       VoxCeleb-trained pretrained models under CC-BY-4.0, an explicit weights grant with \
       attribution as a CONDITION, which is why `speakerkit/wespeaker_v2.mlmodelc` is an \
       attribution row and this one is not. The corpus layer below is the binding constraint \
       and it is unchanged, so this does not disqualify the shipping path; what it does is \
       remove a written permission we previously had, and the register should show that as \
       `unresolved` rather than borrow the source licence's identifier for weight bytes it \
       does not name. Re-tagging an upstream CODE licence onto a weights artifact is the \
       exact conflation this campaign has already paid for once — `aufklarer/\
       ReDimNet2-B6-CoreML` declares `license: mit` over VoxBlink2-trained weights whose \
       corpus is CC-BY-NC-SA-4.0.",
    ),
    corpus: Terms::attribution(
      "CC-BY-4.0",
      CREDIT_AUTHOR_VOXCELEB,
      "VoxCeleb2-dev, and NO NEW EXPOSURE: this is the same corpus lineage the incumbent \
       WeSpeaker embedder already carries, so the decision it needs has already been taken. \
       The `-vox2-` lineage is the only one usable here — the same upstream release publishes \
       `M-vb2+vox2+cnc-ft_mix.pt` and `S-vb2-ptn.pt` trained on VoxBlink2, whose distributor \
       states the CC-BY-NC-SA-4.0 term propagates to the trained model (\"The license of the \
       model is also CC BY-NC-SA 4.0, no commercial application is allowed\"). The conversion \
       recipe refuses any asset whose name is not `-vox2-` \
       (`conversion/redimnet/scripts/_redimnet_common.py::verify_asset_name`), so the \
       distinction is enforced at the point the bytes are loaded rather than remembered here.",
    ),
    source: "conversion/redimnet/README.md; IDRnD/redimnet LICENSE (MIT, over \"the Software\"); \
             wenet-e2e/wespeaker model licence; voxblink2.github.io",
  },
```

Two notes on the modelling, both deliberate:

- **`Terms::unresolved` and not `Terms::permissive("MIT", …)`.** MIT is what the *source
  repository* is under. Claiming it over the weight bytes would put an identifier in the
  register that no document attaches to those bytes, and
  `identical_bytes_carry_identical_terms` exists precisely to catch a row that claims more
  than the upstream said.
- **`unresolved` is not disqualifying, and it is NOT unchecked.** `Terms::ResearchOnly`
  is still the only verdict that sets `forbids_commercial_use`, and that is the axis the
  `commercial-` prefix rule hangs on — so this row needs no `commercial-`prefixed gate and
  `identity` stays a plain feature. That is a different statement from "invisible", which
  is what it used to mean: `Terms::permits_a_shipping_claim` is a SECOND axis, false for
  `Unresolved` as well as for `ResearchOnly`, and
  `no_ungranted_artifact_is_reachable_from_default` refuses to let this row be reachable
  from the default feature set — reading `default` through the real TOML parser, so the
  refusal holds for every spelling Cargo obeys and not only the one the check was first
  mutation-tested in. Adding a `commercial-` gate over it would not trip
  direction 3 either — `every_commercial_feature_gates_an_artifact_with_no_shipping_grant`
  accepts an unresolved row as a cause a gate may stand on, and says in its own failure
  text that the cause is an open question rather than a found prohibition.

## The `MODELS_LOCK` table, as it must read

```toml
# The ReDimNet-B5 speaker-embedding artifact (`audio::identity`, issue #123). One bundle,
# 15 MiB fp16, converted by `coremlit/conversion/redimnet` from the OFFICIAL public
# `IDRnD/redimnet` release asset `b5-vox2-ft_lm.pt` (sha256 8b0c11bb…, VoxCeleb2-dev).
#
# The graph is `mel [1, 72, 401] f32 -> embedding [1, 192] f32`, RAW (un-normalized). The
# mel front end runs in the CALLER, which is not a style choice: the waveform-in variant
# is exact in fp32 but wrong in fp16 on EVERY compute unit (worst cosine 0.28 on the
# default `All` placement), because its power spectrogram overflows fp16 at the top and
# its `log(x + 1e-6)` guard is subnormal at the bottom. Measured in the recipe's README.
#
# The revision is the ARTIFACT repo's commit SHA. Do not confuse it with the upstream
# SOURCE pins — the `.pt` asset's sha256 and the `IDRnD/redimnet` model-source revision —
# which name what the conversion consumed and live in `scripts/_redimnet_common.py`.
["FinDIT-Studio/redimnetkit-coreml"]
kit       = "identity"
include   = "redimnet_b5.mlmodelc/* CHECKSUMS.sha256"
revision  = "<artifact repo commit SHA — repository not published yet>"
local-dir = "Models/redimnet"
```

The bundle ships `CHECKSUMS.sha256` and `MANIFEST.json`, both emitted by
`scripts/write_manifest.py`, so `ci.yml`'s checksum-verification step applies and the kit
does **not** belong in `CHECKSUMLESS_KITS`.

## The order this was done in (#136), and the order a fourth bundle follows

1. Write the Rust door (`src/audio/identity`), including the caller-side mel front end from
   `README.md`'s table, and its `tests/identity/` gates — this is what creates `loader`,
   `gate` and `pin`. Done in #136; a further bundle behind the same door needs none of it.
2. Run this recipe on the publishing machine (`run_redimnet.sh <variant>` into the publish
   root, which already holds the other bundles); publish the tree to the artifact repository
   and read back its commit SHA and each bundle's `weights/weight.bin` SHA-256.
3. Land the `MODELS_LOCK` table (its `include` widened to the new bundle and its
   `revision` bumped), the licence row keyed on the new bundle's `weights/weight.bin` with
   its OWN pin `const` in `tests/identity/common/mod.rs`, and the artifact's entry in that
   file's `ARTIFACTS` table **in one change** — until the revision is bumped, the new rows
   name a bundle the pinned revision does not stage, and the CI shard's download stages
   nothing for them.
