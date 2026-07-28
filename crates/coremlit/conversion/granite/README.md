# granite-embedding-97m-multilingual-r2 CoreML conversion

Re-derives the **granite** text encoder that `coremlit::embeddings::granite` runs,
converted **from the official public checkpoint** — nothing is consumed from the
published `FinDIT-Studio/embedkit-coreml` artifact repo. Local staging only.

**"Re-derives" means re-derivable to the floors below, NOT bit-reproducible.**
Neither the CoreML compiler nor torch's fp32 reduction order is pinned by these
versions. See [Byte-identity](#byte-identity-measured-not-claimed) for what
actually matched.

## What the gates defend against — and what they do not

Every precondition in this recipe is aimed at **mistakes and failed runs**: a step
that errored, a stale file left by an earlier build, the wrong environment, an
accidental self-comparison, a partially written pair of goldens. Those are the
failures that actually occur when someone re-runs one stage out of order, and they
are what the gates below catch.

That list is a claim about what is *checked*, not a statement of intent, and one
member of it was not true until recently: "a partially written pair of goldens"
was listed here while **nothing downstream read the published crosscheck at all**
— it was absent from the evidence bond and neither `verify_granite.py` nor
`write_manifest.py` referenced it. It is checked now
([below](#the-goldens-are-a-pair-and-are-checked-as-one)); the fix is what made
the sentence accurate, not a rewording of it.

They do **not** defend against someone with write access to the staging tree
between stages. A script pipeline cannot self-certify against that: every gate it
adds is itself a file the same access could edit. Two scenarios are explicitly
outside the boundary and are NOT claimed to be caught:

- **Substituting a shipped file after verification** — in the window between
  `verify_granite.py` and `write_manifest.py`. The evidence bond catches a plain
  substitution because it re-hashes, but a substitution that also rewrites
  `verify_metrics.json` defeats it.
- **A deliberately displaced length witness** — hand-editing `corpus.json` so some
  other entry carries the 512-token sequence. The corpus check ties the witness to
  the specific over-length fixture and `verify_granite.py` re-tokenizes every
  entry, so reaching this requires coordinated editing, not an accident.

Two further limits are about how the run is executed rather than about an
adversary, and they bound every "no interruption leaves…" statement below:

- **Power loss is not ordered, and no ordering is claimed.** Those statements are
  about this *process* stopping — an exception, ENOSPC, a Ctrl-C, a SIGKILL —
  which an atomic rename covers on its own. macOS `fsync` only hands bytes to the
  drive; `man 2 fsync` says the drive "may not physically write the data to the
  platters for quite some time and it may be written in an out-of-order sequence"
  so that "later writes may be present, while earlier writes are not", and adds
  "This is not a theoretical edge case." The recipe still fsyncs, which bounds what
  a kernel panic can lose, but recovery after a power cut is simply to **re-run the
  conversion**: the artifacts are regenerable, and the gates fail closed on a torn
  publication rather than certifying it.
- **One run at a time.** Nothing locks the artifact root or the staging tree. Two
  concurrent runs sharing `$GRANITE_STAGE` already clobber each other's records and
  packages in steps 1-2, and two concurrent promotions in step 3 can interleave
  their renames — the stale-scratch refusal runs at start-up, so it does not
  serialise them. This is an owner-run pipeline; a lock is deliberately not part of
  it.

Stating this is itself part of the provenance claim; nothing else in this document
should be read as promising more.

## Source (pinned)

- Repo: [`ibm-granite/granite-embedding-97m-multilingual-r2`](https://huggingface.co/ibm-granite/granite-embedding-97m-multilingual-r2)
  — **Apache-2.0**, ungated. Weights ship bf16; everything here runs them as a
  lossless fp32 upcast.
- Revision: `835ad14087e140460703cf0fae09f97d469d65c2`
- Per-file SHA-256 (verified on load, fail-closed — `scripts/_granite_common.py`):
  - `model.safetensors` — `f3ea88b230492811046145513710e76b4cc8c2ad49e8708da0e7247e548903be`
  - `tokenizer.json` — `4f2842d568e2724370aec203652a42ac783c7937f8347a1a2cc7506d71f1582f`
    (this file is COPIED INTO the published artifact by `write_manifest.py` —
    the Rust crate no longer embeds it and `TextEmbedder::load` reads it from
    beside the `.mlmodelc` — so this digest is also
    `granite::contract::TOKENIZER_SHA256_HEX`, the identity the crate enforces at
    load and `tests/granite/tokenizer_identity.rs` asserts)
  - plus `config.json`, `tokenizer_config.json`, `special_tokens_map.json`,
    `config_sentence_transformers.json`, `sentence_bert_config.json`,
    `modules.json`, `1_Pooling/config.json` — pinned because the graph shape
    (layer types, RoPE thetas, window, `norm_eps`) and the pooling/prompt
    contract are READ from them rather than hardcoded.

## Toolchain

`python 3.11`, `torch==2.6.0`, `transformers==5.14.0`,
`sentence-transformers==5.6.0`, `coremltools==9.0`, `numpy==1.26.4`. pip resolves
that exact set with no substitutions.

These are pinned in `_granite_common.REQUIRED_TOOLCHAIN`, and the one shared
`observed_toolchain()` reads what is **actually** running (the live interpreter
and the installed distributions), asserts it equal to the pins, and returns only
what it observed. `convert_granite.py` and `verify_granite.py` both call it, and
verification refuses to continue if the two disagree — the pins allow any 3.11.x,
so converting under 3.11.14 and verifying under 3.11.15 is two environments for
one artifact. The producer's reading is carried in the digest-bound
`verify_metrics.json`; `MANIFEST.json` copies it from there rather than sampling
the writer's shell, which can trivially be a third environment.
`generate_goldens.py` records its own observed reading into `corpus.json`'s
`_provenance`. No hardcoded version dict remains anywhere else.

transformers **5.14.0** is load-bearing and this recipe does NOT share the
4.53.3 venv the `clap`/`siglip`/`ced` recipes use. In 5.14.0 ModernBERT reads its
RoPE parameters from `config.rope_parameters[layer_type]`, builds attention masks
through `transformers.masking_utils.create_bidirectional_*`, and dispatches masks
and RoPE per `config.layer_types`. The driver below calls those APIs directly, so
a venv where they are absent or shaped differently cannot run this recipe —
`_granite_common.py` asserts the config surface it depends on before converting,
so such a venv fails loudly rather than cutting a wrong artifact.

## Contract

| | name | dtype | shape |
|---|---|---|---|
| input | `input_ids` | int32 | `[1, 512]` |
| input | `attention_mask` | int32 | `[1, 512]` |
| output | `embedding` | float32 | `[1, 384]` |

The output is the **pre-L2-norm** CLS vector (`hidden_states[:, 0]` after the
final LayerNorm) — the Rust caller normalizes, which keeps the fp16 rsqrt guard
class out of the graph. Fixed sequence length 512; RoPE makes any fixed length
sound, and longer inputs are the caller's windowing layer's job.

Retrieval is **prompt-free** (`config_sentence_transformers.json` carries empty
query/document prompts), asserted before any golden is cut.

## The static-graph rewrite (why there is a driver at all)

The stock `ModernBertModel.forward` recomputes, per call, the dual RoPE tables
and both attention masks — data-dependent work that cannot lower to one static
CoreML graph. `GraniteGraph` (`scripts/_granite_common.py`) drives the STOCK
submodules (`embeddings`, every `ModernBertEncoderLayer`, `final_norm`) unchanged
and hoists exactly those two things to fixed-512 constants:

- **RoPE** — `cos`/`sin` per layer type, produced by the checkpoint's own
  `rotary_emb` at positions `0..511`, held as fp32 buffers. Global θ=150000 on
  the full-attention layers (0/3/6/9), local θ=160000 on the eight sliding ones.
- **Attention masks** — the sliding-window geometry is a constant bool taken from
  the checkpoint's OWN `create_bidirectional_sliding_window_mask` (probed twice,
  once with the last position padded and once with the first, then OR-ed to
  cancel the pad component; independently cross-checked against the documented
  ±64 band). Only the pad component depends on the runtime `attention_mask`, so
  the additive masks are rebuilt in-graph as `where(allowed, 0, -1e4)`.

Faithfulness is proven BEFORE tracing, against the UNMODIFIED canonical
sentence-transformers pipeline over all 16 corpus entries. `convert_granite.py`
computes that measurement, gates the conversion on it, and writes it to staging;
`generate_goldens.py` publishes THAT record verbatim as
`tests/granite/fixtures/goldens/driver_crosscheck.json`, adding only the
`corpus_sha256` pair binding, and refuses to run if it is absent, belongs to a
different run, or measured different inputs than the goldens are being written
from ([below](#what-the-crosschecks-two-corpus-digests-each-prove)). The goldens
step never recomputes it — a later recomputation would be a different measurement
of a different graph instance, however close its numbers.
`tests/granite/driver_crosscheck.rs` gates the published file.

### `-1e4`, not `finfo.min`, is the mask block value

The shipped artifact is fp16 and `torch.finfo(torch.float32).min` overflows to
`-inf` in coremltools' fp16 cast. A fully-padded query row is then all `-inf`,
whose softmax is NaN. **Measured** on the `-3.4e38` build: `CpuOnly` returned
non-finite output on **15/16** corpus entries, while `CpuAndGpu`,
`CpuAndNeuralEngine`, and `All` stayed clean — a one-arm failure that a
check exercising only the non-`CpuOnly` requests would have shipped. `-1e4` is exactly representable in
fp16 and far below any attention logit this graph produces, so blocked keys
underflow to zero in the softmax and an all-blocked row degrades to a uniform,
finite distribution. It is also the value in the published artifact's mask
constants (read back from its `weights/weight.bin`: allow `0.0`, block
`-10000.0`, in both the `[1,1,1,512]` global and the `[1,1,512,512]` local mask).

## Measured verification (this machine; `scripts/verify_granite.py`, fail-closed)

All 16 committed corpus entries; the fp32 reference is the fp32 CoreML build on
`CpuOnly`; cosine is computed in float64.

| regime | worst cosine | floor | non-finite |
|---|---|---|---|
| **fp32 · CpuOnly** vs the committed transformers-fp32 goldens | **1.00000000** | 0.99999998 | 0/16 |
| **fp16 · CpuOnly** vs fp32 | **0.99997885** | 0.99996 | 0/16 |
| **fp16 · CpuAndGpu** vs fp32 | **0.99999934** | 0.99996 | 0/16 |
| **fp16 · CpuAndNeuralEngine** vs fp32 | **0.99996360** | 0.99996 | 0/16 |
| **fp16 · All** vs fp32 | **0.99999926** | 0.99996 | 0/16 |

Unlike siglip, **no arm here is merely characterized** — granite ships fp16 on
the default planner, so every compute unit is floor-gated and NaN-gated.

Every attestation the recipe writes — the producer record, the staged crosscheck,
the compilation record, `verify_metrics.json`, `MANIFEST.json` — carries the id of
the conversion run that produced it, and every consumer requires that id to match.
Conversion mints a fresh id and discards all four prior records BEFORE any
failure-prone work, so a conversion that dies while loading, checking the driver
or tracing leaves nothing a later stage can mistake for a result.

### What binds the COMPILED bundle

`producer.json` is written at the end of conversion and binds the two
`.mlpackage`s. It cannot cover the `.mlmodelc` — compilation happens after it —
and the `.mlmodelc` is what ships and what every Rust gate loads. Left open, that
gap is reachable with no adversary at all: run A completes, run B converts and
then resumes at staging without recompiling, A's bundle is copied beside B's
packages, root-versus-staging equality still holds because they are the same
bytes, every floor still clears, and the manifest stamps run B's identity onto
compiled bytes run A produced.

Two things close it. `begin_run` deletes every staged `.mlmodelc` along with the
four records, so the skipped-compilation resume fails at the copy instead of
succeeding quietly. And `compile_granite.py` — which replaced the shell's compile
loop, so the record cannot be separated from the compile — writes `compile.json`:
run id, input package digest, output bundle digest, and the compiler environment
(`coremlcompiler` version, developer dir, macOS product and build).
`verify_granite.py` requires it before it measures anything and carries it into
`verify_metrics.json`; `write_manifest.py` re-reads it and refuses to publish if
it has moved since verification. The compiler is recorded because it is the part
of this recipe the Python pins do not cover: two runs here produced identical
`model.mil` and `weight.bin` while their compiled `coremldata.bin` differed.

### Publication is one transition, not two writes

`CHECKSUMS.sha256` and `MANIFEST.json` are not part of the bundle replacement, so
they used to outlive the bytes they describe — a run that replaced the bundles and
then failed verification left an artifact root still carrying a complete,
internally consistent attestation for a build that was no longer in it.

Now step 3 removes both before it replaces the bundles, and `write_manifest.py`
removes them again — but **only immediately before it publishes**, after every
precondition has passed and both payloads are fully built. Each then goes down as
one rename of a unique temp file, **manifest first, checksums last**. That
ordering is the invariant: `CHECKSUMS.sha256` present implies a `MANIFEST.json`
from the same run is beside it. An interruption leaves the root unpublished, or
carrying a manifest with no checksums — visibly unfinished. It can no longer leave
a truncated checksum list, which `shasum -c` accepts as a full verdict because it
only checks the lines it is given, and it can no longer leave new checksums beside
a manifest from another run.

**"Interruption" means this process stopping** — an exception, ENOSPC, a Ctrl-C, a
SIGKILL — which is exactly what an atomic rename covers. It is not a power-loss
claim. The recipe fsyncs around these renames, but macOS `fsync` only hands the
bytes to the drive, which `man 2 fsync` says may still write them "in an
out-of-order sequence"; real ordering would need `F_FULLFSYNC` and even that would
not make a multi-file publication recover as a unit. Recovery from a power cut is
to re-run the conversion — these artifacts are regenerable, and the gates fail
closed on a torn publication rather than certifying it.

The timing matters as much as the ordering. Invalidating on entry meant the
independently-runnable writer destroyed a valid attestation whenever any
precondition failed — an unset `GRANITE_GOLDENS` or a mistyped stage path removed
both files and only then aborted, and the directory sync pushed that loss straight
out to the drive. A failed precondition now leaves an existing publication exactly
as it was.

**Nothing in this recipe deletes a temp file to recover.** A publication or
model-card copy killed between writing its temp and renaming it leaves
`<name>.<hex>.tmp` behind, and `assert_no_stale_publication_temps` names it, says
it will not remove it, and prints the `rm` to run. An automatic sweep was tried
and was wrong three times about which names only this recipe could produce —
prefix matching ate `README.md.notes.tmp`; a `<32 hex>` match still ate
`README.md.000…0.tmp`, which `uuid4` cannot emit; and a UUIDv4 grammar would
still be a guess. Each round narrowed an irreversible blast radius without
removing it, so the destructive step is gone rather than narrowed again. This is
the same call [`assert_no_os_sidecars`](#byte-identity-measured-not-claimed)
already makes for AppleDouble files.

### The goldens are a pair, and are checked as one

`generate_goldens.py` writes `corpus.json` and `driver_crosscheck.json` as two
renames, and the crosscheck names the corpus bytes it belongs to. Until this was
fixed **no later stage read the crosscheck at all** — it was missing from the
evidence bond and neither `verify_granite.py` nor `write_manifest.py` mentioned
it. So a regeneration killed between the two renames left the new corpus beside
the previous run's crosscheck (`corpus.json` differs at minimum through
`_provenance.generated_utc`), and verification, `MANIFEST.json` and
`CHECKSUMS.sha256` then all attested successfully to a pair that
`tests/granite/driver_crosscheck.rs` rejects. No tampering needed — a killed
process, which is exactly the class this recipe's boundary claims.

Three things now close it, mirroring what the manifest pair already had:

- **Generation.** The previous pair is invalidated before either half is
  replaced, then the corpus lands and the crosscheck lands last. The crosscheck
  is what names the corpus, so its presence can only mean the corpus it names is
  already there. Invariant: **`driver_crosscheck.json` present implies the
  `corpus.json` it names is beside it.** The reachable states are neither file,
  the corpus alone — every gate fails loudly on the missing crosscheck — or both
  from one run.
- **Verification.** `require_published_crosscheck` loads the crosscheck before
  the corpus is even parsed and requires BOTH bindings, UNCONDITIONALLY:
  `corpus_sha256` must name the corpus bytes on disk, and `corpus_input_sha256`
  must be present and must match the corpus's ordered inputs. Success is recorded
  explicitly as the `golden_pair` check.

  An earlier version required the input digest only when the crosscheck's
  `run_id` matched the run being verified, so a plain replay against an older
  committed pair would not fail. That was unsound, and it is recorded here rather
  than quietly dropped: the committed pair carries a `run_id` and no digest, so on
  ANY fresh run the ids differ AND the field is absent — neither branch fires and
  the pair is accepted. It also accepted a record the old generator had
  republished over an EDITED corpus, because that generator stamped the new
  `corpus_sha256` on while leaving the measurement untouched. A conditional guard
  whose condition is false in the common case is an unguarded claim wearing a
  guard's clothes.
- **The bond.** `driver_crosscheck.json`'s digest is in `evidence_digests`, so a
  crosscheck swapped after verification breaks the manifest step the same way a
  swapped artifact does.

> **The committed pair predates `corpus_input_sha256`, so `verify_granite.py`
> REFUSES to attest until the goldens are regenerated by a conversion run**
> (`GRANITE_REGEN_GOLDENS=1`, owner-gated). The recipe is loudly unrunnable
> end-to-end until then. That is the intended failure, not an oversight: the
> alternative is accepting a record nothing ties to its measurement.

What the two digests together **catch**: a stale pair, a mismatched pair, a
regeneration interrupted between the two renames, and a fixture edit published
without re-running the conversion. All four are mistakes or failed runs — the
class this recipe's gates claim.

What they **do not catch**: a digest computed from the corpus it is stamped
beside and written into the record by hand. It matches by construction, and
nothing in the recipe can tell it apart from one a conversion recorded. That is
circumvention — editing a fixture specifically to defeat the refusal above — not
a mistake or a failed run, so it sits outside the boundary, alongside the two
scenarios named at the top of this document. Binding a record to a real
conversion of these exact texts would need either re-deriving the measurement (a
conversion run) or a signed attestation. Neither exists here, and neither is
claimed.

### What the crosscheck's two corpus digests each prove

They are not the same claim and they do not have the same value.

`corpus_input_sha256` is computed DURING the measurement, over the ordered
`(id, text)` it consumed, and `generate_goldens.py` requires it to equal the
ordered inputs the goldens are being written from. That is the measurement-input
binding. Matching run ids never provided it: a fixture edit does not touch the
packages, so editing a `text` in `_fixtures.py` and re-running only the goldens
step used to pass every id, shape and run-identity check while republishing a
measurement taken over the PREVIOUS texts, relabelled with the new corpus's
digest. It now reds.

`corpus_sha256` is stamped on afterwards, over the serialized `corpus.json` bytes,
which do not exist while the measurement runs. It binds the two goldens as a
published PAIR and nothing more.

**What is still not bound.** The goldens' embedding VALUES are recomputed by the
goldens step. They are tied to the conversion-time measurement through the shared
ordered input and the asserted toolchain, not by comparison against the canonical
vectors the crosscheck actually scored — those are not published, and they cannot
be compared byte-for-byte anyway, because torch's fp32 reduction order moves
components by up to 2.0e-7 between runs (measured, above). And the COMMITTED
`driver_crosscheck.json` predates `corpus_input_sha256`: it gains the field at the
next regeneration, so `tests/granite/driver_crosscheck.rs` does not check it yet —
while `verify_granite.py`, which does require it, refuses to run against that pair
at all until it is regenerated.

`verify_granite.py` discards any previous verdict before it measures anything and
writes `verify_metrics.json` only after every check has passed, by atomic rename —
so a failing run leaves no evidence at all rather than a usable-looking file. The
record carries the numbers above, explicit success flags for the non-numeric
checks (I/O contract, tokenizer identity, corpus identity, golden-pair identity),
the PRODUCER's toolchain, and SHA-256 digests of the exact bytes measured: the
staged bundle, the fp32 reference package, and BOTH committed goldens —
`corpus.json` and `driver_crosscheck.json`. It also asserts the shipped tree is byte-identical to
the staging build, so verification cannot measure one package while another ships.
`write_manifest.py` re-checks all of that before writing anything, and takes the
toolchain from that record rather than resampling its own environment.

Pre-trace, in float64 over the same 16 entries:

- **driver vs the canonical pipeline** (against the COMMITTED goldens): worst
  **0.9999999999996902** (entry `special`), floor 0.99999997.
- **traced vs eager**: worst **0.9999999999999999**, floor 0.99999997.

The five artifact rows above reproduce the published model card's measured table
to all eight printed decimals.

## Byte-identity (measured, not claimed)

`shasum -a 256 -c` of the re-derived tree against the published
`CHECKSUMS.sha256` (`FinDIT-Studio/embedkit-coreml` @ `81852f70`): **3 of 9 files
matched, 6 did not.** The revision is named because it is the one this
measurement was taken against; it predates the artifact-root `tokenizer.json`,
so that file is not among the 9 and the counts here do not move when
`MODELS_LOCK` advances to a revision carrying it. (The tokenizer is staged
verbatim from the digest-verified source snapshot, so byte-identity for it is
guaranteed by construction rather than measured.)

- **Matched**: `granite_97m_512.mlmodelc/weights/weight.bin` and
  `granite_97m_512.mlpackage/Data/com.apple.CoreML/weights/weight.bin` — every
  fp16 weight, both RoPE tables, and both mask constants are bit-identical.
  (`README.md` also matched: the model card is *staged* by the recipe, never
  generated by it, so it matches trivially and is not evidence of re-derivation.)
- **Differed**: `model.mil`, `model.mlmodel`, `metadata.json`,
  `coremldata.bin` ×2, and the mlpackage `Manifest.json`.

`write_manifest.py` pins the published 10-path set in
`_granite_common.EXPECTED_ARTIFACT_FILES` and compares it against a **recursive,
unfiltered walk of the whole artifact root** before writing, so `CHECKSUMS.sha256` is always
set-comparable with the published manifest: a missing path, or a stray file
anywhere under the root — beside the bundles or nested inside one — is a hard
failure, not a note. Only the two files this step itself generates
(`CHECKSUMS.sha256`, `MANIFEST.json`) are excluded. Nothing is excluded by name:
macOS AppleDouble (`._*`) and `.DS_Store` files are real files that appear on
their own on exFAT/FAT/SMB volumes, so the walk surfaces them and the step stops
with the removal command rather than excusing them and hiding whatever carries
those names. Two members are not produced by the conversion itself and are
staged by this step instead: the model card (already staged, or
`GRANITE_MODEL_CARD` points at it — the step fails otherwise) and
`tokenizer.json`, copied verbatim from the verified source snapshot. Both are
digest-checked on the way in, so a stale or wrong copy fails rather than being
published. The tokenizer is not optional packaging: the crate loads it from the
artifact root, so an artifact without it has no working default constructor.

The differences are container-level, and were characterized rather than assumed:
`model.mil` has the same 1051 ops with an **identical 493-op non-const
sequence** — only `const` emission naming and ordering move; `metadata.json`
differs in exactly one key, `com.github.apple.coremltools.conversion_date`; the
mlpackage `Manifest.json` carries a fresh UUID. **The pipeline re-derives an
equivalent artifact, not the identical bytes.**

Run-to-run on this machine the recipe is stable where it can be: two full runs
produced identical `model.mil`, `metadata.json`, and `weights/weight.bin`, while
the compiled `coremldata.bin` files and the mlpackage `model.mlmodel` /
`Manifest.json` changed — so the residual instability is the CoreML compiler and
the package UUID, not the conversion.

This is why `tests/granite/model_io.rs` pins the PUBLISHED bundle's hashes: a
locally re-derived bundle will not satisfy that gate, and is not meant to.

## Replay

Run from the repository root, on a machine with `python3.11` and the Xcode command
line tools (`xcrun coremlcompiler`, used by step 2). Every binary below is invoked
by its explicit path — the venv is never "activated", so nothing depends on shell
state, and a stray global `hf` or `python` on `PATH` cannot be picked up silently.

```sh
export GRANITE_CONV=/path/to/scratch          # holds .venv + src-model
export GRANITE_GOLDENS="$PWD/crates/coremlit/tests/granite/fixtures/goldens"
export GRANITE_MODELS_OUT="$PWD/Models/embedkit-granite"
export GRANITE_MODEL_CARD=/path/to/model-card/README.md   # staged, not generated
python3.11 -m venv "$GRANITE_CONV/.venv"
"$GRANITE_CONV/.venv/bin/pip" install torch==2.6.0 transformers==5.14.0 \
  sentence-transformers==5.6.0 coremltools==9.0 numpy==1.26.4 huggingface_hub
"$GRANITE_CONV/.venv/bin/hf" download ibm-granite/granite-embedding-97m-multilingual-r2 \
  --revision 835ad14087e140460703cf0fae09f97d469d65c2 --local-dir "$GRANITE_CONV/src-model"
bash crates/coremlit/conversion/granite/run_granite.sh
```

`run_granite.sh` converts, compiles (recording the compilation), stages transactionally,
optionally regenerates the goldens, verifies, and publishes the manifest and
checksums — in that order. `GRANITE_MODEL_CARD` may be
omitted only when the published card is already staged under
`$GRANITE_MODELS_OUT/granite-97m-multilingual-r2/README.md`; its SHA-256 is pinned
and checked either way, so a stale or wrong card fails rather than being
checksummed in as though it were the published one.

It does **not** rewrite the committed goldens unless you pass
`GRANITE_REGEN_GOLDENS=1`. When you do, regeneration runs BEFORE verification, not
after: `verify_granite.py` scores against `$GRANITE_GOLDENS/corpus.json` and
records its digest, so regenerating afterwards would leave a manifest whose
numbers are labelled "vs the committed goldens" while describing comparisons
against the pair that was just replaced. Regeneration also requires a completed
conversion in the same run, because it publishes that conversion's crosscheck
rather than computing one — and requires that conversion to have MEASURED the
corpus being written, so a fixture edit means re-running the conversion, not just
the goldens step.

The committed goldens are themselves output of this recipe. They were regenerated
once it could emit the evidence the Rust gate needs — `min_max_abs_component_delta`
and the `corpus_sha256` pair binding — which the previously committed pair, cut by
the since-lost original script, did not carry. Against that previous pair:
identical ids, texts and `token_ids`; embeddings at worst cosine
**0.999999999999376** with worst per-component |Δ| **2.0e-7**; and the crosscheck's
worst cosine moved from `0.999999977090618` to `0.9999999999996902` because this
recipe reduces the cosine in `f64` where the old magnitudes are what an `f32`
reduction produces (which is also why several old per-entry values sat a few ULP
above 1.0, and the new ones do not).

Re-running the generator again will not reproduce those bytes either: torch's fp32
reduction order varies with the BLAS path, so components move in the last digit or
two, and `_provenance` carries a fresh `generated_utc` and the observed toolchain.
Byte-identity is not a property this recipe claims of its goldens — the gates score
by cosine and by the recorded distinctness, not by equality.

## Scripts

| file | role |
|---|---|
| `scripts/_granite_common.py` | the shared discipline every stage calls: pins, env paths, SHA verify, `observed_toolchain`, `observed_compiler`, `assert_corpus_identity`, `assert_artifact_file_set`, `assert_no_stale_publication_temps`, `require_verify_evidence`, `require_compile_record`, `require_published_crosscheck`, `corpus_input_sha256`, the all-or-nothing `replace_file_atomic`, the NaN-poisoning `worst_update`, plus config/prompt/pooling asserts, the official window extraction, the `GraniteGraph` driver and the crosscheck |
| `scripts/_fixtures.py` | the committed 16-entry multilingual corpus (ids + raw texts) |
| `scripts/convert_granite.py` | mint the run id, invalidate prior records AND every staged `.mlmodelc` -> faithfulness assert (staged for the goldens step) -> trace -> `ct.convert` fp16 (shipped) + fp32 (reference) |
| `scripts/compile_granite.py` | compile every staged mlpackage -> mlmodelc and write the run-bound `compile.json` (input package digest, output bundle digest, compiler environment) |
| `scripts/stage_artifact.py` | promote the compiled bundles into the artifact root TRANSACTIONALLY: copy to scratch beside the root, validate each copy against its source by digest, then rename into place in three phases, with a rollback that stops at its first failure — attestations move aside before the bundles and go back last, so no point at which the PROCESS can stop leaves one describing bytes that are gone. Process-crash safe; NOT power-loss ordered (after a power cut, re-run the conversion) and not resumable: a killed run's scratch holds the only copy of what left the root, so the next run refuses while it is there. Assumes one run at a time — nothing locks the root |
| `scripts/verify_granite.py` | producer, compilation-record, golden-pair and corpus-identity preconditions, then the fail-closed I/O-contract + fp32-vs-goldens + per-unit fp16 matrix; writes `verify_metrics.json` bound to this build's digests |
| `scripts/write_manifest.py` | requires passing, build-bound evidence including the compilation record, stages the two members the conversion does not produce (the model card and the pinned `tokenizer.json`), asserts the artifact root holds exactly the published 10 paths, then publishes `MANIFEST.json` + `CHECKSUMS.sha256` (source pins, observed toolchain, measured verify numbers) in that order, atomically |
| `scripts/generate_goldens.py` | `corpus.json` (computed) + `driver_crosscheck.json` (published from the conversion run, not recomputed, and only when it measured these exact ordered inputs); invalidates the previous pair, then lands the corpus and the crosscheck in that order |
| `run_granite.sh` | the env-driven end-to-end driver |
