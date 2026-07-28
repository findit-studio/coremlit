"""Write the committed granite goldens into ``$GRANITE_GOLDENS``.

  * ``corpus.json`` — the ORACLE every Rust granite gate reads: per entry the raw
    ``text``, the exact ``token_ids`` (specials included, truncated at 512),
    ``n_tokens``, and the canonical transformers-fp32 UNIT-NORMALIZED 384-d
    ``embedding``. The CoreML graph emits the pre-L2-norm CLS vector; the
    normalization the goldens carry is the one the Rust caller applies.
  * ``driver_crosscheck.json`` — the CONVERSION-TIME pre-trace measurement,
    read from staging and published here verbatim (plus the corpus binding
    below). This step does NOT recompute it: the committed fixture has to be the
    measurement conversion actually gated on, and a second computation made later
    would be a different measurement of a different graph instance, however close
    its numbers. Consumed hermetically by ``tests/granite/driver_crosscheck.rs``.
    Conversion must therefore run before goldens may be regenerated.

Two bindings sit on that record and they are NOT the same claim.
``corpus_input_sha256`` is computed during the measurement, over the ordered
``(id, text)`` it consumed, and is REQUIRED here to equal the ordered inputs
these goldens are written from — that is what makes "this crosscheck measured
this corpus" checkable, and what stops a goldens-only re-run after a fixture edit
from relabelling an older measurement. ``corpus_sha256`` is stamped on afterwards
over the serialized ``corpus.json`` bytes, which do not exist while the
measurement runs; it binds the two files as a published PAIR and nothing more.

Embeddings are serialized at 7 DECIMAL PLACES — the committed goldens' format.
Unit-norm components here are ~0.05, where an fp32 ULP is ~4e-9, so this is a
deliberate quantization to +-5e-8, coarser than fp32 but orders of magnitude
below every cosine floor the gates score against, and it keeps the fixture
human-diffable.

Regeneration is REPRODUCIBLE TO THE COSINE FLOORS, not byte-identical: torch's
fp32 reduction order varies with the BLAS path, so components move beyond that
quantization anyway (measured worst |delta| 2.0e-7 against the committed file,
at worst cosine 0.999999999999376). Byte-identity is not a property this recipe
claims, and the Rust gates score by cosine, not by equality.
"""
import datetime
import hashlib
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from _granite_common import (
    DRIVER_FLOOR,
    EMBED_DIM,
    GOLDEN_CORPUS,
    GOLDEN_CROSSCHECK,
    MODEL_ID,
    REV,
    SEQ_LEN,
    FP32_REFERENCE,
    RUN_ID_KEY,
    SHIPPED_PACKAGE,
    assert_corpus_identity,
    assert_prompt_free,
    corpus_input_sha256,
    digest_tree,
    discard_file,
    load_sentence_transformer,
    read_producer_record,
    read_staged_crosscheck,
    observed_toolchain,
    padded_inputs,
    replace_file_atomic,
    stage_dir,
)
from _fixtures import CORPUS, goldens_dir

DECIMALS = 7

PROVENANCE = {
    "artifact": f"embedkit / {MODEL_ID} -> CoreML (T1 goldens)",
    "purpose": "committed-oracle fixtures: tokenizer identity + transformers-fp32 embedding parity",
    "canonical_oracle": "sentence-transformers granite pipeline in float32 (ModernBert -> CLS "
                        "pool -> L2); lossless fp32 upcast of the bf16 checkpoint",
    "cross_check": "cosine(canonical vs the fixed-512 static-mask GraniteGraph driver, the CoreML "
                   "trace source); see driver_crosscheck.json",
    "output": "PRE-L2-norm 384-d CLS in the CoreML graph; goldens are UNIT-NORMALIZED (L2 in Rust)",
    "embedding_dim": EMBED_DIM,
    "prompt_free": True,
    "prompt_note": "granite r2 retrieval is PROMPT-FREE: config_sentence_transformers.json prompts "
                   "are empty; the corpus is RAW strings, no task prefixes",
    "pooling": "cls (hidden[:,0])",
    "tokenizer": "granite tokenizer.json (o200k-style BPE, vocab 180000); token_ids truncated at 512",
    "export_seq_len": SEQ_LEN,
    "hf_model": MODEL_ID,
    "hf_revision": REV,
    # Placeholder only, so the emitted key order stays stable; main() overwrites
    # it with observed_toolchain(). A golden must never record a version that was
    # not the one that computed it.
    "toolchain": None,
}


def serialize(obj):
    """Match the committed goldens' serialization: indent 1, raw UTF-8 (the
    corpus is multilingual — escaping it would make the fixtures unreadable), no
    trailing newline."""
    return json.dumps(obj, indent=1, ensure_ascii=False)


def main():
    out = goldens_dir()
    os.makedirs(out, exist_ok=True)
    # Bind to the conversion run whose crosscheck this step publishes.
    stage = stage_dir()
    run_id = read_producer_record(stage, {
        name: digest_tree(os.path.join(stage, name))
        for name in (SHIPPED_PACKAGE, FP32_REFERENCE)
    })[RUN_ID_KEY]
    st = load_sentence_transformer()
    assert_prompt_free(st)
    tokenizer = st.tokenizer
    pad_id = st[0].auto_model.config.pad_token_id

    entries = []
    for entry in CORPUS:
        _input_ids, _attention_mask, ids = padded_inputs(tokenizer, entry["text"], pad_id)
        with torch.no_grad():
            emb = st.encode([entry["text"]], convert_to_numpy=True,
                            normalize_embeddings=True, batch_size=1)[0]
        emb = np.asarray(emb, np.float32).ravel()
        assert emb.shape == (EMBED_DIM,), emb.shape
        assert np.all(np.isfinite(emb)), f"non-finite golden for {entry['id']}"
        entries.append({
            "id": entry["id"],
            "text": entry["text"],
            "token_ids": [int(i) for i in ids],
            "n_tokens": len(ids),
            "embedding": [round(float(v), DECIMALS) for v in emb],
        })
        print(f"  {entry['id']:<10} {len(ids):>3} tokens")

    # The same precondition verify_granite.py applies to the committed corpus,
    # applied here to what is about to BECOME it.
    assert_corpus_identity(entries, "the entries about to be written")

    provenance = dict(PROVENANCE)
    provenance["toolchain"] = observed_toolchain()
    provenance["generated_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    provenance["n_entries"] = len(entries)
    # Serialized but NOT yet on disk: the crosscheck below can still fail, and a
    # new corpus sitting beside the previous run's crosscheck would be a pair the
    # Rust gate joins by id and happily accepts.
    corpus_text = serialize({"_provenance": provenance, "entries": entries})

    crosscheck = read_staged_crosscheck(stage, run_id)

    # The crosscheck must have MEASURED the inputs this corpus is being written
    # from — not merely have been produced by the same run.
    #
    # Matching run ids are not enough: the packages are unchanged by a fixture
    # edit, so editing a `text` in _fixtures.py and re-running ONLY this step
    # passes every id, shape and run-identity check while republishing a
    # measurement taken over the PREVIOUS texts, relabelled with the new
    # corpus's digest. The ordered-input digest is recorded during the
    # measurement itself, so that re-run reds here instead.
    measured_inputs = crosscheck.get("corpus_input_sha256")
    written_inputs = corpus_input_sha256([(e["id"], e["text"]) for e in entries])
    if not measured_inputs:
        raise SystemExit(
            f"STAGED CROSSCHECK CARRIES NO `corpus_input_sha256`: it predates the binding that "
            f"ties the measurement to its inputs, so publishing it here would assert nothing "
            f"about which texts were measured. Re-run convert_granite.py."
        )
    if measured_inputs != written_inputs:
        raise SystemExit(
            f"CROSSCHECK MEASURED DIFFERENT INPUTS than these goldens are being written from:\n"
            f"  measured  {measured_inputs}\n  writing   {written_inputs}\n"
            f"  The conversion's faithfulness measurement consumed a different ordered (id, text) "
            f"corpus than the one about to be committed. Re-run convert_granite.py against the "
            f"current fixtures — a goldens-only re-run would relabel that older measurement."
        )

    worst = crosscheck["worst_cosine_canonical_vs_driver"]
    print(f"  driver crosscheck: {crosscheck['verdict']}, worst {worst:.16f}, "
          f"min component delta {crosscheck['min_max_abs_component_delta']:.3e}")
    # Fail-closed: this script can run standalone, so it re-gates the floor
    # rather than trusting that convert_granite.py ran first. A diverged
    # measurement must NOT be written out as a golden.
    if not (worst >= DRIVER_FLOOR):
        raise SystemExit(f"driver UNFAITHFUL: worst {worst:.16f} < {DRIVER_FLOOR}; "
                         "refusing to write the goldens")
    # Cross-bind: the crosscheck names the exact corpus bytes it is published
    # alongside, so the two files can be checked as a PAIR rather than joined by id.
    crosscheck["corpus_sha256"] = hashlib.sha256(corpus_text.encode("utf-8")).hexdigest()
    crosscheck_text = serialize(crosscheck)

    # Both payloads are complete before either is put in place, and the previous
    # pair is invalidated before either half is replaced. Without that, a
    # regeneration killed between the two renames left the NEW corpus beside the
    # PREVIOUS crosscheck — two files that each look fine and together describe
    # nothing, which every later stage used to attest to happily.
    #
    # The corpus lands first and the crosscheck last, because the crosscheck is
    # what names the corpus (`corpus_sha256`): its presence can then only mean
    # the corpus it names is already there. Reachable states are neither file,
    # the corpus alone — every gate fails loudly on the missing crosscheck — or
    # both from this run. A plausible mismatched pair is not among them.
    discard_file(os.path.join(out, GOLDEN_CORPUS))
    discard_file(os.path.join(out, GOLDEN_CROSSCHECK))
    replace_file_atomic(os.path.join(out, GOLDEN_CORPUS), corpus_text)
    replace_file_atomic(os.path.join(out, GOLDEN_CROSSCHECK), crosscheck_text)
    print(f"  wrote {GOLDEN_CORPUS} + {GOLDEN_CROSSCHECK} to {out}")
    print("goldens DONE")


if __name__ == "__main__":
    main()
