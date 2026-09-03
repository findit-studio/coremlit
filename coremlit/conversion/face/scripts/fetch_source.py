"""Download ``buffalo_l.zip``, verify it against its pin, and extract the two members this
recipe reads. Usage: ``python fetch_source.py``.

The pack is 275 MiB of which one file (166 MiB) is converted and one (16 MiB) is used to
CUT FIXTURES and never converted. The other three are extracted too, so the SHA-256 of
every member can be recorded — a pack-level record is what lets a later reader confirm
that the files this recipe does not touch are the ones they think they are, and it is the
same discipline that caught four of ``fal/AuraFace-v1``'s five files being InsightFace
artifacts under terms the repository's own tag contradicted (issue #115).

**InsightFace publishes no digest for this pack.** ``insightface/utils/storage.py`` builds
a CloudFront URL and unzips whatever arrives; there is no manifest, signature or hash in
that path. The pin here is therefore a witness to the bytes this conversion consumed, not
a check against an upstream claim — see the model card.
"""
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from _arcface_common import (PACK_BYTES, PACK_MEMBERS, PACK_NAME, PACK_SHA256, PACK_URL,
                             RECOGNITION_MEMBER, conv_dir, observed_toolchain, sha256_file,
                             source_dir)


def download(dest: Path):
    if dest.is_file() and sha256_file(dest) == PACK_SHA256:
        print(f"[ok] {dest.name} already present and matches the pin")
        return
    print(f"[..] GET {PACK_URL}")
    with urllib.request.urlopen(PACK_URL) as response, open(dest, "wb") as fh:
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)


def main():
    observed = observed_toolchain()
    dest = conv_dir() / PACK_NAME
    download(dest)

    size = dest.stat().st_size
    digest = sha256_file(dest)
    if size != PACK_BYTES or digest != PACK_SHA256:
        raise SystemExit(f"REFUSING {dest}: {size} bytes / sha256 {digest}; pinned "
                         f"{PACK_BYTES} bytes / {PACK_SHA256}")
    print(f"[ok] {PACK_NAME}: {size} bytes, sha256 {digest}")

    out = source_dir()
    with zipfile.ZipFile(dest) as zf:
        names = sorted(n for n in zf.namelist() if not n.endswith("/"))
        if names != sorted(PACK_MEMBERS):
            raise SystemExit(f"pack members changed: {names} != {sorted(PACK_MEMBERS)}")
        for name in names:
            target = out / Path(name).name
            target.write_bytes(zf.read(name))

    members = {}
    for name, want in sorted(PACK_MEMBERS.items()):
        got = sha256_file(out / name)
        if got != want:
            raise SystemExit(f"{name}: sha256 {got}, pinned {want}")
        members[name] = {"sha256": got, "bytes": (out / name).stat().st_size}
        mark = "  <- CONVERTED" if name == RECOGNITION_MEMBER else ""
        print(f"[ok] {name:16s} {members[name]['bytes']:>10d}  {got}{mark}")

    record = {
        "pack": {"url": PACK_URL, "name": PACK_NAME, "bytes": size, "sha256": digest,
                 "upstream_publishes_a_digest": False},
        "members": members,
        "converted": RECOGNITION_MEMBER,
        "toolchain": observed,
    }
    (conv_dir() / "source.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"[ok] wrote {conv_dir() / 'source.json'}")


if __name__ == "__main__":
    main()
