#!/usr/bin/env python3
"""pack the participant handout: chat.py + model/ only.

  python tools/pack_dist.py
  python tools/pack_dist.py --out spooky-vault-dist.zip
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BANNED_SUBSTR = (
    "spooky_vault_bones",
    "flag{spooky",
    "flag{violet",
    "secret.txt",
)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--chat", type=Path, default=ROOT / "chat.py")
    p.add_argument("--model", type=Path, default=ROOT / "model")
    p.add_argument("--out", type=Path, default=ROOT / "spooky-vault-dist.zip")
    args = p.parse_args()

    if not args.chat.is_file():
        raise SystemExit(f"missing {args.chat}")
    if not args.model.is_dir():
        raise SystemExit(f"missing model dir {args.model}")
    weights = args.model / "model.safetensors"
    if not weights.is_file():
        raise SystemExit(f"missing {weights}")

    # soft check: do not zip obvious flag strings sitting in small text files.
    for path in args.model.rglob("*"):
        if not path.is_file() or path.suffix in {".safetensors", ".bin", ".pt"}:
            continue
        if path.stat().st_size > 2_000_000:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore").lower()
        except OSError:
            continue
        for bad in BANNED_SUBSTR:
            if bad.lower() in text:
                raise SystemExit(f"refusing to pack {path}: contains {bad!r}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(args.chat, arcname="chat.py")
        for path in sorted(args.model.rglob("*")):
            if not path.is_file() or path.name == ".gitkeep":
                continue
            zf.write(path, arcname=str(Path("model") / path.relative_to(args.model)))
    print(f"wrote {args.out} ({args.out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
