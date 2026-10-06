"""
verify_inputs.py -- check input files against a SHA-256 checksum list.

Purpose : first step of run_all.sh. Confirms that the deposited input files
          are the ones the analysis was verified with.
Usage   : python verify_inputs.py FILE [FILE ...] [--checksums LIST]
Inputs  : FILE            input files to check (identified by file name)
          --checksums     checksum list in the `sha256sum` format
                          ("<hash>  <path>", path relative to the list). If
                          omitted, `checksums.sha256` in the parent folder of
                          the first FILE is used. If neither exists the check
                          is skipped (exit 0, message printed).
Outputs : one line per file (ok / MISMATCH / not listed) on the console; exit
          status 1 if any file is not ok.
Environment : Python 3.11.14; standard library only.
"""

import argparse
import hashlib
import sys
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("files", nargs="+")
    parser.add_argument("--checksums", default=None)
    args = parser.parse_args()

    files = [Path(f) for f in args.files]
    listing = (Path(args.checksums) if args.checksums
               else files[0].resolve().parent.parent / "checksums.sha256")
    if not listing.is_file():
        print(f"skipped: no checksum list found ({listing.name})")
        return 0

    expected = {}
    for line in listing.read_text().splitlines():
        if line.strip():
            digest, name = line.split(None, 1)
            expected[Path(name.strip().lstrip("*")).name] = digest

    failed = False
    for f in files:
        if f.name not in expected:
            print(f"not listed: {f.name}")
            failed = True
        elif not f.is_file():
            print(f"missing: {f.name}")
            failed = True
        elif sha256(f) == expected[f.name]:
            print(f"ok: {f.name}")
        else:
            print(f"MISMATCH: {f.name}")
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
