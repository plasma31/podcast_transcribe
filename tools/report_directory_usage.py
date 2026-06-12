#!/usr/bin/env python3
"""
Count files and total size under a directory.

Usage:
  python tools/report_directory_usage.py /path/to/dir
  python tools/report_directory_usage.py /path/to/dir --follow-symlinks
"""

import os
import sys
import argparse
from pathlib import Path

def human_bytes(n: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]
    f = float(n)
    for u in units:
        if f < 1024.0 or u == units[-1]:
            return f"{f:.2f} {u}" if u != "B" else f"{int(f)} {u}"
        f /= 1024.0
    return f"{n} B"

def dir_stats(root: Path, follow_symlinks: bool = False):
    total_files = 0
    total_dirs = 0
    total_bytes = 0

    # For avoiding loops when following symlinks
    seen_inodes = set()

    def on_error(err):
        print(f"[WARN] {err}", file=sys.stderr)

    for dirpath, dirnames, filenames in os.walk(root, onerror=on_error, followlinks=follow_symlinks):
        total_dirs += 1

        # If following symlinks, prevent infinite recursion via inode tracking
        if follow_symlinks:
            try:
                st = os.stat(dirpath)
                key = (st.st_dev, st.st_ino)
                if key in seen_inodes:
                    # We've already visited this directory (symlink loop)
                    dirnames[:] = []  # stop descending
                    continue
                seen_inodes.add(key)
            except OSError as e:
                print(f"[WARN] Cannot stat dir {dirpath}: {e}", file=sys.stderr)
                dirnames[:] = []
                continue

        for name in filenames:
            fp = os.path.join(dirpath, name)
            try:
                st = os.stat(fp) if follow_symlinks else os.lstat(fp)
            except FileNotFoundError:
                # file disappeared between listing and stat
                continue
            except PermissionError:
                print(f"[WARN] Permission denied: {fp}", file=sys.stderr)
                continue
            except OSError as e:
                print(f"[WARN] Cannot access {fp}: {e}", file=sys.stderr)
                continue

            # Count only regular files (avoid sockets, fifos, etc.)
            if os.path.isfile(fp) or (follow_symlinks and os.path.isfile(os.path.realpath(fp))):
                total_files += 1
                total_bytes += st.st_size

    return total_files, total_dirs, total_bytes

def main():
    ap = argparse.ArgumentParser(description="Calculate total file count and size of a directory.")
    ap.add_argument("path", help="Directory path to scan")
    ap.add_argument("--follow-symlinks", action="store_true", help="Follow symlinks (avoid loops)")
    args = ap.parse_args()

    root = Path(args.path).expanduser()
    if not root.exists():
        print(f"ERROR: Path does not exist: {root}", file=sys.stderr)
        sys.exit(2)
    if not root.is_dir():
        print(f"ERROR: Not a directory: {root}", file=sys.stderr)
        sys.exit(2)

    files, dirs, nbytes = dir_stats(root, follow_symlinks=args.follow_symlinks)

    print(f"Path: {root}")
    print(f"Directories: {dirs}")
    print(f"Files: {files}")
    print(f"Total size: {nbytes} bytes ({human_bytes(nbytes)})")

if __name__ == "__main__":
    main()

