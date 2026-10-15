#!/usr/bin/env bash
# Compile src/ business logic with Cython and replace the .py sources
# in place with the compiled .so extension modules. Run from repo root
# with build deps (cython, setuptools) already installed.
set -euo pipefail

rm -rf build
python setup.py build_ext

LIBDIR=$(find build -maxdepth 1 -type d -name 'lib.*')

find "$LIBDIR" -name '*.so' | while read -r so_file; do
    rel="${so_file#"$LIBDIR"/}"
    mkdir -p "$(dirname "$rel")"
    cp "$so_file" "$rel"

    py="${rel%%.cpython-*}.py"
    rm -f "$py"
done

rm -rf build
