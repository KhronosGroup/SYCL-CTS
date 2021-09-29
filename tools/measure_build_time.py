#!/usr/bin/env python3

"""
Utility script for measuring the build time of a translation unit.
Not intended for manual use.
To enable, specify SYCL_CTS_MEASURE_BUILD_TIMES=ON during CMake configuration.
"""

import os
import subprocess
import sys

from pathlib import Path
from timeit import default_timer as timer

args = sys.argv[1:]

# We assume arguments to end with '-o <object file> -c <source file>'
# FIXME: This may not work with MSVC
obj_file = os.path.basename(args[-3])
src_file = args[-1]

# Locate build root: The compiler may not always be launched directly from
# within the main build directory. We want to write all results into the
# same file within the build directory, so we have to locate it first.
# Walk parents until we find 'CMakeCache.txt'.
build_root = Path(os.getcwd())
for p in [build_root] + list(build_root.parents):
    if os.path.isfile(p / 'CMakeCache.txt'):
        build_root = p
        break

# Make source file path relative to build directory
src_file = os.path.relpath(src_file, build_root)

ts_before = timer()
subprocess.run(' '.join(args), shell=True)
ts_after = timer()
dt = ts_after - ts_before

with open(build_root / "build_times.log", "a") as output_file:
    print(f"{dt:.1f} {obj_file} ({src_file})",
          file=output_file)
