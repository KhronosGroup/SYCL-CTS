#!/usr/bin/env python3

# NOTE: This script requires Python >= 3.7 and CMake >= 3.15

import argparse
import asyncio
import atexit
import os
import re
import shutil
import subprocess
import sys
import tempfile

from enum import Enum
from timeit import default_timer as timer

enable_verbose_logging = False


class LogLevel(Enum):
    DEFAULT = 37
    VERBOSE = 34
    WARNING = 33
    ERROR = 31


def log(msg: str, level=LogLevel.DEFAULT):
    if (level == LogLevel.VERBOSE) and not enable_verbose_logging:
        return
    stream = sys.stdout if level == LogLevel.DEFAULT else sys.stderr
    if sys.stderr.isatty():
        print(f"\x1b[{level.value};21m{msg}\x1b[0m", file=stream)
    else:
        print(msg, file=stream)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="""This script attempts to compile all test category
        targets for a given SYCL implementation. It then generates
        configuration-time test category filters for all failing targets.""")

    parser.add_argument('sycl_implementation', metavar="SYCL-Implementation",
                        choices=['DPCPP', 'AdaptiveCpp'], type=str,
                        help="The SYCL implementation to use")
    parser.add_argument('--cmake-args', type=str,
                        help="Arguments to pass on to CMake during configuration")
    parser.add_argument('-j', type=int, default=0, dest='parallel_jobs',
                        help="Number of parallel jobs to use during build")
    parser.add_argument('--output', '-o', type=str,
                        help="Name of the output filter file")
    parser.add_argument('--verbose', action='store_true',
                        help="Enable verbose logging")

    return parser.parse_args()


def find_all_categories(cts_dir: str):
    tests_dir = os.path.realpath(os.path.join(cts_dir, "tests"))
    return sorted([
        f.name for f in os.scandir(tests_dir) if f.is_dir() and f.name != 'common'
    ])


def configure_cmake(cts_dir: str, build_dir: str, sycl_impl: str, cmake_args: str):
    cmake_call = f"cmake -S {cts_dir} -B {build_dir} -G Ninja"
    cmake_call += f" -DSYCL_IMPLEMENTATION={sycl_impl} {cmake_args or ''}"
    log(cmake_call, LogLevel.VERBOSE)
    p = subprocess.run(cmake_call, shell=True, capture_output=True)
    if p.returncode != 0:
        log("CMake configuration failed with output:", LogLevel.ERROR)
        log(p.stderr.decode(), LogLevel.ERROR)
        exit(1)


def query_cmake_targets(build_dir: str):
    p = subprocess.run(
        f"cmake --build {build_dir} --target help", shell=True, capture_output=True)
    if p.returncode != 0:
        print("Failed to obtain list of available targets. Received output:",
              LogLevel.ERROR)
        log(p.stderr.decode(), LogLevel.ERROR)
        exit(1)

    raw_targets = p.stdout.decode().splitlines()
    target_pattern = re.compile(r"^(test_.+?)(?<!_objects): phony$")
    cmake_targets = []
    for rt in raw_targets:
        m = target_pattern.match(rt)
        if m:
            cmake_targets.append(m.group(1))

    return cmake_targets


did_warn_memory_usage = False


def monitor_memory_usage():
    """
    Running out of memory can cause jobs to fail that would otherwise compile.
    We alert the user if system memory usage exceeds the 95% threshold.
    """
    global did_warn_memory_usage
    # FIXME: This is non-portable
    (total, used, _) = map(int, os.popen(
        'free -t').readlines()[-1].split()[1:])
    if used / total > .95 and not did_warn_memory_usage:
        # Print additional newline so message doesn't get swallowed
        # by build progress (carriage return).
        log(("Warning: System is running low on memory, "
             "which can cause false negative results. "
             "Consider reducing number of parallel jobs (-j).\n"),
            LogLevel.WARNING)
        did_warn_memory_usage = True


async def compile_all_async(build_dir: str, targets: list, parallel_jobs: int):
    # Keep going even if some targets fail
    ninja_args = f"-k 0 -j {parallel_jobs}"
    cmake_call = f"cmake --build {build_dir} --clean-first --target {' '.join(targets)} -- {ninja_args}"
    log(cmake_call, LogLevel.VERBOSE)

    proc = await asyncio.create_subprocess_shell(
        cmake_call,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE)

    # Forward only CMake progress outputs (unless --verbose is set)
    progress_pattern = re.compile(r"^\[\d+/\d+\].*$")
    print("")
    while True:
        monitor_memory_usage()
        buf = await proc.stdout.readline()
        if len(buf) == 0:
            break
        line = buf.decode()
        if enable_verbose_logging:
            log(line, level=LogLevel.VERBOSE)
        elif progress_pattern.match(line):
            if sys.stdout.isatty():
                sys.stdout.write("\033[A")
                sys.stdout.write("\033[K")
                print('\r', end='')
            print(line, end='')

    await proc.wait()
    return proc


def find_failing_categories(build_dir: str, targets: list):
    # Do a dry run to find targets that are out of date (= not built)
    ninja_args = '-n'
    p = subprocess.run(
        f"cmake --build {build_dir} --target {' '.join(targets)} -- {ninja_args}", shell=True, capture_output=True)
    link_target_pattern = re.compile(
        r"^\[\d+/\d+\] Linking.*?bin/test_(.*)$", flags=re.M)
    return link_target_pattern.findall(p.stdout.decode())


def main():
    args = parse_arguments()
    global enable_verbose_logging
    enable_verbose_logging = args.verbose

    cts_dir = os.path.realpath(os.path.join(sys.path[0], ".."))
    all_categories = find_all_categories(cts_dir)
    log(f"Found {len(all_categories)} test categories.")
    log(', '.join(all_categories), LogLevel.VERBOSE)

    build_dir = tempfile.mkdtemp(prefix='sycl_cts_')
    atexit.register(lambda: shutil.rmtree(build_dir))

    log("Configuring CMake...")
    configure_cmake(cts_dir, build_dir,
                    args.sycl_implementation, args.cmake_args)

    cmake_targets = query_cmake_targets(build_dir)

    missing_targets = []
    available_targets = []
    for c in all_categories:
        t = "test_" + c
        if t in cmake_targets:
            available_targets.append(t)
        else:
            missing_targets.append(t)

    if len(missing_targets) != 0:
        log(
            f"Warning: The following {len(missing_targets)} category targets do not exist:", LogLevel.WARNING)
        log(', '.join(missing_targets), LogLevel.WARNING)

    ts_before = timer()
    log("Attempting to compile all category targets. This may take a while...")
    asyncio.run(compile_all_async(
        build_dir, available_targets, args.parallel_jobs))
    ts_after = timer()
    log(f"Done after {ts_after - ts_before:.1f} seconds.")

    failing_categories = find_failing_categories(build_dir, available_targets)

    if len(failing_categories) > 0:
        log(f"{len(failing_categories)} out of {len(all_categories)} categories failed to compile.")
        log("The following categories did not compile successfully: " +
            ', '.join(sorted(failing_categories)))
    else:
        log("All categories compiled successfully!")

    if args.output is not None:
        log(f"Writing output to file {args.output}", LogLevel.VERBOSE)
        with open(args.output, "w") as output_file:
            print("\n".join(sorted(failing_categories)), file=output_file)


if __name__ == '__main__':
    main()
