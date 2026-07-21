#!/usr/bin/env python3
# ************************************************************************
#
#   SPDX-FileCopyrightText: 2018-2022 Codeplay Software LTD.
#   SPDX-FileCopyrightText: 2022 The Khronos Group Inc.
#   SPDX-License-Identifier: Apache-2.0
#
#   SYCL Conformance Test Suite
#
# ************************************************************************

import sys
import argparse
from string import Template
sys.path.append('../common/')
from common_python_vec import (get_types, make_swizzles_tests)

def main():
    argparser = argparse.ArgumentParser(
        description='Generates vector swizzles opencl test')
    argparser.add_argument(
        'template',
        metavar='<code template path>',
        help='Path to code template')
    argparser.add_argument(
        '-type',
        dest='ty',
        required=True,
        choices=get_types(),
        help='Type to generate the test for')
    argparser.add_argument(
        '-num_batches',
        dest='num_batches',
        required=True,
        type=int,
        help='Number of batches to split the test cases into')
    argparser.add_argument(
        '-batch_index',
        dest='batch_index',
        required=True,
        type=int,
        help='Batch index of the test batch to write to the output file.') 
    argparser.add_argument(
        '-o',
        required=True,
        dest="output",
        metavar='<out file>',
        help='CTS test output')
    args = argparser.parse_args()

    make_swizzles_tests(args.ty, args.template, args.output, args.num_batches, args.batch_index - 1)


if __name__ == '__main__':
    main()
