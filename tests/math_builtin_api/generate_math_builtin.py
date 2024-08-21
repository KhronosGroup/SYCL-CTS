#!/usr/bin/env python3
################################################################################
##
##  SYCL 2020 Conformance Test Suite
##
#
#   Copyright (c) 2017-2022 Codeplay Software LTD. All Rights Reserved.
#   Copyright (c) 2022-2023 The Khronos Group Inc.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
##
################################################################################

import os
import sys
import argparse
import math
from modules import sycl_types
from modules import sycl_functions
from modules import test_generator

# Used to include types that are supported by implementation
class runner:
    def __init__(self, marray):
        self.base_types = ["bool", "float", "double", "sycl::half", "char",
            "signed char", "unsigned char", "short", "unsigned short", "int", "unsigned", "long", "unsigned long", "long long", "unsigned long long",
            "int8_t", "int16_t", "int32_t", "int64_t", "uint8_t", "uint16_t", "uint32_t", "uint64_t"]
        self.var_types = ["scalar","vector"]
        if marray:
            self.var_types.append("marray")
        self.dimensions = [1,2,3,4,8,16]
        if marray:
            self.dimensions.extend([5,17])

def contains_base_type(sig, base_type):
    data = sig.arg_types[:]
    data.append(sig.ret_type)
    for dt in data:
        if dt.base_type == base_type:
            return True
    return False

def write_cases_to_file(generated_test_cases, inputFile, outputFile, extension=None):
    # Determine generator directory
    generatorDirectory = os.path.dirname(os.path.realpath(__file__))

    # Load the template from the input file
    with open(inputFile, 'r') as input:
        source = input.read()

    # Execute the tests if the extensions are supported by target device.
    if extension:
        checkPoint = "\n\nif(once_per_unit::get_queue().get_device().has(sycl::aspect::" + extension + ")){\n"
        generated_test_cases = checkPoint + generated_test_cases + "\n\n}"

    newSource = source.replace("$TEST_CASES", generated_test_cases)
    newSource = newSource.replace("$math_builtins", os.path.splitext(os.path.basename(outputFile))[0])
    if extension is None:
        extension = ""
    else:
        extension = "#ifdef __SYCL_DEVICE_ONLY__\n#ifdef $s\n#pragma OPENCL EXTENSION %s : enable\n#endif\n#endif" % extension
    newSource = newSource.replace("$pragma_ext", extension)

    # Write the source to the output file
    with open(outputFile, 'w+') as output:
        output.write(newSource)

def create_tests(test_id, types, signatures, template, file_name, extension, check = False):
    generated_test_cases = test_generator.generate_test_cases(test_id, types, signatures, check)
    write_cases_to_file(generated_test_cases, template, file_name, extension)

def main():
    argparser = argparse.ArgumentParser(
        description='Generates SYCL 2020 mathematical functions test'
    )
    argparser.add_argument(
        'template',
        metavar='<code template path>',
        help='Path to code template')
    argparser.add_argument(
        '-test',
        required=True,
        choices=['integer', 'common', 'geometric', 'relational', 'float', 'native', 'half'],
        help='')
    argparser.add_argument(
        '-variante',
        choices=['base', 'half', 'double'],
        default='base',
        help='Generate the half or double overload to a given test category')
    argparser.add_argument(
        '-marray',
        choices=['true', 'false'],
        default='false',
        help='Generate tests with marray function arguments')
    argparser.add_argument(
        '-print-output-files',
        action='store_true',
        help='Print all generated output files instead of generating the test files')
    argparser.add_argument(
        '-fragment-size',
        type=int,
        required=False,
        help='Generate test files with tests for N math function signatures')
    argparser.add_argument(
        '-output-prefix',
        required=True,
        help='CTS test output file name prefix')
    argparser.add_argument(
        '-ext',
        required=True,
        help='CTS test output file extension')
    args = argparser.parse_args()

    use_marray = (args.marray == 'true')
    run = runner(use_marray)
    if not use_marray:
        print("WARNING: marray types are not used in the tests!")

    created_types = sycl_types.create_types()

    expanded_types =  test_generator.expand_types(run, created_types)

    verifyResults = True

    test_signatures = []
    test_id_offset = 0
    if args.test == 'integer':
        test_signatures = sycl_functions.create_integer_signatures()
        test_id_offset = 0
    elif args.test == 'common':
        test_signatures = sycl_functions.create_common_signatures()
        test_id_offset = 1000000
    elif args.test == 'geometric':
        test_signatures = sycl_functions.create_geometric_signatures()
        test_id_offset = 2000000
    elif args.test == 'relational':
        test_signatures = sycl_functions.create_relational_signatures()
        test_id_offset = 3000000
    elif args.test == 'float':
        test_signatures = sycl_functions.create_float_signatures()
        test_id_offset = 4000000
    elif args.test == 'native':
        test_signatures = sycl_functions.create_native_signatures()
        test_id_offset = 5000000
    elif args.test == 'half':
        test_signatures = sycl_functions.create_half_signatures()
        test_id_offset = 6000000

    test_signatures = test_generator.expand_signatures(expanded_types, test_signatures, silent=args.print_output_files)

    # Extensions should be placed on separate files.
    base_signatures = []
    half_signatures = []
    double_signatures = []
    for sig in test_signatures:
        if contains_base_type(sig, "double"):
            double_signatures.append(sig)
            continue
        if contains_base_type(sig, "sycl::half"):
            half_signatures.append(sig)
            continue
        base_signatures.append(sig)

    # Update signatures based on the variants
    extension = None
    if base_signatures and args.variante == 'base':
        test_signatures = base_signatures
    elif half_signatures and args.variante == 'half':
        test_signatures = half_signatures
        test_id_offset = test_id_offset + 300000
        extension = "fp16"
    elif double_signatures and args.variante == 'double':
        test_signatures = double_signatures
        test_id_offset = test_id_offset + 600000
        extension = "fp64"
    else:
        print("No %s overloads to generate for the test category" % args.variante)
        sys.exit(1)

    output_files = []
    if len(test_signatures) != 0:
        if not args.fragment_size:
            output_files = [args.output_prefix + "." + args.ext]
        else:
            output_files = [args.output_prefix + "_" + str(i) + "." + args.ext for i in range(0, math.ceil(len(test_signatures) / args.fragment_size))]

    if args.print_output_files:
        print(';'.join(output_files))
        # If output files are being printed we will not generate files.
        return

    if not args.fragment_size:
        create_tests(test_id_offset, expanded_types, test_signatures, args.variante, args.template, output_files[0], extension, verifyResults)
    else:
        for i in range(0, math.ceil(len(test_signatures) / args.fragment_size)):
            fragment_start = i * args.fragment_size
            fragment_end = fragment_start + args.fragment_size
            current_offset = test_id_offset + fragment_start * 100
            create_tests(current_offset, expanded_types, test_signatures[fragment_start:fragment_end], args.template, output_files[i], extension, verifyResults)

if __name__ == "__main__":
    main()
