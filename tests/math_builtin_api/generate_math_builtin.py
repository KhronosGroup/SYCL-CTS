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

def create_tests(test_id, types, signatures, kind, template, file_name, check = False):
    expanded_signatures =  test_generator.expand_signatures(types, signatures)

    # Extensions should be placed on separate files.
    base_signatures = []
    half_signatures = []
    double_signatures = []
    for sig in expanded_signatures:
        if contains_base_type(sig, "double"):
            double_signatures.append(sig)
            continue
        if contains_base_type(sig, "sycl::half"):
            half_signatures.append(sig)
            continue
        base_signatures.append(sig)

    if base_signatures and kind == 'base':
        generated_base_test_cases = test_generator.generate_test_cases(test_id, types, base_signatures, check)
        write_cases_to_file(generated_base_test_cases, template, file_name)
    elif half_signatures and kind == 'half':
        generated_half_test_cases = test_generator.generate_test_cases(test_id + 300000, types, half_signatures, check)
        write_cases_to_file(generated_half_test_cases, template, file_name, "fp16")
    elif double_signatures and kind == 'double':
        generated_double_test_cases = test_generator.generate_test_cases(test_id + 600000, types, double_signatures, check)
        write_cases_to_file(generated_double_test_cases, template, file_name, "fp64")
    else:
        print("No %s overloads to generate for the test category" % kind)
        sys.exit(1)

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
        '-o',
        dest="output",
        required=True,
        metavar='<out file>',
        help='CTS test output')
    args = argparser.parse_args()

    use_marray = (args.marray == 'true')
    run = runner(use_marray)
    if not use_marray:
        print("WARNING: marray types are not used int the tests!")

    created_types = sycl_types.create_types()

    expanded_types =  test_generator.expand_types(run, created_types)

    verifyResults = True

    if args.test == 'integer':
        integer_signatures = sycl_functions.create_integer_signatures()
        create_tests(0, expanded_types, integer_signatures, args.variante, args.template, args.output, verifyResults)

    if args.test == 'common':
        common_signatures = sycl_functions.create_common_signatures()
        create_tests(1000000, expanded_types, common_signatures, args.variante, args.template, args.output, verifyResults)

    if args.test == 'geometric':
        geomteric_signatures = sycl_functions.create_geometric_signatures()
        create_tests(2000000, expanded_types, geomteric_signatures, args.variante, args.template, args.output, verifyResults)

    if args.test == 'relational':
        relational_signatures = sycl_functions.create_relational_signatures()
        create_tests(3000000, expanded_types, relational_signatures, args.variante, args.template, args.output, verifyResults)

    if args.test == 'float':
        float_signatures = sycl_functions.create_float_signatures()
        create_tests(4000000, expanded_types, float_signatures, args.variante, args.template, args.output, verifyResults)

    if args.test == 'native':
        native_signatures = sycl_functions.create_native_signatures()
        create_tests(5000000, expanded_types, native_signatures, args.variante, args.template, args.output, verifyResults)

    if args.test == 'half':
        half_signatures = sycl_functions.create_half_signatures()
        create_tests(6000000, expanded_types, half_signatures, args.variante, args.template, args.output, verifyResults)

if __name__ == "__main__":
    main()
