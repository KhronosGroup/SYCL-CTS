#!/usr/bin/env python3
# ************************************************************************
#
#   SYCL Conformance Test Suite
#
#
#   Copyright (c) 2018-2022 Codeplay Software LTD. All Rights Reserved.
#   Copyright (c) 2022 The Khronos Group Inc.
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
#
# ************************************************************************

import sys
import argparse
from string import Template
from itertools import permutations
sys.path.append('../common/')
from common_python_vec import (
    Data, swap_pairs, generate_value_list, append_fp_postfix, wrap_with_kernel,
    wrap_with_test_func, make_func_call, write_source_file, get_types, cast_to_bool)

TEST_NAME = 'SWIZZLE_ASSIGNMENT'

swizzle_xyzw_rgba_assignment_template = Template("""
        sycl::vec<${type}, ${size}> ${vecName}DimTestVec = sycl::vec<${type}, ${size}>(${testVecValues});
        sycl::vec<${type}, ${size}> swizzledVec = sycl::vec<${type}, ${size}>();
        swizzledVec.${indexes}() = ${vecName}DimTestVec;
        ${type} vals[] = {${orderedValues}};
        if (!check_vector_values<${type}, ${size}>(swizzledVec, vals)) {
          resAcc[0] = false;
        }
""")

swizzle_elem_assignment_template = Template("""
        {
          sycl::vec<${type}, ${size}> ${vecName}DimTestVec = sycl::vec<${type}, ${size}>(${testVecValues});
          sycl::vec<${type}, ${size}> swizzledVec = sycl::vec<${type}, ${size}>();
          swizzledVec.template swizzle<${indexes}>() = ${vecName}DimTestVec;
          ${type} vals[] = {${orderedValues}};
          if (!check_vector_values<${type}, ${size}>(swizzledVec, vals)) {
            resAcc[0] = false;
          }
        }
""")

index_positions_dict = {
    'x': 0,
    'y': 1,
    'z': 2,
    'w': 3,
    'r': 0,
    'g': 1,
    'b': 2,
    'a': 3,
    'sycl::elem::s0': 0,
    'sycl::elem::s1': 1,
    'sycl::elem::s2': 2,
    'sycl::elem::s3': 3,
    'sycl::elem::s4': 4,
    'sycl::elem::s5': 5,
    'sycl::elem::s6': 6,
    'sycl::elem::s7': 7,
    'sycl::elem::s8': 8,
    'sycl::elem::s9': 9,
    'sycl::elem::sA': 10,
    'sycl::elem::sB': 11,
    'sycl::elem::sC': 12,
    'sycl::elem::sD': 13,
    'sycl::elem::sE': 14,
    'sycl::elem::sF': 15
}


def gen_ordered_values(index_string, type_str):
    """
    Generates a list of values from 0 to 16
    """
    ordered_val_list = [None] * len(index_string)
    val = 0
    for index in index_string:
        ordered_val_list[index_positions_dict[index]] = '(' + str(val) + ' % 2 == 0)' if type_str == 'bool' else str(val)
        val += 1
    return ordered_val_list


def gen_xyzw_str(type_str, size):
    """Generates tests for each combination of xyzw up to the size given.
    For example: size=3 will generate all permutations of xyz"""
    xyzw_string = ''
    for length in range(size, size + 1):
        for index_subset, value_subset in zip(
                permutations(Data.swizzle_xyzw_list_dict[size][:size], length),
                permutations(Data.vals_list_dict[size][:size], length)):
            index_string = ''
            for index in index_subset:
                index_string += index
            val_list = append_fp_postfix(type_str,
                                         gen_ordered_values(index_string, type_str))
            val_string = ', '.join(val_list)
            test_string = swizzle_xyzw_rgba_assignment_template.substitute(
                type=type_str,
                size=size,
                testVecValues=generate_value_list(type_str, size),
                indexes=index_string,
                vecName=Data.vec_name_dict[size],
                orderedValues=val_string)
            xyzw_string += wrap_with_kernel(
                type_str,
                'XYZW_KERNEL_' + type_str + str(size) + index_string,
                'Swizzle assignment: vec<' + type_str + ', ' + str(size) + '>.'
                + index_string, test_string)
    return xyzw_string


def gen_rgba_str(type_str, size):
    """Generates tests for each combination of rgba up to the size given.
    For example: size=4 will generate all permutations of rgba"""
    rgba_string = ''
    for length in range(size, size + 1):
        for index_subset, value_subset in zip(
                permutations(Data.swizzle_rgba_list_dict[size][:size], length),
                permutations(Data.vals_list_dict[size][:size], length)):
            index_string = ''
            for index in index_subset:
                index_string += index
            val_list = append_fp_postfix(type_str,
                                         gen_ordered_values(index_string, type_str))
            val_string = ', '.join(val_list)
            test_string = swizzle_xyzw_rgba_assignment_template.substitute(
                type=type_str,
                size=size,
                testVecValues=generate_value_list(type_str, size),
                indexes=index_string,
                vecName=Data.vec_name_dict[size],
                orderedValues=val_string)
            rgba_string += wrap_with_kernel(
                type_str,
                'RGBA_KERNEL_' + type_str + str(size) + index_string,
                'Swizzle assignment test for vec<' + type_str + ', ' +
                str(size) + '>.' + index_string, test_string)
    return rgba_string


def gen_elem_str(type_str, size):
    """Generates in order, reverse order, and swapped pairs
    of elements up to the given size.
    For example: size 4 will generate element sequences
      0, 1, 2, 3
      1, 0, 3, 2
      2, 3, 1, 0
      3, 2, 1, 0"""
    index_list = Data.swizzle_elem_list_dict[size][:size]
    index_string = ', '.join(index_list)
    test_string = swizzle_elem_assignment_template.substitute(
        type=type_str,
        size=size,
        testVecValues=generate_value_list(type_str, size),
        indexes=index_string,
        vecName=Data.vec_name_dict[size],
        orderedValues=', '.join(
            append_fp_postfix(type_str, gen_ordered_values(index_list, type_str))))
    if size == 1:
        return wrap_with_kernel(
            test_string, 'ELEM_KERNEL_' + type_str + str(size) +
            ''.join(Data.swizzle_elem_list_dict[size][:size]).replace(
                'sycl::elem::', ''), 'Swizzle assignment test for vec<' +
            type_str + ', ' + str(size) + '> .swizzle<' +
            ', '.join(Data.swizzle_elem_list_dict[size][:size]) + '>',
            test_string)
    index_list = swap_pairs(Data.swizzle_elem_list_dict[size][:size])
    index_string = ', '.join(index_list)
    test_string += swizzle_elem_assignment_template.substitute(
        type=type_str,
        size=size,
        testVecValues=generate_value_list(type_str, size),
        indexes=', '.join(
            swap_pairs(Data.swizzle_elem_list_dict[size][:size])),
        vecName=Data.vec_name_dict[size],
        orderedValues=', '.join(
            append_fp_postfix(type_str, gen_ordered_values(index_list, type_str))))
    index_list = swap_pairs(Data.swizzle_elem_list_dict[size][::-1])
    index_string = ', '.join(index_list)
    test_string += swizzle_elem_assignment_template.substitute(
        type=type_str,
        size=size,
        testVecValues=generate_value_list(type_str, size),
        indexes=', '.join(swap_pairs(Data.swizzle_elem_list_dict[size][::-1])),
        vecName=Data.vec_name_dict[size],
        orderedValues=', '.join(
            append_fp_postfix(type_str, gen_ordered_values(index_list, type_str))))
    index_list = Data.swizzle_elem_list_dict[size][::-1]
    index_string = ', '.join(index_list)
    test_string += swizzle_elem_assignment_template.substitute(
        type=type_str,
        size=size,
        testVecValues=generate_value_list(type_str, size),
        indexes=', '.join(Data.swizzle_elem_list_dict[size][::-1]),
        vecName=Data.vec_name_dict[size],
        orderedValues=', '.join(
            append_fp_postfix(type_str, gen_ordered_values(index_list, type_str))))
    return wrap_with_kernel(
        test_string, 'KERNEL_SWIZZLE_ASSIGNMENT_ELEM_' + type_str + str(size) +
        ''.join(Data.swizzle_elem_list_dict[size][:size]).replace(
            'sycl::elem::', ''), 'Swizzle assignment test for vec<' +
        type_str + ', ' + str(size) + '> .swizzle<' +
        ', '.join(Data.swizzle_elem_list_dict[size][:size]) + '>', test_string)


def gen_test(type_str, size):
    """Generates all tests for xyzw, rgba, and elem up to the given size"""
    string = ''
    # Generate xyzw, rgba tests only for size 1, 2, 3, 4 vecs
    if size <= 4:
        string += gen_xyzw_str(type_str, size)
        if size == 4:
            string += gen_rgba_str(type_str, size)
    string += gen_elem_str(type_str, size)
    return wrap_with_test_func(TEST_NAME, type_str, string, str(size))


def make_tests(type_str, input_file, output_file):
    if type_str == 'bool':
        Data.vals_list_dict = cast_to_bool(Data.vals_list_dict)

    test_str = ''
    func_calls = ''
    for size in Data.standard_sizes:
        test_str += gen_test(type_str, size)
        func_calls += make_func_call(TEST_NAME, type_str, str(size))
    write_source_file(test_str, func_calls, TEST_NAME, input_file,
                      output_file, type_str)

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
        '-o',
        required=True,
        dest="output",
        metavar='<out file>',
        help='CTS test output')
    args = argparser.parse_args()

    make_tests(args.ty, args.template, args.output)


if __name__ == '__main__':
    main()
