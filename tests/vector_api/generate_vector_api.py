#!/usr/bin/env python3
# ************************************************************************
#
#   SYCL Conformance Test Suite
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
sys.path.append('../common/')
from common_python_vec import (Data, ReverseData, append_fp_postfix, wrap_with_kernel,
                               wrap_with_test_func, make_func_call,
                               write_source_file)

TEST_NAME = 'API'

vector_api_template = Template("""
        auto inputVec = cl::sycl::vec<${type}, ${size}>(${vals});
        ${type} vals[] = {${vals}};
        ${type} reversed_vals[] = {${reversed_vals}};
        if (!check_vector_member_functions<${type}, ${convertType}, ${asType}>(inputVec, vals)) {
          resAcc[0] = false;
        }
        cl::sycl::vec<${type}, ${size}> swizzledVec {inputVec.template swizzle<${swizIndexes}>()};
        if (!check_vector_values<${type}, ${size}>(swizzledVec, reversed_vals)) {
          resAcc[0] = false;
        }
        if (std::alignment_of<cl::sycl::vec<${type}, ${size}>>::value !=
            sizeof(${type}) * (${size} == 3 ? 4 : ${size})) {
          resAcc[0] = false;
        }
""")

lo_hi_odd_even_template = Template("""
        if (!check_lo_hi_odd_even<${type}>(inputVec, vals)) {
          resAcc[0] = false;
        }
""")


def make_host_kernel_name(type_str, size):
    return 'KERNEL_API_HOST_T' + type_str.replace('cl::sycl::', '').replace(
        ' ', '') + str(size)


def make_interop_kernel_name(type_str, size):
    return 'KERNEL_APIINTEROP_T_' + type_str.replace('cl::sycl::', '').replace(
        ' ', '') + str(size)


def gen_host_checks(type_str, reverse_type_str, size):
    """Uses the above string templates to generate tests for each vec api function except load and store.
    Load and store are handled separately.
    lo() hi() odd() and even() are handled with a separate function and template to other api functions
    as they can only be performed on vectors of size 2 or greater."""
    vals_list = append_fp_postfix(type_str, Data.vals_list_dict[size])
    reverse_vals_list = vals_list[::-1]
    kernel_name = make_host_kernel_name(type_str, size)
    test_string = vector_api_template.substitute(
        type=type_str,
        size=size,
        vals=', '.join(vals_list),
        reversed_vals=', '.join(reverse_vals_list),
        convertType=reverse_type_str,
        asType=reverse_type_str,
        swizIndexes=', '.join(Data.swizzle_elem_list_dict[size][::-1]))
    if size != 1:
        test_string += lo_hi_odd_even_template.substitute(type=type_str)
    string = wrap_with_kernel(
        type_str, kernel_name,
        'API test for cl::sycl::vec<' + type_str + ', ' + str(size) + '>',
        test_string)
    return wrap_with_test_func(TEST_NAME, type_str, string, str(size))


def gen_interop_checks(type_str, reverse_type_str, size):
    vals_list = append_fp_postfix(type_str, Data.vals_list_dict[size])
    reverse_vals_list = vals_list[::-1]
    kernel_name = make_interop_kernel_name(type_str, size)
    test_string = vector_api_template.substitute(
        type=type_str,
        size=size,
        vals=', '.join(vals_list),
        reversed_vals=', '.join(reverse_vals_list),
        convertType=reverse_type_str,
        asType=reverse_type_str,
        swizIndexes=', '.join(Data.swizzle_elem_list_dict[size][::-1]))
    if size != 1:
        test_string += lo_hi_odd_even_template.substitute(type=type_str)
    string = wrap_with_kernel(
        type_str, kernel_name,
        'API test for cl::sycl::vec<' + type_str + ', ' + str(size) + '>',
        test_string)
    return wrap_with_test_func(TEST_NAME, type_str, string, str(size))

def get_reverse_type(type_str):
    if type_str == 'char':
        return 'char'
    if type_str in ReverseData.rev_standard_type_dict:
        type_dict =  Data.standard_type_dict
        rev_type_dict = ReverseData.rev_standard_type_dict
    else:
        type_dict =  Data.opencl_type_dict
        rev_type_dict = ReverseData.rev_opencl_type_dict
    (sign, base_type) = rev_type_dict[type_str]
    if (not sign, base_type) in type_dict:
        reverse_type_str = type_dict[(not sign, base_type)]
    else:
        reverse_type_str = type_str
    return reverse_type_str

def make_tests(type_str, input_file, output_file):
    reverse_type_str = get_reverse_type(type_str)
    is_opencl_type = type_str in ReverseData.rev_opencl_type_dict
    api_checks = ''
    func_calls = ''
    for size in Data.standard_sizes:
        if is_opencl_type:
            api_checks += gen_interop_checks(type_str, reverse_type_str, size)
        else:
            api_checks += gen_host_checks(type_str, reverse_type_str, size)
        func_calls += make_func_call(TEST_NAME, type_str, str(size))
    write_source_file(api_checks, func_calls, TEST_NAME, input_file,
                      output_file, type_str)

def get_types():
    types = list()
    types.append('char')
    for base_type in Data.standard_types:
        for sign in Data.signs:
            if (base_type == 'float' or base_type == 'double'
                or base_type == 'cl::sycl::half') and sign is False:
                continue
            types.append(Data.standard_type_dict[(sign, base_type)])

    for base_type in Data.opencl_types:
        for sign in Data.signs:
            if (base_type == 'cl::sycl::cl_float'
                    or base_type == 'cl::sycl::cl_double'
                    or base_type == 'cl::sycl::cl_half') and sign is False:
                continue
            types.append(Data.opencl_type_dict[(sign, base_type)])
    return types

def main():
    argparser = argparse.ArgumentParser(
        description='Generates vector swizzles opencl test'
    )
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
