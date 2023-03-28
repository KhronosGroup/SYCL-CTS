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
sys.path.append('../common/')
from common_python_vec import (Data, ReverseData, append_fp_postfix, wrap_with_kernel,
                               wrap_with_test_func, make_func_call,
                               write_source_file, get_types, cast_to_bool)

TEST_NAME = 'API'

vector_element_type_template = Template("""
    CHECK(std::is_same_v<typename sycl::vec<${type}, ${size}>::element_type, ${type}>);
// FIXME: re-enable when element_type for swizzle_vec is implemented
// link to issue: https://github.com/intel/llvm/issues/8879
#if !SYCL_CTS_COMPILING_WITH_DPCPP
    sycl::vec<${type}, ${size}> vec;
    CHECK(std::is_same_v<typename decltype(
        vec.template swizzle<${swizIndexes}>())::element_type, ${type}>);
#endif
""")

vector_api_template = Template("""
        auto inputVec = sycl::vec<${type}, ${size}>(${vals});
        ${type} reversed_vals[] = {${reversed_vals}};
        if (!check_vector_size_byte_size<${type}, ${size}>(inputVec)) {
          resAcc[0] = false;
        }
        sycl::vec<${type}, ${size}> swizzledVec {inputVec.template swizzle<${swizIndexes}>()};
        if (!check_vector_values<${type}, ${size}>(swizzledVec, reversed_vals)) {
          resAcc[0] = false;
        }
        if (std::alignment_of<sycl::vec<${type}, ${size}>>::value !=
            sizeof(${type}) * (${size} == 3 ? 4 : ${size})) {
          resAcc[0] = false;
        }
// FIXME: re-enable type conversion for bool when bool -> other types
// https://github.com/intel/llvm/issues/8543
#ifdef SYCL_CTS_COMPILING_WITH_DPCPP
        if constexpr (!std::is_same_v<${type}, bool>)
#endif
          if (!check_convert_as_all_types<${type}, ${size}>(inputVec)) {
            resAcc[0] = false;
          }
""")

lo_hi_odd_even_template = Template("""
        ${type} vals[] = {${vals}};
        if (!check_lo_hi_odd_even<${type}>(inputVec, vals)) {
          resAcc[0] = false;
        }
""")

as_convert_call_template = Template("""
        auto inputVec = sycl::vec<${type}, ${size}>(${vals});
        if (!check_convert_as_all_dims<${type}, ${size}, ${dest_type}>(inputVec)) {
            resAcc[0] = false;
        }
""")


def gen_checks(type_str, size):
    vals_list = append_fp_postfix(type_str, Data.vals_list_dict[size])
    if 'double' in type_str or 'half' in type_str or 'float' in type_str:
        vals_list =  Data.vals_list_dict_float[size]
    reverse_vals_list = vals_list[::-1]
    kernel_name = 'KERNEL_API_' + type_str + str(size)
    test_string = vector_api_template.substitute(
        type=type_str,
        size=size,
        vals=', '.join(vals_list),
        reversed_vals=', '.join(reverse_vals_list),
        swizIndexes=', '.join(Data.swizzle_elem_list_dict[size][::-1]))
    if 'double' in type_str:
        test_string += 'check_convert_as_all_dims<'+type_str +','+ str(
                size) + ', double>(inputVec);\n'
    if 'half' in type_str:
        test_string += 'check_convert_as_all_dims<'+type_str +','+ str(
                size) + ', sycl::half>(inputVec);\n'
    if size != 1:
        test_string += lo_hi_odd_even_template.substitute(
        type=type_str,
        vals=', '.join(vals_list))
    string = wrap_with_kernel(
        type_str, kernel_name,
        'API test for sycl::vec<' + type_str + ', ' + str(size) + '>',
        test_string)
    string+= vector_element_type_template.substitute(
        type=type_str,
        size=size,
        swizIndexes=', '.join(Data.swizzle_elem_list_dict[size][::-1]))
    return wrap_with_test_func(TEST_NAME, type_str, string, str(size))

def gen_optional_checks(type_str, size, dest, dest_type, TEST_NAME_OP):
    kernel_name = 'KERNEL_CONVERT_AS_' + type_str + str(size) + dest
    test_string = as_convert_call_template.substitute(
        type=type_str,
        size=size,
        vals=(', ' + type_str).join(Data.vals_list_dict_float[size]),
        dest_type=dest_type)

    string = wrap_with_kernel(
        dest_type, kernel_name,
        'convert() as() test for sycl::vec<' + type_str + ', ' + str(size) + '> to '+ dest,
        test_string)
    return wrap_with_test_func(TEST_NAME_OP, type_str, string, str(size))

def make_optional_tests(type_str, input_file, output_file, dest, dest_type):
    api_checks = ''
    func_calls = ''
    TEST_NAME_OP = TEST_NAME + '_as_convert_to_' + dest
    for size in Data.standard_sizes:
        api_checks += gen_optional_checks(type_str, size, dest, dest_type, TEST_NAME_OP)
        func_calls += make_func_call(TEST_NAME_OP, type_str, str(size))
    write_source_file(api_checks, func_calls, TEST_NAME_OP, input_file,
                    output_file.replace('.cpp','_as_convert_to_'+dest+'.cpp'), type_str)

def make_tests(type_str, input_file, output_file, target_enable):
    if type_str == 'bool':
        Data.vals_list_dict = cast_to_bool(Data.vals_list_dict)

    api_checks = ''
    func_calls = ''
    for size in Data.standard_sizes:
        api_checks += gen_checks(type_str, size)
        func_calls += make_func_call(TEST_NAME, type_str, str(size))
    write_source_file(api_checks, func_calls, TEST_NAME, input_file,
                      output_file, type_str)

    if '64' in target_enable and not('double' in type_str):
        make_optional_tests(type_str, input_file, output_file, 'fp64',
                            'double')

    if '16' in target_enable and not('half' in type_str):
        make_optional_tests(type_str, input_file, output_file, 'fp16',
                            'sycl::half')


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
    argparser.add_argument(
        '-target-enable',
        required=True,
        dest="target_enable",
        help='Option to generate tests for convert() and as() with double and half as target types')
    args = argparser.parse_args()

    make_tests(args.ty, args.template, args.output, args.target_enable)

if __name__ == '__main__':
    main()
