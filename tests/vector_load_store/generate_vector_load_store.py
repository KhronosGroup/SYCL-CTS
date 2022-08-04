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
from common_python_vec import (Data, append_fp_postfix, make_func_call,
                               wrap_with_test_func, write_source_file,
                               wrap_with_extension_checks)

TEST_NAME = 'LOAD_STORE'

load_store_test_template = Template(
    """        ${type} inputData${type_as_str}${size}[${size}] = {${in_order_vals}};
        ${type} outputData${type_as_str}${size}[${size}] = {${val}};
        ${type} swizzleInputData${type_as_str}${size}[${size}] = {${reverse_order_vals}};
        ${type} swizzleOutputData${type_as_str}${size}[${size}] = {${val}};
        {
          cl::sycl::buffer<${type}, 1> inBuffer${type_as_str}${size}(inputData${type_as_str}${size}, cl::sycl::range<1>(${size}));
          cl::sycl::buffer<${type}, 1> outBuffer${type_as_str}${size}(outputData${type_as_str}${size}, cl::sycl::range<1>(${size}));
          cl::sycl::buffer<${type}, 1> swizzleInBuffer${type_as_str}${size}(swizzleInputData${type_as_str}${size}, cl::sycl::range<1>(${size}));
          cl::sycl::buffer<${type}, 1> swizzleOutBuffer${type_as_str}${size}(swizzleOutputData${type_as_str}${size}, cl::sycl::range<1>(${size}));

          testQueue.submit([&](cl::sycl::handler &cgh) {
            auto inPtr${type_as_str}${size} = inBuffer${type_as_str}${size}.get_access<cl::sycl::access::mode::read_write>(cgh);
            auto outPtr${type_as_str}${size} = outBuffer${type_as_str}${size}.get_access<cl::sycl::access::mode::read_write>(cgh);

            auto swizzleInPtr${type_as_str}${size} = swizzleInBuffer${type_as_str}${size}.get_access<cl::sycl::access::mode::read_write>(cgh);
            auto swizzleOutPtr${type_as_str}${size} = swizzleOutBuffer${type_as_str}${size}.get_access<cl::sycl::access::mode::read_write>(cgh);

            cgh.single_task<class ${kernelName}>([=]() {
              auto testVec${type_as_str}${size} = cl::sycl::vec<${type}, ${size}>(${val});
              testVec${type_as_str}${size}.load(0, inPtr${type_as_str}${size});
              testVec${type_as_str}${size}.store(0, outPtr${type_as_str}${size});

              auto multiPtrIn${type_as_str}${size} = inPtr${type_as_str}${size}.get_pointer();
              cl::sycl::global_ptr<const ${type}> constMultiPtrIn${type_as_str}${size} = multiPtrIn${type_as_str}${size};
              auto multiPtrOut${type_as_str}${size} = outPtr${type_as_str}${size}.get_pointer();
              testVec${type_as_str}${size}.load(0, multiPtrIn${type_as_str}${size});
              testVec${type_as_str}${size}.load(0, constMultiPtrIn${type_as_str}${size});
              testVec${type_as_str}${size}.store(0, multiPtrOut${type_as_str}${size});

              auto cleanVec${type_as_str}${size} = cl::sycl::vec<${type}, ${size}>(${val});
              cl::sycl::vec<${type}, ${size}> swizzledVec {cleanVec${type_as_str}${size}.template swizzle<${swizVals}>()};
              swizzledVec.load(0, swizzleInPtr${type_as_str}${size});
              swizzledVec.store(0, swizzleOutPtr${type_as_str}${size});

              auto multiPtrInSwizzle${type_as_str}${size} = swizzleInPtr${type_as_str}${size}.get_pointer();
              cl::sycl::global_ptr<const ${type}> constMultiPtrInSwizzle${type_as_str}${size} = multiPtrInSwizzle${type_as_str}${size};
              auto multiPtrOutSwizzle${type_as_str}${size} = swizzleOutPtr${type_as_str}${size}.get_pointer();
              swizzledVec.load(0, multiPtrInSwizzle${type_as_str}${size});
              swizzledVec.load(0, constMultiPtrInSwizzle${type_as_str}${size});
              swizzledVec.store(0, multiPtrOutSwizzle${type_as_str}${size});
            });
          });

        }
        check_array_equality<${type}, ${size}>(log, inputData${type_as_str}${size}, outputData${type_as_str}${size});
        check_array_equality<${type}, ${size}>(log, swizzleInputData${type_as_str}${size}, swizzleOutputData${type_as_str}${size});

        testQueue.wait_and_throw();
      """)


def gen_kernel_name(type_str, size):
    return 'KERNEL_load_store_' + type_str.replace('cl::sycl::', '').replace(
        ' ', '') + str(size)


def gen_load_store_test(type_str, size):
    no_whitespace_type_str = type_str.replace(' ', '').replace(
        'cl::sycl::', '')
    test_string = load_store_test_template.substitute(
        type=type_str,
        type_as_str=no_whitespace_type_str,
        size=size,
        val=Data.value_default_dict[type_str],
        in_order_vals=', '.join(
            append_fp_postfix(type_str, Data.vals_list_dict[size])),
        reverse_order_vals=', '.join(
            append_fp_postfix(type_str, Data.vals_list_dict[size][::-1])),
        kernelName=gen_kernel_name(type_str, size),
        swizVals=', '.join(Data.swizzle_elem_list_dict[size]))
    return wrap_with_test_func(TEST_NAME, type_str,
                               wrap_with_extension_checks(
                                   type_str, test_string), str(size))


def make_tests(type_str, input_file, output_file):
    test_string = ''
    func_calls = ''
    for size in Data.standard_sizes:
        test_string += gen_load_store_test(type_str, size)
        func_calls += make_func_call(TEST_NAME, type_str, str(size))
    write_source_file(test_string, func_calls, TEST_NAME, input_file,
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
