#!/usr/bin/env python3
# ************************************************************************
#
#   SYCL Conformance Test Suite
#
#   Copyright:	(c) 2018 by Codeplay Software LTD. All Rights Reserved.
#
# ************************************************************************

import sys
import argparse
from string import Template
sys.path.append('../common/')
from common_python_vec import (Data, append_fp_postfix, make_func_call,
                               wrap_with_test_func, write_source_file,
                               wrap_with_extension_checks, get_types,
                               remove_namespaces_whitespaces)

TEST_NAME = 'LOAD_STORE'

load_store_test_template = Template(
    """        ${type} inputData${type_as_str}${size}[${size}] = {${in_order_vals}};
        ${type} outputData${type_as_str}${size}[${size}] = {${val}};
        ${type} swizzleInputData${type_as_str}${size}[${size}] = {${reverse_order_vals}};
        ${type} swizzleOutputData${type_as_str}${size}[${size}] = {${val}};
        {
          sycl::buffer<${type}, 1> inBuffer${type_as_str}${size}(inputData${type_as_str}${size}, sycl::range<1>(${size}));
          sycl::buffer<${type}, 1> outBuffer${type_as_str}${size}(outputData${type_as_str}${size}, sycl::range<1>(${size}));
          sycl::buffer<${type}, 1> swizzleInBuffer${type_as_str}${size}(swizzleInputData${type_as_str}${size}, sycl::range<1>(${size}));
          sycl::buffer<${type}, 1> swizzleOutBuffer${type_as_str}${size}(swizzleOutputData${type_as_str}${size}, sycl::range<1>(${size}));

          testQueue.submit([&](sycl::handler &cgh) {
            auto inPtr${type_as_str}${size} = inBuffer${type_as_str}${size}.get_access<sycl::access_mode::read_write>(cgh);
            auto outPtr${type_as_str}${size} = outBuffer${type_as_str}${size}.get_access<sycl::access_mode::read_write>(cgh);

            auto swizzleInPtr${type_as_str}${size} = swizzleInBuffer${type_as_str}${size}.get_access<sycl::access_mode::read_write>(cgh);
            auto swizzleOutPtr${type_as_str}${size} = swizzleOutBuffer${type_as_str}${size}.get_access<sycl::access_mode::read_write>(cgh);

            cgh.single_task<class ${kernelName}>([=]() {
              auto testVec${type_as_str}${size} = sycl::vec<${type}, ${size}>(${val});
              testVec${type_as_str}${size}.load(0, inPtr${type_as_str}${size});
              testVec${type_as_str}${size}.store(0, outPtr${type_as_str}${size});

              auto multiPtrIn${type_as_str}${size} = inPtr${type_as_str}${size}.get_pointer();
              sycl::global_ptr<const ${type}> constMultiPtrIn${type_as_str}${size} = multiPtrIn${type_as_str}${size};
              auto multiPtrOut${type_as_str}${size} = outPtr${type_as_str}${size}.get_pointer();
              testVec${type_as_str}${size}.load(0, multiPtrIn${type_as_str}${size});
              testVec${type_as_str}${size}.load(0, constMultiPtrIn${type_as_str}${size});
              testVec${type_as_str}${size}.store(0, multiPtrOut${type_as_str}${size});

              auto cleanVec${type_as_str}${size} = sycl::vec<${type}, ${size}>(${val});
              sycl::vec<${type}, ${size}> swizzledVec {cleanVec${type_as_str}${size}.template swizzle<${swizVals}>()};
              swizzledVec.load(0, swizzleInPtr${type_as_str}${size});
              swizzledVec.store(0, swizzleOutPtr${type_as_str}${size});

              auto multiPtrInSwizzle${type_as_str}${size} = swizzleInPtr${type_as_str}${size}.get_pointer();
              sycl::global_ptr<const ${type}> constMultiPtrInSwizzle${type_as_str}${size} = multiPtrInSwizzle${type_as_str}${size};
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
    return 'KERNEL_load_store_' + remove_namespaces_whitespaces(type_str) + str(size)


def gen_load_store_test(type_str, size):
    no_whitespace_type_str = remove_namespaces_whitespaces(type_str)
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
