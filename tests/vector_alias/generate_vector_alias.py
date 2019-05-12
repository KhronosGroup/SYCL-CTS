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
from common_python_vec import (Data, wrap_with_kernel, wrap_with_test_func,
                               make_func_call, write_source_file)

TEST_NAME = 'ALIAS'

alias_test_template = Template("""
        auto aliasVec = ${aliasVecName}();
        resAcc[0] = check_equal_type_bool<cl::sycl::vec<${typeName}, ${size}>>(aliasVec);
""")


def gen_alias_test(type_str, size):
    alias_vec_name = Data.alias_dict[type_str] + str(size)
    test_string = alias_test_template.substitute(
        aliasVecName=alias_vec_name, typeName=type_str, size=size)
    string = wrap_with_kernel(
        type_str, 'KERNEL_alias_' + alias_vec_name.replace('cl::sycl::', ''),
        'Alias vector test: ' + alias_vec_name, test_string)
    return wrap_with_test_func(TEST_NAME, type_str, string, str(size))


def make_tests(type_str, input_file, output_file):
    alias_test = ''
    func_calls = ''
    for size in [2, 3, 4, 8, 16]:
        alias_test += gen_alias_test(type_str, size)
        func_calls += make_func_call(TEST_NAME, type_str, str(size))
    write_source_file(alias_test, func_calls, TEST_NAME, input_file,
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
