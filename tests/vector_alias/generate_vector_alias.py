# ************************************************************************
#
#   SYCL Conformance Test Suite
#
#   Copyright:	(c) 2018 by Codeplay Software LTD. All Rights Reserved.
#
# ************************************************************************

import sys
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


def make_tests(input_file, output_file):

    # Test with type_str='char'
    alias_test = ''
    func_calls = ''
    for size in [2, 3, 4, 8, 16]:
        alias_test += gen_alias_test('char', size)
        func_calls += make_func_call(TEST_NAME, 'char', str(size))
    write_source_file(alias_test, func_calls, TEST_NAME, input_file,
                      output_file, 'char')

    for base_type in Data.standard_types:
        for sign in Data.signs:
            if (base_type == 'float' or base_type == 'double'
                    or base_type == 'cl::sycl::half') and sign is False:
                continue
            type_str = Data.standard_type_dict[(sign, base_type)]
            alias_test = ''
            func_calls = ''
            for size in [2, 3, 4, 8, 16]:
                alias_test += gen_alias_test(type_str, size)
                func_calls += make_func_call(TEST_NAME, type_str, str(size))
            write_source_file(alias_test, func_calls, TEST_NAME, input_file,
                              output_file, type_str)

    for base_type in Data.opencl_types:
        for sign in Data.signs:
            if (base_type == 'cl::sycl::cl_float'
                    or base_type == 'cl::sycl::cl_double'
                    or base_type == 'cl::sycl::cl_half') and sign is False:
                continue
            type_str = Data.opencl_type_dict[(sign, base_type)]
            alias_test = ''
            func_calls = ''
            for size in [2, 3, 4, 8, 16]:
                alias_test += gen_alias_test(type_str, size)
                func_calls += make_func_call(TEST_NAME, type_str, str(size))
            write_source_file(alias_test, func_calls, TEST_NAME, input_file,
                              output_file, type_str)


def main():
    make_tests('../common/vector.template', 'vector_alias.cpp')


if __name__ == '__main__':
    main()
