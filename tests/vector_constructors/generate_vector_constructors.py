# ************************************************************************
#
#   SYCL Conformance Test Suite
#
#   Copyright:	(c) 2018 by Codeplay Software LTD. All Rights Reserved.
#
# ************************************************************************

import sys
import itertools
from string import Template
sys.path.append('../common/')
from common_python_vec import (Data, wrap_with_kernel, wrap_with_test_func,
                               make_func_call, write_source_file)

TEST_NAME = 'CONSTRUCTORS'

default_constructor_vec_template = Template(
    """        auto test = cl::sycl::vec<${type}, ${size}>();
        if (!check_equal_type_bool<cl::sycl::vec<${type}, ${size}>>(test)) {
          resAcc[0] = false;
        }
        if (!check_vector_size<${type}, ${size}>(test)) {
          resAcc[0] = false;
        }
""")

explicit_constructor_vec_template = Template(
    """        const ${type} val = ${val};
        ${type} vals[] = {${vals}};
        auto test = cl::sycl::vec<${type}, ${size}>(val);
        if (!check_equal_type_bool<cl::sycl::vec<${type}, ${size}>>(test)) {
          resAcc[0] = false;
        }
        if (!check_vector_size<${type}, ${size}>(test)) {
          resAcc[0] = false;
        }
        if (!check_vector_values<${type}, ${size}>(test, vals)) {
          resAcc[0] = false;
        }
""")

vec_constructor_vec_template = Template(
    """        auto test = cl::sycl::vec<${type}, ${size}>(${val});
        ${type} vals[] = {${vals}};
        if (!check_equal_type_bool<cl::sycl::vec<${type}, ${size}>>(test)) {
          resAcc[0] = false;
        }
        if (!check_vector_size<${type}, ${size}>(test)) {
          resAcc[0] = false;
        }
        if (!check_vector_values<${type}, ${size}>(test, vals)) {
          resAcc[0] = false;
        }
""")

opencl_constructor_vec_template = Template(
    """        cl::sycl::vec<${type}, ${size}>::vector_t interopVec;
        auto test = cl::sycl::vec<${type}, ${size}>(interopVec);
""")


def generate_default(type_str, size):
    """Generates test for vec()"""
    test_string = default_constructor_vec_template.substitute(
        type=type_str, size=size)
    return wrap_with_kernel(
        type_str, 'VEC_DEFAULT_CONSTRUCTOR_KERNEL_' + type_str.replace(
            'cl::sycl::', '').replace(' ', '') + str(size),
        'Default constructor, cl::sycl::vec<' + type_str + ', ' + str(size) +
        '>', test_string)


def generate_explicit(type_str, size):
    """Generates test for vec(const T &arg)"""
    val_list = []
    for _ in itertools.repeat(None, size):
        val_list.append(Data.value_default_dict[type_str])
    test_string = explicit_constructor_vec_template.substitute(
        type=type_str,
        size=size,
        val=Data.value_default_dict[type_str],
        vals=', '.join(val_list))
    return wrap_with_kernel(
        type_str, 'VEC_EXPLICIT_CONSTRUCTOR_KERNEL_' + type_str.replace(
            'cl::sycl::', '').replace(' ', '') + str(size),
        'Explicit constructor, cl::sycl::vec<' + type_str + ', ' + str(size) +
        '>', test_string)


def generate_vec(type_str, size):
    """Generates test for vec<T, dims>(const &vec<T, dims>)"""
    val_list = []
    for _ in itertools.repeat(None, size):
        val_list.append(Data.value_default_dict[type_str])
    test_string = vec_constructor_vec_template.substitute(
        type=type_str,
        size=size,
        val=Data.value_default_dict[type_str],
        vals=', '.join(val_list))
    return wrap_with_kernel(type_str,
                            'VEC_VEC_CONSTRUCTOR_KERNEL_' + type_str.replace(
                                'cl::sycl::', '').replace(' ', '') + str(size),
                            'const &vec constructor, cl::sycl::vec<' + type_str
                            + ', ' + str(size) + '>', test_string)


def generate_opencl(type_str, size):
    """Generates test for vec(vector_t openclVector)"""
    test_string = opencl_constructor_vec_template.substitute(
        type=type_str, size=size)
    return '#ifdef __SYCL_DEVICE_ONLY__\n' + wrap_with_kernel(
        type_str, 'VEC_OPENCL_CONSTRUCTOR_KERNEL_' + type_str.replace(
            'cl::sycl::', '').replace(' ', '') + str(size),
        'vec(vector_t openclVector), cl::sycl::vec<' + type_str + ', ' +
        str(size) + '>', test_string) + '#endif  // __SYCL_DEVICE_ONLY__\n'


def generate_constructor_tests(input_file, output_file):
    """Generates a string for each constructor type containing each combination of test
    Constructor types: default, explicit, vec, opencl
    A cross section of variadic constructors are provided by the template"""
    default_tests = ''
    explicit_tests = ''
    vec_tests = ''
    opencl_tests = ''

    # Test with type_str='char'
    test_str = ''
    test_func_str = ''
    func_calls = ''
    for size in Data.standard_sizes:
        test_str += generate_default('char', size)
        test_str += generate_explicit('char', size)
        test_str += generate_vec('char', size)
        test_func_str += wrap_with_test_func(TEST_NAME, 'char', test_str,
                                             str(size))
        test_str = ''
        func_calls += make_func_call(TEST_NAME, 'char', str(size))
    write_source_file(test_func_str, func_calls, TEST_NAME, input_file,
                      output_file, 'char')

    for type_name in Data.standard_types:
        for sign in Data.signs:
            if (type_name == 'float' or type_name == 'double'
                    or type_name == 'cl::sycl::half') and sign is False:
                continue
            type_str = Data.standard_type_dict[(sign, type_name)]
            test_str = ''
            test_func_str = ''
            func_calls = ''
            for size in Data.standard_sizes:
                test_str += generate_default(type_str, size)
                test_str += generate_explicit(type_str, size)
                test_str += generate_vec(type_str, size)
                test_func_str += wrap_with_test_func(TEST_NAME, type_str,
                                                     test_str, str(size))
                test_str = ''
                func_calls += make_func_call(TEST_NAME, type_str, str(size))
            write_source_file(test_func_str, func_calls, TEST_NAME, input_file,
                              output_file, type_str)

    for type_name in Data.opencl_types:
        for sign in Data.signs:
            if (type_name == 'cl::sycl::cl_float'
                    or type_name == 'cl::sycl::cl_double'
                    or type_name == 'cl::sycl::cl_half') and sign is False:
                continue
            type_str = Data.opencl_type_dict[(sign, type_name)]
            test_str = ''
            test_func_str = ''
            func_calls = ''
            for size in [2, 3, 4, 8, 16]:
                test_str += generate_default(type_str, size)
                test_str += generate_explicit(type_str, size)
                test_str += generate_vec(type_str, size)
                test_str += generate_opencl(type_str, size)
                test_func_str += wrap_with_test_func(TEST_NAME, type_str,
                                                     test_str, str(size))
                test_str = ''
                func_calls += make_func_call(TEST_NAME, type_str, str(size))
            write_source_file(test_func_str, func_calls, TEST_NAME, input_file,
                              output_file, type_str)


def main():
    generate_constructor_tests('../common/vector.template',
                               'vector_constructors.cpp')


if __name__ == '__main__':
    main()
