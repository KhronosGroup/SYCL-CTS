# ************************************************************************
#
#   SYCL Conformance Test Suite
#
#   Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
#
# ************************************************************************

import sys
sys.path.append('../common/')
from common_python_vec import Data, replace_string_in_source_string, wrap_with_kernel
import itertools
from string import Template

default_constructor_vec_template = Template("""        auto test = cl::sycl::vec<${type}, ${size}>();
        if (check_equal_type_bool<cl::sycl::vec<${type}, ${size}>>(test)) {
          resAcc[0] = false;
        }
        if (check_vector_size<${type}, ${size}>(test)) {
          resAcc[0] = false;
        }
""")

explicit_constructor_vec_template = Template("""        const ${type} val = ${val};
        ${type} vals[] = {${vals}};
        auto test = cl::sycl::vec<${type}, ${size}>(val);
        if (check_equal_type_bool<cl::sycl::vec<${type}, ${size}>>(test)) {
          resAcc[0] = false;
        }
        if (check_vector_size<${type}, ${size}>(test)) {
          resAcc[0] = false;
        }
        if (check_vector_values<${type}, ${size}>(test, vals)) {
          resAcc[0] = false;
        }
""")

vec_constructor_vec_template = Template("""        auto test = cl::sycl::vec<${type}, ${size}>(${val});
        ${type} vals[] = {${vals}};
        if (check_equal_type_bool<cl::sycl::vec<${type}, ${size}>>(test)) {
          resAcc[0] = false;
        }
        if (check_vector_size<${type}, ${size}>(test)) {
          resAcc[0] = false;
        }
        if (check_vector_values<${type}, ${size}>(test, vals)) {
          resAcc[0] = false;
        }
""")

opencl_constructor_vec_template = Template("""        cl::sycl::vec<${type}, ${size}>::vector_t interopVec;
        auto test = cl::sycl::vec<${type}, ${size}>(interopVec);
""")


def generate_default(type_str, size):
    """Generates test for vec()"""
    test_string = default_constructor_vec_template.substitute(type=type_str,
                                                              size=size)
    return wrap_with_kernel(
        type_str,
        'DEFAULT_KERNEL_' +
        type_str.replace(
            'cl::sycl::',
            '').replace(
            ' ',
            '') +
        str(size),
        'Default constructor, cl::sycl::vec<' +
        type_str +
        ', ' +
        str(size) +
        '>',
        test_string)


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
    return wrap_with_kernel(type_str, 'EXPLICIT_KERNEL_' +
                            type_str.replace(
                                'cl::sycl::',
                                '').replace(
                                ' ',
                                '') +
                            str(size),
                            'Explicit constructor, cl::sycl::vec<' +
                            type_str +
                            ', ' +
                            str(size) +
                            '>',
                            test_string)


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
    return wrap_with_kernel(type_str, 'VEC_KERNEL_' +
                            type_str.replace(
                                'cl::sycl::',
                                '').replace(
                                ' ',
                                '') +
                            str(size),
                            'const &vec constructor, cl::sycl::vec<' +
                            type_str +
                            ', ' +
                            str(size) +
                            '>', test_string)


def generate_opencl(type_str, size):
    """Generates test for vec(vector_t openclVector)"""
    test_string = opencl_constructor_vec_template.substitute(
        type=type_str, size=size)
    return wrap_with_kernel(
        type_str,
        'OPENCL_KERNEL_' +
        type_str.replace(
            'cl::sycl::',
            '').replace(
            ' ',
            '') +
        str(size),
        'vec(vector_t openclVector), cl::sycl::vec<' +
        type_str +
        ', ' +
        str(size) +
        '>',
        test_string)


def generate_constructor_tests(input_file, output_file):
    """Generates a string for each constructor type containing each combination of test
    Constructor types: default, explicit, vec, opencl
    A cross section of variadic constructors are provided by the template"""
    opencl_tests = ''

    # Test with type_str='char'
    for size in Data.standard_sizes:
        default_tests = generate_default('char', size)
        explicit_tests = generate_explicit('char', size)
        vec_tests = generate_vec('char', size)

    for type_name in Data.standard_types:
        for sign in Data.signs:
            if (type_name == 'float' or type_name ==
                    'double' or type_name == 'cl::sycl::half') and sign is False:
                continue
            type_str = Data.standard_type_dict[(sign, type_name)]
            for size in Data.standard_sizes:
                default_tests += generate_default(type_str, size)
                explicit_tests += generate_explicit(type_str, size)
                vec_tests += generate_vec(type_str, size)

    for type_name in Data.opencl_types:
        for sign in Data.signs:
            if (type_name == 'cl::sycl::cl_float' or type_name ==
                    'cl::sycl::cl_double' or type_name == 'cl::sycl::cl_half') and sign is False:
                continue
            type_str = Data.opencl_type_dict[(sign, type_name)]
            for size in [2, 3, 4, 8, 16]:
                default_tests += generate_default(type_str, size)
                explicit_tests += generate_explicit(type_str, size)
                vec_tests += generate_vec(type_str, size)
                opencl_tests += generate_opencl(type_str, size)

    with open(input_file, 'r') as stuff:
        source = stuff.read()

    source = replace_string_in_source_string(
        source, default_tests, '$DEFAULT_TESTS')
    source = replace_string_in_source_string(
        source, explicit_tests, '$EXPLICIT_TESTS')
    source = replace_string_in_source_string(source, vec_tests, '$VEC_TESTS')
    source = replace_string_in_source_string(
        source, opencl_tests, '$OPENCL_TESTS')

    with open(output_file, 'w+') as output:
        output.write(source)


def main():
    generate_constructor_tests(
        'vector_constructors.template', 'vector_constructors.cpp')


if __name__ == '__main__':
    main()
