# ************************************************************************
#
#   SYCL Conformance Test Suite
#
#   Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
#
# ************************************************************************

import sys
sys.path.append('../common/')
from common_python_vec import Data, replace_string_in_source_string
import itertools
from string import Template
from collections import defaultdict


opencl_types = ['char', 'short', 'int', 'long', 'float', 'double', 'half']
opencl_sizes = [1, 2, 3, 4, 8, 16]
opencl_type_dict = {(False, 'char'): 'cl_uchar',
                    (True, 'char'): 'cl_char',
                    (False, 'short'): 'cl_ushort',
                    (True, 'short'): 'cl_short',
                    (False, 'int'): 'cl_uint',
                    (True, 'int'): 'cl_int',
                    (False, 'long'): 'cl_ulong',
                    (True, 'long'): 'cl_long',
                    (False, 'float'): 'cl_float',
                    (True, 'float'): 'cl_float',
                    (False, 'double'): 'cl_double',
                    (True, 'double'): 'cl_double',
                    (False, 'half'): 'cl_half',
                    (True, 'half'): 'cl_half'}
value_dict = defaultdict(lambda: '1',
                         float='1.0f',
                         double='1.0')


def get_standard_type_str(standard_type, signed):
    """Gets standard type string from dictionary"""
    return Data.standard_type_dict[(signed, standard_type)]


def get_opencl_type_str(opencl_type, signed):
    """Gets opencl type string from dictionary"""
    return opencl_type_dict[(signed, opencl_type)]

def get_val_for_type(base_type):
    """Returns a value as a string for each base_type"""
    return value_dict[base_type]


def generate_value_array(base_type, size, signed):
    """Generates "TYPE vals[] = {...};\n" """
    type_str = get_standard_type_str(base_type, signed)
    # string used for constructing {...}
    val_str = get_val_for_type(base_type)
    # Construct the string
    string = '  ' + type_str + ' vals[] = {'
    for _ in itertools.repeat(None, size - 1):
        string += val_str + ', '
    string += val_str + '};\n'
    return string


check_equal_type_template = Template(
    """  check_equal_type<vec<${type}, ${size}>>(log, test,
  "Testing constructor ${conname}");\n""")
check_vector_size_template = Template(
    """  check_vector_size<${type}, ${size}>(log, test);\n""")
check_vector_values_template = Template(
    """  check_vector_values<${type}, ${size}>(log, test, vals);\n""")


def generate_check_equal_type(base_type, size, signed, con_name):
    """Generates "check_equal_type<vec<TYPE, SIZE>>(log, test,\n
    "Testing constructor CONNAME");\n"""
    type_str = get_standard_type_str(base_type, signed)
    return check_equal_type_template.substitute(
        type=type_str, size=str(size), conname=con_name)


def generate_check_vector_size(base_type, size, signed):
    """Generates "check_vector_size<TYPE, SIZE>(log, test);\n" """
    type_str = get_standard_type_str(base_type, signed)
    return check_vector_size_template.substitute(type=type_str, size=str(size))


def generate_check_vector_values(base_type, size, signed):
    """Generates "check_vector_values<TYPE, SIZE>(log, test, vals);\n" """
    type_str = get_standard_type_str(base_type, signed)
    return check_vector_values_template.substitute(
        type=type_str, size=str(size))


default_constructor_vec_template = Template(
    """auto test = vec<${type}, ${size}>();\n""")


def generate_default(base_type, size, signed):
    """Generates test for vec()"""
    type_str = get_standard_type_str(base_type, signed)
    # test vector
    string = '{ '
    string += default_constructor_vec_template.substitute(
        type=type_str, size=size)
    # Common util calls
    string += generate_check_equal_type(base_type, size, signed, 'vec()')
    string += generate_check_vector_size(base_type, size, signed)
    string += '}\n'
    return string


explicit_constructor_val_template = Template(
    """const ${type} val = ${val};\n""")
explicit_constructor_vec_template = Template(
    """  auto test = vec<${type}, ${size}>(val);\n""")


def generate_explicit(base_type, size, signed):
    """Generates test for vec(const T &arg)"""
    type_str = get_standard_type_str(base_type, signed)
    # const val
    string = '{ '
    string += explicit_constructor_val_template.substitute(
        type=type_str, val=get_val_for_type(base_type))
    # Generate val array
    string += generate_value_array(base_type, size, signed)
    # test vector
    string += explicit_constructor_vec_template.substitute(
        type=type_str, size=size)
    # Common util calls
    string += generate_check_equal_type(base_type,
                                        size, signed, 'vec(const T &arg)')
    string += generate_check_vector_size(base_type, size, signed)
    string += generate_check_vector_values(base_type, size, signed)
    string += '}\n'
    return string


vec_constructor_anotherVec_template = Template(
    """vec<${type}, ${size}> anotherVec = vec<${type}, ${size}>(${val});\n""")
vec_constructor_vec_template = Template(
    """  auto test = vec<${type}, ${size}>(anotherVec);\n""")


def generate_vec(base_type, size, signed):
    """Generates test for vec<T, dims>(const &vec<T, dims>)"""
    type_str = get_standard_type_str(base_type, signed)
    # Starting vector
    string = '{ '
    string += vec_constructor_anotherVec_template.substitute(
        type=type_str, size=size, val=get_val_for_type(base_type))
    # test vector
    string += vec_constructor_vec_template.substitute(type=type_str, size=size)
    # Generate val array
    string += generate_value_array(base_type, size, signed)
    # Common util calls
    string += generate_check_equal_type(
        base_type, size, signed, 'vec<T, dims>(const &vec<T, dims>)')
    string += generate_check_vector_size(base_type, size, signed)
    string += generate_check_vector_values(base_type, size, signed)
    string += '}\n'
    return string


opencl_constructor_interopVec_template = Template(
    """auto interopVec = cl::sycl::${opencl_type}();\n""")
opencl_constructor_vec_template = Template(
    """  auto test = vec<${type}, ${size}>(interopVec);\n""")


def generate_opencl(base_type, size, signed):
    """Generates test for vec(vector_t openclVector)"""
    if base_type is 'long':
        type_str = get_standard_type_str('long long', signed)
    else:
        type_str = get_standard_type_str(base_type, signed)
    # OpenCL interop vec
    string = '{ '
    string += opencl_constructor_interopVec_template.substitute(
        opencl_type=get_opencl_type_str(base_type, signed) + str(size))
    # test vector
    string += opencl_constructor_vec_template.substitute(
        type=type_str, size=size)
    string += '}\n'
    return string


def generate_constructor_tests(input_file, output_file):
    """Generates a string for each constructor type containing each combination of test
    Constructor types: default, explicit, vec, opencl
    A cross section of variadic constructors are provided by the template"""
    default_tests = ''
    explicit_tests = ''
    vec_tests = ''
    opencl_tests = ''

    for type_name in Data.standard_types:
        for sign in Data.signs:
            if (type_name == 'float' or type_name == 'double' or type_name == 'half') and sign is False:
                continue
            for size in Data.standard_sizes:
                default_tests += generate_default(type_name, size, sign)
                explicit_tests += generate_explicit(type_name, size, sign)
                vec_tests += generate_vec(type_name, size, sign)

    for type_name in opencl_types:
        for sign in Data.signs:
            if (type_name == 'float' or type_name == 'double' or type_name == 'half') and sign is False:
                continue
            for size in opencl_sizes:
                opencl_tests += generate_opencl(type_name, size, sign)

    with open(input_file, 'r') as stuff:
        source = stuff.read()

    source = replace_string_in_source_string(source, default_tests, '$DEFAULT_TESTS')
    source = replace_string_in_source_string(source, explicit_tests, '$EXPLICIT_TESTS')
    source = replace_string_in_source_string(source, vec_tests, '$VEC_TESTS')
    source = replace_string_in_source_string(source, opencl_tests, '$OPENCL_TESTS')

    with open(output_file, 'w+') as output:
        output.write(source)


def main():
    generate_constructor_tests(
        'vector_constructors.template', 'vector_constructors.cpp')


if __name__ == '__main__':
    main()
