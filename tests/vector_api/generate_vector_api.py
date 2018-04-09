# ************************************************************************
#
#   SYCL Conformance Test Suite
#
#   Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
#
# ************************************************************************

import sys
sys.path.append('../common/')
from common_python_vec import (
    Data, replace_string_in_source_string, append_fp_postfix, wrap_with_kernel,
    wrap_with_test_func, make_func_call, write_source_file)
from string import Template

TEST_NAME = 'API'

vector_api_template = Template("""
        auto inputVec = cl::sycl::vec<${type}, ${size}>(${vals});
        ${type} vals[] = {${vals}};
        ${type} reversed_vals[] = {${reversed_vals}};
        if (!check_vector_member_functions<${type}, ${size}, ${convertType}, ${asType}>(inputVec, vals)) {
          resAcc[0] = false;
        }
        cl::sycl::vec<${type}, ${size}> swizzledVec {inputVec.template swizzle<${swizIndexes}>()};
        if (!check_vector_values<${type}, ${size}>(swizzledVec, reversed_vals)) {
          resAcc[0] = false;
        }
""")

lo_hi_odd_even_template = Template("""
        if (!check_lo_hi_odd_even<${type}, ${size}>(inputVec, vals)) {
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
        test_string += lo_hi_odd_even_template.substitute(
            type=type_str, size=size)
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
        test_string += lo_hi_odd_even_template.substitute(
            type=type_str, size=size)
    string = wrap_with_kernel(
        type_str, kernel_name,
        'API test for cl::sycl::vec<' + type_str + ', ' + str(size) + '>',
        test_string)
    return wrap_with_test_func(TEST_NAME, type_str, string, str(size))


def make_tests(input_file, output_file):

    # Test with type_str='char'
    api_checks = ''
    func_calls = ''
    for size in Data.standard_sizes:
        api_checks += gen_host_checks('char', 'char', size)
        func_calls += make_func_call(TEST_NAME, 'char', str(size))
    write_source_file(api_checks, func_calls, TEST_NAME, input_file,
                      output_file, 'char')

    for base_type in Data.standard_types:
        if (base_type.count('half') is not 0):
            continue
        for sign in Data.signs:
            if (base_type == 'float' or base_type == 'double'
                    or base_type == 'cl::sycl::half') and sign is False:
                continue
            type_str = Data.standard_type_dict[(sign, base_type)]
            reverse_type_str = Data.standard_type_dict[(not sign, base_type)]
            api_checks = ''
            func_calls = ''
            for size in Data.standard_sizes:
                api_checks += gen_host_checks(type_str, reverse_type_str, size)
                func_calls += make_func_call(TEST_NAME, type_str, str(size))
            write_source_file(api_checks, func_calls, TEST_NAME, input_file,
                              output_file, type_str)

    for base_type in Data.opencl_types:
        if (base_type.count('half') is not 0):
            continue
        for sign in Data.signs:
            if (base_type == 'cl::sycl::cl_float'
                    or base_type == 'cl::sycl::cl_double'
                    or base_type == 'cl::sycl::cl_half') and sign is False:
                continue
            type_str = Data.opencl_type_dict[(sign, base_type)]
            reverse_type_str = Data.opencl_type_dict[(not sign, base_type)]
            api_checks = ''
            func_calls = ''
            for size in Data.standard_sizes:
                api_checks += gen_interop_checks(type_str, reverse_type_str,
                                                 size)
                func_calls += make_func_call(TEST_NAME, type_str, str(size))
            write_source_file(api_checks, func_calls, TEST_NAME, input_file,
                              output_file, type_str)


def main():
    make_tests('../common/vector.template', 'vector_api.cpp')


if __name__ == '__main__':
    main()
