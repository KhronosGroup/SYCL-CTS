# ************************************************************************
#
#   SYCL Conformance Test Suite
#
#   Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
#
# ************************************************************************

import sys
sys.path.append('../common/')
from common_python_vec import Data, replace_string_in_source_string, append_fp_postfix, wrap_with_kernel
from string import Template


vector_api_template = Template("""
        auto inputVec = cl::sycl::vec<${type}, ${size}>(${vals});
        ${type} vals[] = {${vals}};
        ${type} reversed_vals[] = {${reversed_vals}};
        if (check_vector_member_functions<${type}, ${size}, ${convertType}, ${asType}>(inputVec, vals)) {
            resAcc[0] = false;
        }
        cl::sycl::vec<${type}, ${size}> swizzledVec = inputVec.template swizzle<${swizIndexes}>();
        if (check_vector_values<${type}, ${size}>(swizzledVec, reversed_vals)) {
            resAcc[0] = false;
        }
""")

lo_hi_odd_even_template = Template("""      check_lo_hi_odd_even<${type}, ${size}>(inputVec, vals);
""")


def gen_host_checks(type_str, reverse_type_str, size):
    """Uses the above string templates to generate tests for each vec api function except load and store.
    Load and store are handled separately.
    lo() hi() odd() and even() are handled with a separate function and template to other api functions
    as they can only be performed on vectors of size 2 or greater."""
    vals_list = append_fp_postfix(type_str, Data.vals_list_dict[size])
    reverse_vals_list = vals_list[::-1]
    test_string = vector_api_template.substitute(
        type=type_str,
        size=size,
        kernelName='KERNEL_' + type_str.replace('cl::sycl::', '').replace(' ', '') + str(size),
        vals=', '.join(vals_list),
        reversed_vals=', '.join(reverse_vals_list),
        convertType=reverse_type_str,
        asType=reverse_type_str,
        swizIndexes=', '.join(Data.swizzle_elem_list_dict[size][::-1]))
    if size != 1:
        test_string += lo_hi_odd_even_template.substitute(
            type=type_str, size=size)
    string = wrap_with_kernel(
        type_str,
        'KERNEL_' +
        type_str.replace(
            'cl::sycl::',
            '').replace(
            ' ',
            '') +
        str(size),
        'API test for cl::sycl::vec<' +
        type_str +
        ', ' +
        str(size) +
        '>',
        test_string)
    return string


def gen_interop_checks(type_str, reverse_type_str, size):
    vals_list = append_fp_postfix(type_str, Data.vals_list_dict[size])
    reverse_vals_list = vals_list[::-1]
    test_string = vector_api_template.substitute(
        type=type_str,
        size=size,
        kernelName='KERNEL_INTEROP_T_' + type_str.replace('cl::sycl::', '').replace(' ', '') + str(size),
        vals=', '.join(vals_list),
        reversed_vals=', '.join(reverse_vals_list),
        convertType=reverse_type_str,
        asType=reverse_type_str,
        swizIndexes=', '.join(Data.swizzle_elem_list_dict[size][::-1]))
    if size != 1:
        test_string += lo_hi_odd_even_template.substitute(
            type=type_str, size=size)
    string = wrap_with_kernel(
        type_str,
        'KERNEL_' +
        type_str.replace(
            'cl::sycl::',
            '').replace(
            ' ',
            '') +
        str(size),
        'API test for cl::sycl::vec<' +
        type_str +
        ', ' +
        str(size) +
        '>',
        test_string)
    return string


def make_tests(input_file, output_file):
    host_api_checks = ''
    interop_api_checks = ''

    # Test with type_str='char'
    for size in Data.standard_sizes:
        host_api_checks += gen_host_checks('char', 'char', size)

    for base_type in Data.standard_types:
        for sign in Data.signs:
            if (base_type == 'float' or base_type ==
                    'double' or base_type == 'cl::sycl::half') and sign is False:
                continue
            type_str = Data.standard_type_dict[(sign, base_type)]
            reverse_type_str = Data.standard_type_dict[(not sign, base_type)]
            for size in Data.standard_sizes:
                host_api_checks += gen_host_checks(type_str,
                                                   reverse_type_str, size)
    for base_type in Data.opencl_types:
        for sign in Data.signs:
            if (base_type == 'cl::sycl::cl_float' or base_type ==
                    'cl::sycl::cl_double' or base_type == 'cl::sycl::cl_half') and sign is False:
                continue
            type_str = Data.opencl_type_dict[(sign, base_type)]
            reverse_type_str = Data.opencl_type_dict[(not sign, base_type)]
            for size in Data.standard_sizes:
                interop_api_checks += gen_interop_checks(
                    type_str, reverse_type_str, size)

    with open(input_file, 'r') as source_file:
        source = source_file.read()

    source = replace_string_in_source_string(
        source, host_api_checks, '$HOST_API_CHECKS')
    source = replace_string_in_source_string(
        source, interop_api_checks, '$INTEROP_API_CHECKS')

    with open(output_file, 'w+') as output:
        output.write(source)


def main():
    make_tests('vector_api.template', 'vector_api.cpp')


if __name__ == '__main__':
    main()
