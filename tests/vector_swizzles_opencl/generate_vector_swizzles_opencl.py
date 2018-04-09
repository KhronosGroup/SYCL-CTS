# ************************************************************************
#
#   SYCL Conformance Test Suite
#
#   Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
#
# ************************************************************************

import sys
sys.path.append('../common/')
from common_python_vec import (Data, replace_string_in_source_string,
                               swap_pairs, generate_value_list,
                               append_fp_postfix, wrap_with_kernel)
from string import Template
from itertools import product

swizzle_template = Template(
    """        cl::sycl::vec<${type}, ${size}> ${name}DimTestVec = cl::sycl::vec<${type}, ${size}>(${testVecValues});
        cl::sycl::vec<${type}, ${size}> swizzledVec {${name}DimTestVec.${indexes}()};
        ${type} in_order_vals[] = {${in_order_vals}};
        ${type} reversed_vals[] = {${reversed_vals}};
        ${type} in_order_reversed_pair_vals[] = {${in_order_pair_vals}};
        ${type} reverse_order_reversed_pair_vals[] = {${reverse_order_pair_vals}};
        if (!check_equal_type_bool<cl::sycl::vec<${type}, ${size}>>(swizzledVec)) {
            resAcc[0] = false;
        }
        if (!check_vector_size<${type}, ${size}>(swizzledVec)) {
            resAcc[0] = false;
        }
        if (!check_vector_values<${type}, ${size}>(swizzledVec, in_order_vals)) {
            resAcc[0] = false;
        }
        if (!check_vector_member_functions<${type}, ${size}, ${convert_type}, ${as_type}>(swizzledVec, in_order_vals)) {
            resAcc[0] = false;
        }
""")

lo_hi_odd_even_template = Template(
    """        if (!check_lo_hi_odd_even<${type}, ${size}>(swizzledVec, in_order_vals)) {
          resAcc[0] = false;
        }
""")

swizzle_elem_template = Template(
    """        cl::sycl::vec<${type}, ${size}> inOrderSwizzleFunctionVec {swizzledVec.template swizzle<${in_order_swiz_indexes}>()};
        if (!check_vector_values<${type}, ${size}>(inOrderSwizzleFunctionVec, in_order_vals)) {
            resAcc[0] = false;
        }
        cl::sycl::vec<${type}, ${size}> reverseOrderSwizzleFunctionVec {swizzledVec.template swizzle<${reverse_order_swiz_indexes}>()};
        if (!check_vector_values<${type}, ${size}>(reverseOrderSwizzleFunctionVec, reversed_vals)) {
            resAcc[0] = false;
        }
        cl::sycl::vec<${type}, ${size}> inOrderReversedPairSwizzleFunctionVec {swizzledVec.template swizzle<${in_order_reversed_pair_swiz_indexes}>()};
        if (!check_vector_values<${type}, ${size}>(inOrderReversedPairSwizzleFunctionVec, in_order_reversed_pair_vals)) {
            resAcc[0] = false;
        }
        cl::sycl::vec<${type}, ${size}> reverseOrderReversedPairSwizzleFunctionVec {swizzledVec.template swizzle<${reverse_order_reversed_pair_swiz_indexes}>()};
        if (!check_vector_values<${type}, ${size}>(reverseOrderReversedPairSwizzleFunctionVec, reverse_order_reversed_pair_vals)) {
            resAcc[0] = false;
        }
""")

swizzle_full_test_template = Template(
    """        cl::sycl::vec<${type}, ${size}> ${name}DimTestVec = cl::sycl::vec<${type}, ${size}>(${testVecValues});
        ${type} in_order_vals[] = {${in_order_vals}};
        cl::sycl::vec<${type}, ${size}> inOrderSwizzleFunctionVec {${name}DimTestVec.template swizzle<${in_order_swiz_indexes}>()};
        if (!check_equal_type_bool<cl::sycl::vec<${type}, ${size}>>(inOrderSwizzleFunctionVec)) {
            resAcc[0] = false;
        }
        if (!check_vector_size<${type}, ${size}>(inOrderSwizzleFunctionVec)) {
            resAcc[0] = false;
        }
        if (!check_vector_values<${type}, ${size}>(inOrderSwizzleFunctionVec, in_order_vals)) {
            resAcc[0] = false;
        }
        if (!check_vector_member_functions<${type}, ${size}, ${convert_type}, ${as_type}>
          (inOrderSwizzleFunctionVec, in_order_vals)) {
            resAcc[0] = false;
        }

        ${type} reversed_vals[] = {${reversed_vals}};
        cl::sycl::vec<${type}, ${size}> reverseOrderSwizzleFunctionVec {${name}DimTestVec.template swizzle<${reverse_order_swiz_indexes}>()};
        if (!check_equal_type_bool<cl::sycl::vec<${type}, ${size}>>(reverseOrderSwizzleFunctionVec)) {
            resAcc[0] = false;
        }
        if (!check_vector_size<${type}, ${size}>(reverseOrderSwizzleFunctionVec)) {
            resAcc[0] = false;
        }
        if (!check_vector_values<${type}, ${size}>(reverseOrderSwizzleFunctionVec, reversed_vals)) {
            resAcc[0] = false;
        }
        if (!check_vector_member_functions<${type}, ${size}, ${convert_type}, ${as_type}>
          (reverseOrderSwizzleFunctionVec, reversed_vals)) {
            resAcc[0] = false;
        }

        ${type} in_order_reversed_pair_vals[] = {${in_order_pair_vals}};
        cl::sycl::vec<${type}, ${size}> inOrderReversedPairSwizzleFunctionVec {${name}DimTestVec.template swizzle<${in_order_reversed_pair_swiz_indexes}>()};
        if (!check_equal_type_bool<cl::sycl::vec<${type}, ${size}>>(inOrderReversedPairSwizzleFunctionVec)) {
            resAcc[0] = false;
        }
        if (!check_vector_size<${type}, ${size}>(inOrderReversedPairSwizzleFunctionVec)) {
            resAcc[0] = false;
        }
        if (!check_vector_values<${type}, ${size}>(inOrderReversedPairSwizzleFunctionVec, in_order_reversed_pair_vals)) {
            resAcc[0] = false;
        }
        if (!check_vector_member_functions<${type}, ${size}, ${convert_type}, ${as_type}>
          (inOrderReversedPairSwizzleFunctionVec, in_order_reversed_pair_vals)) {
            resAcc[0] = false;
        }

        ${type} reverse_order_reversed_pair_vals[] = {${reverse_order_pair_vals}};
        cl::sycl::vec<${type}, ${size}> reverseOrderReversedPairSwizzleFunctionVec {${name}DimTestVec.template swizzle<${reverse_order_reversed_pair_swiz_indexes}>()};
        if (!check_equal_type_bool<cl::sycl::vec<${type}, ${size}>>(reverseOrderReversedPairSwizzleFunctionVec)) {
            resAcc[0] = false;
        }
        if (!check_vector_size<${type}, ${size}>(reverseOrderReversedPairSwizzleFunctionVec)) {
            resAcc[0] = false;
        }
        if (!check_vector_values<${type}, ${size}>(reverseOrderReversedPairSwizzleFunctionVec, reverse_order_reversed_pair_vals)) {
            resAcc[0] = false;
        }
        if (!check_vector_member_functions<${type}, ${size}, ${convert_type}, ${as_type}>
          (reverseOrderReversedPairSwizzleFunctionVec,
           reverse_order_reversed_pair_vals)) {
            resAcc[0] = false;
        }
""")


def gen_swizzle_test(type_str, convert_type_str, as_type_str, size):
    string = ''
    if size <= 4:
        for length in range(size, size + 1):
            for index_subset, value_subset in zip(
                    product(
                        Data.swizzle_xyzw_list_dict[size][:size],
                        repeat=length),
                    product(Data.vals_list_dict[size][:size], repeat=length)):
                index_list = []
                val_list = []
                for index, value in zip(index_subset, value_subset):
                    index_list.append(index)
                    val_list.append(value)
                val_list = append_fp_postfix(type_str, val_list)
                index_string = ''.join(index_list)
                test_string = swizzle_template.substitute(
                    name=Data.vec_name_dict[size],
                    indexes=index_string,
                    type=type_str,
                    testVecValues=generate_value_list(type_str, size),
                    in_order_vals=', '.join(val_list),
                    reversed_vals=', '.join(val_list[::-1]),
                    in_order_pair_vals=', '.join(swap_pairs(val_list)),
                    reverse_order_pair_vals=', '.join(
                        swap_pairs(val_list[::-1])),
                    in_order_positions=''.join(
                        Data.swizzle_xyzw_list_dict[size][:size]),
                    size=size,
                    swiz_vals=Data.swizzle_elem_list_dict[size][::-1],
                    convert_type=convert_type_str,
                    as_type=as_type_str)
                if size > 1:
                    test_string += lo_hi_odd_even_template.substitute(
                        type=type_str, size=size)
                test_string += swizzle_elem_template.substitute(
                    type=type_str,
                    size=size,
                    in_order_swiz_indexes=', '.join(
                        Data.swizzle_elem_list_dict[size]),
                    reverse_order_swiz_indexes=', '.join(
                        Data.swizzle_elem_list_dict[size][::-1]),
                    in_order_reversed_pair_swiz_indexes=', '.join(
                        swap_pairs(Data.swizzle_elem_list_dict[size])),
                    reverse_order_reversed_pair_swiz_indexes=', '.join(
                        swap_pairs(Data.swizzle_elem_list_dict[size][::-1])))
                string += wrap_with_kernel(
                    type_str,
                    'KERNEL_' + type_str.replace('cl::sycl::', '').replace(
                        ' ', '') + str(size) + index_string,
                    'vec<' + type_str + ', ' + str(size) + '>.' + index_string,
                    test_string)
        if size == 4:
            for length in range(size, size + 1):
                for index_subset, value_subset in zip(
                        product(
                            Data.swizzle_rgba_list_dict[size][:size],
                            repeat=length),
                        product(
                            Data.vals_list_dict[size][:size], repeat=length)):
                    index_list = []
                    val_list = []
                    for index, value in zip(index_subset, value_subset):
                        index_list.append(index)
                        val_list.append(value)
                    index_string = ''.join(index_list)
                    test_string = swizzle_template.substitute(
                        name=Data.vec_name_dict[size],
                        indexes=index_string,
                        type=type_str,
                        testVecValues=generate_value_list(type_str, size),
                        in_order_vals=', '.join(val_list),
                        reversed_vals=', '.join(val_list[::-1]),
                        in_order_pair_vals=', '.join(swap_pairs(val_list)),
                        reverse_order_pair_vals=', '.join(
                            swap_pairs(val_list[::-1])),
                        in_order_positions=''.join(
                            Data.swizzle_rgba_list_dict[size][:size]),
                        size=size,
                        swiz_vals=Data.swizzle_elem_list_dict[size][::-1],
                        convert_type=convert_type_str,
                        as_type=as_type_str)
                    test_string += lo_hi_odd_even_template.substitute(
                        type=type_str, size=size)
                    test_string += swizzle_elem_template.substitute(
                        type=type_str,
                        size=size,
                        in_order_swiz_indexes=', '.join(
                            Data.swizzle_elem_list_dict[size]),
                        reverse_order_swiz_indexes=', '.join(
                            Data.swizzle_elem_list_dict[size][::-1]),
                        in_order_reversed_pair_swiz_indexes=', '.join(
                            swap_pairs(Data.swizzle_elem_list_dict[size])),
                        reverse_order_reversed_pair_swiz_indexes=', '.join(
                            swap_pairs(Data.swizzle_elem_list_dict[size][::
                                                                         -1])))
                    string += wrap_with_kernel(
                        type_str,
                        'KERNEL_' + type_str.replace('cl::sycl::', '').replace(
                            ' ', '') + str(size) + index_string, 'vec<' +
                        type_str + ', ' + str(size) + '>.' + index_string,
                        test_string)
    else:
        test_string = swizzle_full_test_template.substitute(
            name=Data.vec_name_dict[size],
            type=type_str,
            size=size,
            testVecValues=generate_value_list(type_str, size),
            convert_type=convert_type_str,
            as_type=as_type_str,
            in_order_swiz_indexes=', '.join(Data.swizzle_elem_list_dict[size]),
            reverse_order_swiz_indexes=', '.join(
                Data.swizzle_elem_list_dict[size][::-1]),
            in_order_reversed_pair_swiz_indexes=', '.join(
                swap_pairs(Data.swizzle_elem_list_dict[size])),
            reverse_order_reversed_pair_swiz_indexes=', '.join(
                swap_pairs(Data.swizzle_elem_list_dict[size][::-1])),
            in_order_vals=', '.join(Data.vals_list_dict[size]),
            reversed_vals=', '.join(Data.vals_list_dict[size][::-1]),
            in_order_pair_vals=', '.join(
                swap_pairs(Data.vals_list_dict[size])),
            reverse_order_pair_vals=', '.join(
                swap_pairs(Data.vals_list_dict[size][::-1])))
        string += wrap_with_kernel(
            type_str, 'ELEM_KERNEL_' + type_str.replace(
                'cl::sycl::', '').replace(' ', '') + str(size) +
            ''.join(Data.swizzle_elem_list_dict[size][:size]).replace(
                'cl::sycl::elem::', ''),
            'vec<' + type_str + ', ' + str(size) + '> .swizzle<' +
            ', '.join(Data.swizzle_elem_list_dict[size][:size]) + '>',
            test_string)
    return string


def write_source_file(swizzles, input_file, output_file, type_str):

    with open(input_file, 'r') as source_file:
        source = source_file.read()

    source = replace_string_in_source_string(source,
                                             type_str.replace(
                                                 'cl::sycl::', '').replace(
                                                     ' ', '_'), '$TYPE_NAME')
    source = replace_string_in_source_string(source, swizzles[0],
                                             '$1D_SWIZZLES')
    source = replace_string_in_source_string(source, swizzles[1],
                                             '$2D_SWIZZLES')
    source = replace_string_in_source_string(source, swizzles[2],
                                             '$3D_SWIZZLES')
    source = replace_string_in_source_string(source, swizzles[3],
                                             '$4D_SWIZZLES')
    source = replace_string_in_source_string(source, swizzles[4],
                                             '$8D_SWIZZLES')
    source = replace_string_in_source_string(source, swizzles[5],
                                             '$16D_SWIZZLES')

    with open(
            output_file.strip('.cpp') + '_' + type_str.replace(
                'cl::sycl::', '').replace(' ', '_') + '.cpp', 'w+') as output:
        output.write(source)


def make_tests(input_file, output_file):
    swizzles = [None] * 6

    for base_type in Data.opencl_types:
        for sign in Data.signs:
            if (base_type == 'cl::sycl::cl_float'
                    or base_type == 'cl::sycl::cl_double'
                    or base_type == 'cl::sycl::cl_half') and sign is False:
                continue
            type_str = Data.opencl_type_dict[(sign, base_type)]
            convert_type_str = Data.opencl_type_dict[(not sign, base_type)]
            as_type_str = Data.opencl_type_dict[(not sign, base_type)]
            swizzles[0] = gen_swizzle_test(type_str, convert_type_str,
                                            as_type_str, 1)
            swizzles[1] = gen_swizzle_test(type_str, convert_type_str,
                                            as_type_str, 2)
            swizzles[2] = gen_swizzle_test(type_str, convert_type_str,
                                            as_type_str, 3)
            swizzles[3] = gen_swizzle_test(type_str, convert_type_str,
                                            as_type_str, 4)
            swizzles[4] = gen_swizzle_test(type_str, convert_type_str,
                                            as_type_str, 8)
            swizzles[5] = gen_swizzle_test(type_str, convert_type_str,
                                            as_type_str, 16)
            write_source_file(swizzles, input_file, output_file, type_str)


def main():
    make_tests('vector_swizzles_opencl.template', 'vector_swizzles_opencl.cpp')


if __name__ == '__main__':
    main()
