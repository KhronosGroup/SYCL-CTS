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
from string import Template


x = ['x']
xy = ['x', 'y']
xyz = ['x', 'y', 'z']
xyzw = ['x', 'y', 'z', 'w']
r = ['r']
rg = ['r', 'g']
rgb = ['r', 'g', 'b']
rgba = ['r', 'g', 'b', 'a']
swizzle_vals = {'x': '0', 'y': '1', 'z': '2', 'w': '3',
                'r': '0', 'g': '1', 'b': '2', 'a': '3'}

all_vector_checks_template = Template(
    """  check_equal_type<vec<${type}, ${size}>>(log, swizzledVec,
    "Testing swizzled constructor .${conname}()");
  check_vector_size<${type}, ${size}>(log, swizzledVec);
  check_vector_values<${type}, ${size}>(log, swizzledVec, vals);
  check_vector_member_functions<${type}, ${size}, ${convert_type}, ${as_type}>
    (log, swizzledVec, vals);\n""")

lo_hi_odd_even_template = Template(
    """  check_lo_hi_odd_even<${type}, ${size}>(log, swizzledVec, vals);\n""")

vals_template = Template("""  ${type} vals[] = {${vals}};
  ${type} in_order_vals[] = {${in_order_vals}};
  ${type} reversed_vals[] = {${reversed_vals}};\n""")

swizzle_function_template = Template("""  auto inOrderSwizzledVec = ${dims}DimTestVec.${in_order_positions}();
  vec<${type}, ${size}> swizzleFunctionVec = swizzledVec.template swizzle<${swiz_vals}>();
  check_vector_values<${type}, ${size}>(log, swizzleFunctionVec, reversed_vals);\n""")

swizzle_template = Template(
    """{ auto swizzledVec = ${dims}DimTestVec.${pos1}${pos2}${pos3}${pos4}();\n""")


def print_vec(
        dims,
        pos1,
        pos2='',
        pos3='',
        pos4=''):
    return swizzle_template.substitute(
        dims=dims,
        pos1=pos1,
        pos2=pos2,
        pos3=pos3,
        pos4=pos4)


def gen_one_dim_swizzles(type_str, convert_type_str, as_type_str, dims=1):
    string = ''
    for pos1 in x:
        string += print_vec('One', pos1)
        string += vals_template.substitute(type=type_str,
                                           vals=swizzle_vals[pos1],
                                           in_order_vals=Data.vals_dict[dims],
                                           reversed_vals=Data.reverse_vals_dict[dims])
        string += swizzle_function_template.substitute(
            dims='One',
            in_order_positions='x',
            type=type_str,
            size=dims,
            swiz_vals=Data.reverse_swizzle_index_dict[dims])
        string += all_vector_checks_template.substitute(
            type=type_str,
            convert_type=convert_type_str,
            as_type=as_type_str,
            size=1,
            conname=pos1)
        string += '}\n'
    for pos1 in r:
        string += print_vec('One', pos1)
        string += vals_template.substitute(type=type_str,
                                           vals=swizzle_vals[pos1],
                                           in_order_vals=Data.vals_dict[dims],
                                           reversed_vals=Data.reverse_vals_dict[dims])
        string += swizzle_function_template.substitute(
            dims='One',
            in_order_positions='r',
            type=type_str,
            size=dims,
            swiz_vals=Data.reverse_swizzle_index_dict[dims])
        string += all_vector_checks_template.substitute(
            type=type_str,
            convert_type=convert_type_str,
            as_type=as_type_str,
            size=1,
            conname=pos1)
        string += '}\n'
    return string


def gen_two_dim_swizzles(type_str, convert_type_str, as_type_str, dims=2):
    string = ''
    for pos1 in xy:
        for pos2 in xy:
            string += print_vec('Two', pos1, pos2)
            string += vals_template.substitute(type=type_str,
                                               vals=swizzle_vals[pos1] +
                                               ', ' +
                                               swizzle_vals[pos2],
                                               in_order_vals=Data.vals_dict[dims],
                                               reversed_vals=Data.reverse_vals_dict[dims])
            string += swizzle_function_template.substitute(
                dims='Two',
                in_order_positions='xy',
                type=type_str,
                size=dims,
                swiz_vals=Data.reverse_swizzle_index_dict[dims])
            string += all_vector_checks_template.substitute(
                type=type_str,
                convert_type=convert_type_str,
                as_type=as_type_str,
                size=2,
                conname=pos1 + pos2)
            string += lo_hi_odd_even_template.substitute(
                type=type_str,
                size=2)
            string += '}\n'
    for pos1 in rg:
        for pos2 in rg:
            string += print_vec('Two', pos1, pos2)
            string += vals_template.substitute(type=type_str,
                                               vals=swizzle_vals[pos1] +
                                               ', ' +
                                               swizzle_vals[pos2],
                                               in_order_vals=Data.vals_dict[dims],
                                               reversed_vals=Data.reverse_vals_dict[dims])
            string += swizzle_function_template.substitute(
                dims='Two',
                in_order_positions='rg',
                type=type_str,
                size=dims,
                swiz_vals=Data.reverse_swizzle_index_dict[dims])
            string += all_vector_checks_template.substitute(
                type=type_str,
                convert_type=convert_type_str,
                as_type=as_type_str,
                size=2,
                conname=pos1 + pos2)
            string += lo_hi_odd_even_template.substitute(
                type=type_str,
                size=2)
            string += '}\n'
    return string


def gen_three_dim_swizzles(type_str, convert_type_str, as_type_str, dims=3):
    string = ''
    for pos1 in xyz:
        for pos2 in xyz:
            for pos3 in xyz:
                string += print_vec('Three', pos1, pos2, pos3)
                string += vals_template.substitute(type=type_str,
                                                   vals=swizzle_vals[pos1] +
                                                   ', ' +
                                                   swizzle_vals[pos2] +
                                                   ', ' +
                                                   swizzle_vals[pos3],
                                                   in_order_vals=Data.vals_dict[dims],
                                                   reversed_vals=Data.reverse_vals_dict[dims])
                string += swizzle_function_template.substitute(
                    dims='Three',
                    in_order_positions='xyz',
                    type=type_str,
                    size=dims,
                    swiz_vals=Data.reverse_swizzle_index_dict[dims])
                string += all_vector_checks_template.substitute(
                    type=type_str,
                    convert_type=convert_type_str,
                    as_type=as_type_str,
                    size=3,
                    conname=pos1 + pos2 + pos3)
                string += lo_hi_odd_even_template.substitute(
                    type=type_str,
                    size=3)
                string += '}\n'
    for pos1 in rgb:
        for pos2 in rgb:
            for pos3 in rgb:
                string += print_vec('Three', pos1, pos2, pos3)
                string += vals_template.substitute(type=type_str,
                                                   vals=swizzle_vals[pos1] +
                                                   ', ' +
                                                   swizzle_vals[pos2] +
                                                   ', ' +
                                                   swizzle_vals[pos3],
                                                   in_order_vals=Data.vals_dict[dims],
                                                   reversed_vals=Data.reverse_vals_dict[dims])
                string += swizzle_function_template.substitute(
                    dims='Three',
                    in_order_positions='rgb',
                    type=type_str,
                    size=dims,
                    swiz_vals=Data.reverse_swizzle_index_dict[dims])
                string += all_vector_checks_template.substitute(
                    type=type_str,
                    convert_type=convert_type_str,
                    as_type=as_type_str,
                    size=3,
                    conname=pos1 + pos2 + pos3)
                string += lo_hi_odd_even_template.substitute(
                    type=type_str,
                    size=3)
                string += '}\n'
    return string


def gen_four_dim_swizzles(type_str, convert_type_str, as_type_str, dims=4):
    string = ''
    for pos1 in xyzw:
        for pos2 in xyzw:
            for pos3 in xyzw:
                for pos4 in xyzw:
                    string += print_vec('Four', pos1, pos2, pos3, pos4)
                    string += vals_template.substitute(
                        type=type_str,
                        vals=swizzle_vals[pos1] +
                        ', ' +
                        swizzle_vals[pos2] +
                        ', ' +
                        swizzle_vals[pos3] +
                        ', ' +
                        swizzle_vals[pos4],
                        in_order_vals=Data.vals_dict[dims],
                        reversed_vals=Data.reverse_vals_dict[dims])
                    string += swizzle_function_template.substitute(
                        dims='Four',
                        in_order_positions='xyzw',
                        type=type_str,
                        size=dims,
                        swiz_vals=Data.reverse_swizzle_index_dict[dims])
                    string += all_vector_checks_template.substitute(
                        type=type_str,
                        convert_type=convert_type_str,
                        as_type=as_type_str,
                        size=4,
                        conname=pos1 + pos2 + pos3 + pos4)
                    string += lo_hi_odd_even_template.substitute(
                        type=type_str,
                        size=4)
                    string += '}\n'
    for pos1 in rgba:
        for pos2 in rgba:
            for pos3 in rgba:
                for pos4 in rgba:
                    string += print_vec('Four', pos1, pos2, pos3, pos4)
                    string += vals_template.substitute(
                        type=type_str,
                        vals=swizzle_vals[pos1] +
                        ', ' +
                        swizzle_vals[pos2] +
                        ', ' +
                        swizzle_vals[pos3] +
                        ', ' +
                        swizzle_vals[pos4],
                        in_order_vals=Data.vals_dict[dims],
                        reversed_vals=Data.reverse_vals_dict[dims])
                    string += swizzle_function_template.substitute(
                        dims='Four',
                        in_order_positions='rgba',
                        type=type_str,
                        size=dims,
                        swiz_vals=Data.reverse_swizzle_index_dict[dims])
                    string += all_vector_checks_template.substitute(
                        type=type_str,
                        convert_type=convert_type_str,
                        as_type=as_type_str,
                        size=4,
                        conname=pos1 + pos2 + pos3 + pos4)
                    string += lo_hi_odd_even_template.substitute(
                        type=type_str,
                        size=4)
                    string += '}\n'
    return string


def make_tests(input_file, output_file):
    one_dim_swizzles = ''
    two_dim_swizzles = ''
    three_dim_swizzles = ''
    four_dim_swizzles = ''
    for base_type in Data.standard_types:
        for sign in Data.signs:
            if (base_type == 'float' or base_type == 'double' or base_type == 'half') and sign is False:
                continue
            type_str = Data.standard_type_dict[(sign, base_type)]
            convert_type_str = Data.standard_type_dict[(not sign, base_type)]
            as_type_str = Data.standard_type_dict[(not sign, base_type)]
            one_dim_swizzles += gen_one_dim_swizzles(
                type_str, convert_type_str, as_type_str)
            two_dim_swizzles += gen_two_dim_swizzles(
                type_str, convert_type_str, as_type_str)
            three_dim_swizzles += gen_three_dim_swizzles(
                type_str, convert_type_str, as_type_str)
            four_dim_swizzles += gen_four_dim_swizzles(
                type_str, convert_type_str, as_type_str)

    with open(input_file, 'r') as source_file:
        source = source_file.read()

    source = replace_string_in_source_string(source, one_dim_swizzles, '$1D_SWIZZLES')
    source = replace_string_in_source_string(source, two_dim_swizzles, '$2D_SWIZZLES')
    source = replace_string_in_source_string(source, three_dim_swizzles, '$3D_SWIZZLES')
    source = replace_string_in_source_string(source, four_dim_swizzles, '$4D_SWIZZLES')

    with open(output_file, 'w+') as output:
        output.write(source)


def main():
    make_tests('vector_swizzles.template', 'vector_swizzles.cpp')


if __name__ == '__main__':
    main()
