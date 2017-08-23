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

vec_template = Template(
    """auto inputVec = vec<${type}, ${size}>(${vals});\n""")

vals_template = Template("""  ${type} vals[] = {${vals}};
  ${type} reversed_vals[] = {${reversed_vals}};\n""")

vector_member_function_template = Template(
    """  check_vector_member_functions<${type}, ${size}, ${convertType}, ${asType}>(log, inputVec, vals);\n""")

lo_hi_odd_even_template = Template(
    """  check_lo_hi_odd_even<${type}, ${size}>(log, inputVec, vals);\n""")

swizzle_function_template = Template(
    """  vec<${type}, ${size}> swizzledVec = inputVec.template swizzle<${swizVals}>();
  check_vector_values<${type}, ${size}>(log, swizzledVec, reversed_vals);\n""")


def gen_host_checks(base_type, signed, size):
    """Uses the above string templates to generate tests for each vec api function except load and store.
    Load and store are handled separately.
    lo() hi() odd() and even() are handled with a separate function and template to other api functions
    as they can only be performed on vectors of size 2 or greater."""
    type_str = Data.standard_type_dict[(signed, base_type)]
    convert_type_str = Data.standard_type_dict[(not signed, base_type)]
    as_type_str = Data.standard_type_dict[(not signed, base_type)]
    string = '{ '
    string += vec_template.substitute(
        type=type_str, size=size, vals=Data.vals_dict[size])
    string += vals_template.substitute(type=type_str,
                                       vals=Data.vals_dict[size],
                                       reversed_vals=Data.reverse_vals_dict[size])
    string += vector_member_function_template.substitute(
        type=type_str, size=size, convertType=convert_type_str, asType=as_type_str)
    if size != 1:
        string += lo_hi_odd_even_template.substitute(type=type_str, size=size)
    string += swizzle_function_template.substitute(
        type=type_str, size=size, swizVals=Data.reverse_swizzle_index_dict[size])
    string += '}\n'
    return string


def make_tests(input_file, output_file):
    host_api_checks = ''

    for base_type in Data.standard_types:
        for sign in Data.signs:
            if (base_type == 'float' or base_type == 'double' or base_type == 'half') and sign is False:
                continue
            for size in Data.standard_sizes:
                host_api_checks += gen_host_checks(base_type, sign, size)

    with open(input_file, 'r') as source_file:
        source = source_file.read()

    source = replace_string_in_source_string(source, host_api_checks, '$HOST_API_CHECKS')

    with open(output_file, 'w+') as output:
        output.write(source)


def main():
    make_tests('vector_api.template', 'vector_api.cpp')


if __name__ == '__main__':
    main()
