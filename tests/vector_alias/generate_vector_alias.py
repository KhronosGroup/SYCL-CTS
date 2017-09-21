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
from string import Template

alias_test_template = Template("""
        auto aliasVec = ${aliasVecName}();
        resAcc[0] = check_equal_type_bool<cl::sycl::vec<${typeName}, ${size}>>(aliasVec);
""")


def gen_alias_test(type_str, size):
    alias_vec_name = Data.alias_dict[type_str] + str(size)
    test_string = alias_test_template.substitute(aliasVecName=alias_vec_name,
                                                 typeName=type_str,
                                                 size=size)
    string = wrap_with_kernel(
        type_str,
        'KERNEL_' +
        alias_vec_name.replace(
            'cl::sycl::',
            ''),
        'Alias vector test: ' +
        alias_vec_name,
        test_string)
    return string


def make_tests(input_file, output_file):
    sycl_alias = ''
    opencl_alias = ''

    # Test with type_str='char'
    for size in [2, 3, 4, 8, 16]:
        sycl_alias += gen_alias_test('char', size)

    for base_type in Data.standard_types:
        for sign in Data.signs:
            if (base_type == 'float' or base_type ==
                    'double' or base_type == 'cl::sycl::half') and sign is False:
                continue
            type_str = Data.standard_type_dict[(sign, base_type)]
            for size in [2, 3, 4, 8, 16]:
                sycl_alias += gen_alias_test(type_str, size)

    for base_type in Data.opencl_types:
        for sign in Data.signs:
            if (base_type == 'cl::sycl::cl_float' or base_type ==
                    'cl::sycl::cl_double' or base_type == 'cl::sycl::cl_half') and sign is False:
                continue
            type_str = Data.opencl_type_dict[(sign, base_type)]
            for size in [2, 3, 4, 8, 16]:
                opencl_alias += gen_alias_test(type_str, size)

    with open(input_file, 'r') as source_file:
        source = source_file.read()

    source = replace_string_in_source_string(
        source, sycl_alias, '$SYCL_ALIAS')
    source = replace_string_in_source_string(
        source, opencl_alias, '$OPENCL_ALIAS')

    with open(output_file, 'w+') as output:
        output.write(source)


def main():
    make_tests('vector_alias.template', 'vector_alias.cpp')


if __name__ == '__main__':
    main()
