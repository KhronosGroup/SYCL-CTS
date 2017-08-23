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


load_store_buffers_template = Template("""${type} inputData${type_as_str}${size}[${size}] = {${in_order_vals}};
${type} outputData${type_as_str}${size}[${size}] = {0};
${type} swizzleInputData${type_as_str}${size}[${size}] = {${reverse_order_vals}};
${type} swizzleOutputData${type_as_str}${size}[${size}] = {0};

buffer<${type}, 1> inBuffer${type_as_str}${size}(inputData${type_as_str}${size}, range<1>(${size}));
buffer<${type}, 1> outBuffer${type_as_str}${size}(outputData${type_as_str}${size}, range<1>(${size}));
buffer<${type}, 1> swizzleInBuffer${type_as_str}${size}(swizzleInputData${type_as_str}${size}, range<1>(${size}));
buffer<${type}, 1> swizzleOutBuffer${type_as_str}${size}(swizzleOutputData${type_as_str}${size}, range<1>(${size}));\n\n""")

load_store_accessors_template = Template(
    """auto inPtr${type_as_str}${size} = inBuffer${type_as_str}${size}.get_access<access::mode::read>(cgh);
auto outPtr${type_as_str}${size} = outBuffer${type_as_str}${size}.get_access<access::mode::write>(cgh);

auto swizzleInPtr${type_as_str}${size} = swizzleInBuffer${type_as_str}${size}.get_access<access::mode::read>(cgh);
auto swizzleOutPtr${type_as_str}${size} = swizzleOutBuffer${type_as_str}${size}.get_access<access::mode::write>(cgh);\n\n""")

load_store_template = Template("""auto testVec${type_as_str}${size} = vec<${type}, ${size}>(0);
testVec${type_as_str}${size}.load(sizeof(${type}) * ${size}, inPtr${type_as_str}${size});
testVec${type_as_str}${size}.store(sizeof(${type}) * ${size}, outPtr${type_as_str}${size});

auto cleanVec${type_as_str}${size} = vec<${type}, ${size}>(0);
auto swizzledVec = cleanVec${type_as_str}${size}.template swizzle<${swizVals}>();
swizzledVec.load(sizeof(${type}) * ${size}, swizzleInPtr${type_as_str}${size});
swizzledVec.store(sizeof(${type}) * ${size}, swizzleOutPtr${type_as_str}${size});\n\n""")

load_store_check_template = Template(
    """check_array_equality<${type}, ${size}>(log, inputData${type_as_str}${size}, outputData${type_as_str}${size});
${type} reversedInputData${type_as_str}${size}[${size}] = {${reverse_order_vals}};
check_array_equality<${type}, ${size}>(log, reversedInputData${type_as_str}${size}, swizzleOutputData${type_as_str}${size});\n\n""")


def gen_load_store_buffers(base_type, signed, size):
    type_str = Data.standard_type_dict[(signed, base_type)]
    no_whitespace_type_str = type_str.replace(' ', '')
    string = load_store_buffers_template.substitute(
        type=type_str,
        type_as_str=no_whitespace_type_str,
        size=size,
        in_order_vals=Data.vals_dict[size],
        reverse_order_vals=Data.reverse_vals_dict[size])
    return string


def gen_load_store_accessors(base_type, signed, size):
    type_str = Data.standard_type_dict[(signed, base_type)]
    no_whitespace_type_str = type_str.replace(' ', '')
    string = load_store_accessors_template.substitute(
        type_as_str=no_whitespace_type_str, size=size)
    return string


def gen_load_stores(base_type, signed, size):
    type_str = Data.standard_type_dict[(signed, base_type)]
    no_whitespace_type_str = type_str.replace(' ', '')
    string = load_store_template.substitute(
        type=type_str,
        size=size,
        type_as_str=no_whitespace_type_str,
        swizVals=Data.swizzle_index_dict[size])
    return string


def gen_load_store_checks(base_type, signed, size):
    type_str = Data.standard_type_dict[(signed, base_type)]
    no_whitespace_type_str = type_str.replace(' ', '')
    string = load_store_check_template.substitute(
        type=type_str,
        size=size,
        type_as_str=no_whitespace_type_str,
        reverse_order_vals=Data.reverse_vals_dict[size])
    return string


def make_tests(input_file, output_file):
    load_store_buffers = ''
    load_store_accessors = ''
    load_stores = ''
    load_store_checks = ''

    for base_type in Data.standard_types:
        for sign in Data.signs:
            if (base_type == 'float' or base_type == 'double' or base_type == 'half') and sign is False:
                continue
            for size in Data.standard_sizes:
                load_store_buffers += gen_load_store_buffers(
                    base_type, sign, size)
                load_store_accessors += gen_load_store_accessors(
                    base_type, sign, size)
                load_stores += gen_load_stores(base_type, sign, size)
                load_store_checks += gen_load_store_checks(
                    base_type, sign, size)

    with open(input_file, 'r') as source_file:
        source = source_file.read()

    source = replace_string_in_source_string(
        source,
        load_store_buffers,
        '$LOAD_STORE_BUFFERS')
    source = replace_string_in_source_string(
        source,
        load_store_accessors,
        '$LOAD_STORE_ACCESSORS')
    source = replace_string_in_source_string(source, load_stores, '$LOAD_STORES')
    source = replace_string_in_source_string(source, load_store_checks, '$LOAD_STORE_CHECKS')

    with open(output_file, 'w+') as output:
        output.write(source)


def main():
    make_tests('vector_load_store.template', 'vector_load_store.cpp')


if __name__ == '__main__':
    main()
