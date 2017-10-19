# ************************************************************************
#
#   SYCL Conformance Test Suite
#
#   Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
#
# ************************************************************************

import sys
sys.path.append('../common/')
from common_python_vec import Data, replace_string_in_source_string, append_fp_postfix, wrap_with_half_check
from string import Template


load_store_buffers_template = Template("""
  ${type} inputData${type_as_str}${size}[${size}] = {${in_order_vals}};
  ${type} outputData${type_as_str}${size}[${size}] = {${val}};
  ${type} swizzleInputData${type_as_str}${size}[${size}] = {${reverse_order_vals}};
  ${type} swizzleOutputData${type_as_str}${size}[${size}] = {${val}};
  {
    cl::sycl::buffer<${type}, 1> inBuffer${type_as_str}${size}(inputData${type_as_str}${size}, cl::sycl::range<1>(${size}));
    cl::sycl::buffer<${type}, 1> outBuffer${type_as_str}${size}(outputData${type_as_str}${size}, cl::sycl::range<1>(${size}));
    cl::sycl::buffer<${type}, 1> swizzleInBuffer${type_as_str}${size}(swizzleInputData${type_as_str}${size}, cl::sycl::range<1>(${size}));
    cl::sycl::buffer<${type}, 1> swizzleOutBuffer${type_as_str}${size}(swizzleOutputData${type_as_str}${size}, cl::sycl::range<1>(${size}));

""")

load_store_accessors_template = Template("""
    testQueue.submit([&](cl::sycl::handler &cgh) {
      auto inPtr${type_as_str}${size} = inBuffer${type_as_str}${size}.get_access<cl::sycl::access::mode::read>(cgh);
      auto outPtr${type_as_str}${size} = outBuffer${type_as_str}${size}.get_access<cl::sycl::access::mode::write>(cgh);

      auto swizzleInPtr${type_as_str}${size} = swizzleInBuffer${type_as_str}${size}.get_access<cl::sycl::access::mode::read>(cgh);
      auto swizzleOutPtr${type_as_str}${size} = swizzleOutBuffer${type_as_str}${size}.get_access<cl::sycl::access::mode::write>(cgh);

""")

load_store_template = Template("""
      cgh.single_task<class ${kernelName}>([=]() {
        auto testVec${type_as_str}${size} = cl::sycl::vec<${type}, ${size}>(${val});
        testVec${type_as_str}${size}.load(sizeof(${type}) * ${size}, inPtr${type_as_str}${size});
        testVec${type_as_str}${size}.store(sizeof(${type}) * ${size}, outPtr${type_as_str}${size});

        auto cleanVec${type_as_str}${size} = cl::sycl::vec<${type}, ${size}>(${val});
        auto swizzledVec = cleanVec${type_as_str}${size}.template swizzle<${swizVals}>();
        swizzledVec.load(sizeof(${type}) * ${size}, swizzleInPtr${type_as_str}${size});
        swizzledVec.store(sizeof(${type}) * ${size}, swizzleOutPtr${type_as_str}${size});
      });
    });

""")

load_store_check_template = Template("""
  }
  check_array_equality<${type}, ${size}>(log, inputData${type_as_str}${size}, outputData${type_as_str}${size});
  ${type} reversedInputData${type_as_str}${size}[${size}] = {${reverse_order_vals}};
  check_array_equality<${type}, ${size}>(log, reversedInputData${type_as_str}${size}, swizzleOutputData${type_as_str}${size});

  testQueue.wait_and_throw();

""")


def gen_load_store_buffers(type_str, size):
    no_whitespace_type_str = type_str.replace(
        ' ', '').replace(
        'cl::sycl::', '')
    return load_store_buffers_template.substitute(
        type=type_str,
        type_as_str=no_whitespace_type_str,
        size=size,
        val=Data.value_default_dict[type_str],
        in_order_vals=', '.join(append_fp_postfix(type_str, Data.vals_list_dict[size])),
        reverse_order_vals=', '.join(append_fp_postfix(type_str, Data.vals_list_dict[size][::-1])))


def gen_load_store_accessors(type_str, size):
    no_whitespace_type_str = type_str.replace(
        ' ', '').replace(
        'cl::sycl::', '')
    return load_store_accessors_template.substitute(
        type_as_str=no_whitespace_type_str, size=size)


def gen_load_stores(type_str, size):
    no_whitespace_type_str = type_str.replace(
        ' ', '').replace(
        'cl::sycl::', '')
    return load_store_template.substitute(
        type=type_str,
        size=size,
        kernelName='KERNEL_' +
        type_str.replace(
            'cl::sycl::',
            '').replace(
            ' ',
            '') +
        str(size),
        val=Data.value_default_dict[type_str],
        type_as_str=no_whitespace_type_str,
        swizVals=', '.join(
            Data.swizzle_elem_list_dict[size]))


def gen_load_store_checks(type_str, size):
    no_whitespace_type_str = type_str.replace(
        ' ', '').replace(
        'cl::sycl::', '')
    return load_store_check_template.substitute(
        type=type_str,
        size=size,
        type_as_str=no_whitespace_type_str,
        reverse_order_vals=', '.join(append_fp_postfix(type_str, Data.vals_list_dict[size][::-1])))


def make_tests(input_file, output_file):
    string = ''

    # Test with type_str='char'
    for size in Data.standard_sizes:
        test_string = gen_load_store_buffers(
            'char', size)
        test_string += gen_load_store_accessors(
            'char', size)
        test_string += gen_load_stores('char', size)
        test_string += gen_load_store_checks(
            'char', size)
        string += wrap_with_half_check('char', test_string)

    for base_type in Data.standard_types:
        for sign in Data.signs:
            if (base_type == 'float' or base_type ==
                    'double' or base_type == 'cl::sycl::half') and sign is False:
                continue
            type_str = Data.standard_type_dict[(sign, base_type)]
            for size in Data.standard_sizes:
                test_string = gen_load_store_buffers(
                    type_str, size)
                test_string += gen_load_store_accessors(
                    type_str, size)
                test_string += gen_load_stores(type_str, size)
                test_string += gen_load_store_checks(
                    type_str, size)
                string += wrap_with_half_check(type_str, test_string)

    for base_type in Data.opencl_types:
        for sign in Data.signs:
            if (base_type == 'cl::sycl::cl_float' or base_type ==
                    'cl::sycl::cl_double' or base_type == 'cl::sycl::cl_half') and sign is False:
                continue
            type_str = Data.opencl_type_dict[(sign, base_type)]
            for size in Data.standard_sizes:
                test_string = gen_load_store_buffers(
                    type_str, size)
                test_string += gen_load_store_accessors(
                    type_str, size)
                test_string += gen_load_stores(type_str, size)
                test_string += gen_load_store_checks(
                    type_str, size)
                string += wrap_with_half_check(type_str, test_string)

    with open(input_file, 'r') as source_file:
        source = source_file.read()

    source = replace_string_in_source_string(
        source,
        string,
        '$LOAD_STORE_TESTS')

    with open(output_file, 'w+') as output:
        output.write(source)


def main():
    make_tests('vector_load_store.template', 'vector_load_store.cpp')


if __name__ == '__main__':
    main()
