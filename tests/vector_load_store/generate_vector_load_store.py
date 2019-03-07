# ************************************************************************
#
#   SYCL Conformance Test Suite
#
#   Copyright:	(c) 2018 by Codeplay Software LTD. All Rights Reserved.
#
# ************************************************************************

import sys
from string import Template
sys.path.append('../common/')
from common_python_vec import (Data, append_fp_postfix, make_func_call,
                               wrap_with_test_func, write_source_file,
                               wrap_with_extension_checks)

TEST_NAME = 'LOAD_STORE'

load_store_test_template = Template(
    """        ${type} inputData${type_as_str}${size}[${size}] = {${in_order_vals}};
        ${type} outputData${type_as_str}${size}[${size}] = {${val}};
        ${type} swizzleInputData${type_as_str}${size}[${size}] = {${reverse_order_vals}};
        ${type} swizzleOutputData${type_as_str}${size}[${size}] = {${val}};
        {
          cl::sycl::buffer<${type}, 1> inBuffer${type_as_str}${size}(inputData${type_as_str}${size}, cl::sycl::range<1>(${size}));
          cl::sycl::buffer<${type}, 1> outBuffer${type_as_str}${size}(outputData${type_as_str}${size}, cl::sycl::range<1>(${size}));
          cl::sycl::buffer<${type}, 1> swizzleInBuffer${type_as_str}${size}(swizzleInputData${type_as_str}${size}, cl::sycl::range<1>(${size}));
          cl::sycl::buffer<${type}, 1> swizzleOutBuffer${type_as_str}${size}(swizzleOutputData${type_as_str}${size}, cl::sycl::range<1>(${size}));

          testQueue.submit([&](cl::sycl::handler &cgh) {
            auto inPtr${type_as_str}${size} = inBuffer${type_as_str}${size}.get_access<cl::sycl::access::mode::read_write>(cgh);
            auto outPtr${type_as_str}${size} = outBuffer${type_as_str}${size}.get_access<cl::sycl::access::mode::read_write>(cgh);

            auto swizzleInPtr${type_as_str}${size} = swizzleInBuffer${type_as_str}${size}.get_access<cl::sycl::access::mode::read_write>(cgh);
            auto swizzleOutPtr${type_as_str}${size} = swizzleOutBuffer${type_as_str}${size}.get_access<cl::sycl::access::mode::read_write>(cgh);

            cgh.single_task<class ${kernelName}>([=]() {
              auto testVec${type_as_str}${size} = cl::sycl::vec<${type}, ${size}>(${val});
              testVec${type_as_str}${size}.load(0, inPtr${type_as_str}${size});
              testVec${type_as_str}${size}.store(0, outPtr${type_as_str}${size});

              auto cleanVec${type_as_str}${size} = cl::sycl::vec<${type}, ${size}>(${val});
              cl::sycl::vec<${type}, ${size}> swizzledVec {cleanVec${type_as_str}${size}.template swizzle<${swizVals}>()};
              swizzledVec.load(0, swizzleInPtr${type_as_str}${size});
              swizzledVec.store(0, swizzleOutPtr${type_as_str}${size});
            });
          });

        }
        check_array_equality<${type}, ${size}>(log, inputData${type_as_str}${size}, outputData${type_as_str}${size});
        check_array_equality<${type}, ${size}>(log, swizzleInputData${type_as_str}${size}, swizzleOutputData${type_as_str}${size});

        testQueue.wait_and_throw();
      """)


def gen_kernel_name(type_str, size):
    return 'KERNEL_load_store_' + type_str.replace('cl::sycl::', '').replace(
        ' ', '') + str(size)


def gen_load_store_test(type_str, size):
    no_whitespace_type_str = type_str.replace(' ', '').replace(
        'cl::sycl::', '')
    test_string = load_store_test_template.substitute(
        type=type_str,
        type_as_str=no_whitespace_type_str,
        size=size,
        val=Data.value_default_dict[type_str],
        in_order_vals=', '.join(
            append_fp_postfix(type_str, Data.vals_list_dict[size])),
        reverse_order_vals=', '.join(
            append_fp_postfix(type_str, Data.vals_list_dict[size][::-1])),
        kernelName=gen_kernel_name(type_str, size),
        swizVals=', '.join(Data.swizzle_elem_list_dict[size]))
    return wrap_with_test_func(TEST_NAME, type_str,
                               wrap_with_extension_checks(
                                   type_str, test_string), str(size))


def make_tests(input_file, output_file):

    # Test with type_str='char'
    test_string = ''
    func_calls = ''
    for size in Data.standard_sizes:
        test_string += gen_load_store_test('char', size)
        func_calls += make_func_call(TEST_NAME, 'char', str(size))
    write_source_file(test_string, func_calls, TEST_NAME, input_file,
                      output_file, 'char')

    for base_type in Data.standard_types:
        for sign in Data.signs:
            if (base_type == 'float' or base_type == 'double'
                    or base_type == 'cl::sycl::half') and sign is False:
                continue
            type_str = Data.standard_type_dict[(sign, base_type)]
            test_string = ''
            func_calls = ''
            for size in Data.standard_sizes:
                test_string += gen_load_store_test(type_str, size)
                func_calls += make_func_call(TEST_NAME, type_str, str(size))
            write_source_file(test_string, func_calls, TEST_NAME, input_file,
                              output_file, type_str)

    for base_type in Data.opencl_types:
        for sign in Data.signs:
            if (base_type == 'cl::sycl::cl_float'
                    or base_type == 'cl::sycl::cl_double'
                    or base_type == 'cl::sycl::cl_half') and sign is False:
                continue
            type_str = Data.opencl_type_dict[(sign, base_type)]
            test_string = ''
            func_calls = ''
            for size in Data.standard_sizes:
                test_string += gen_load_store_test(type_str, size)
                func_calls += make_func_call(TEST_NAME, type_str, str(size))
            write_source_file(test_string, func_calls, TEST_NAME, input_file,
                              output_file, type_str)


def main():
    make_tests('../common/vector.template', 'vector_load_store.cpp')


if __name__ == '__main__':
    main()
