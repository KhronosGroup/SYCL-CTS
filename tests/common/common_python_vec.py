# ************************************************************************
#
#   SYCL Conformance Test Suite
#
#   Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
#
# ************************************************************************

from collections import defaultdict
from string import Template


class Data:
    signs = [True, False]
    standard_sizes = [1, 2, 3, 4, 8, 16]
    standard_types = [
        'char', 'short', 'int', 'long', 'long long', 'float', 'double',
        'cl::sycl::half'
    ]
    standard_type_dict = {
        (False, 'char'): 'unsigned char',
        (True, 'char'): 'signed char',
        (False, 'short'): 'unsigned short',
        (True, 'short'): 'short',
        (False, 'int'): 'unsigned int',
        (True, 'int'): 'int',
        (False, 'long'): 'unsigned long',
        (True, 'long'): 'long',
        (False, 'long long'): 'unsigned long long',
        (True, 'long long'): 'long long',
        (False, 'float'): 'float',
        (True, 'float'): 'float',
        (False, 'double'): 'double',
        (True, 'double'): 'double',
        (False, 'cl::sycl::half'): 'cl::sycl::half',
        (True, 'cl::sycl::half'): 'cl::sycl::half'
    }
    opencl_types = [
        'cl::sycl::cl_char', 'cl::sycl::cl_short', 'cl::sycl::cl_int',
        'cl::sycl::cl_long', 'cl::sycl::cl_float', 'cl::sycl::cl_double',
        'cl::sycl::cl_half'
    ]
    opencl_type_dict = {
        (False, 'cl::sycl::cl_char'): 'cl::sycl::cl_uchar',
        (True, 'cl::sycl::cl_char'): 'cl::sycl::cl_char',
        (False, 'cl::sycl::cl_short'): 'cl::sycl::cl_ushort',
        (True, 'cl::sycl::cl_short'): 'cl::sycl::cl_short',
        (False, 'cl::sycl::cl_int'): 'cl::sycl::cl_uint',
        (True, 'cl::sycl::cl_int'): 'cl::sycl::cl_int',
        (False, 'cl::sycl::cl_long'): 'cl::sycl::cl_ulong',
        (True, 'cl::sycl::cl_long'): 'cl::sycl::cl_long',
        (False, 'cl::sycl::cl_float'): 'cl::sycl::cl_float',
        (True, 'cl::sycl::cl_float'): 'cl::sycl::cl_float',
        (False, 'cl::sycl::cl_double'): 'cl::sycl::cl_double',
        (True, 'cl::sycl::cl_double'): 'cl::sycl::cl_double',
        (False, 'cl::sycl::cl_half'): 'cl::sycl::cl_half',
        (True, 'cl::sycl::cl_half'): 'cl::sycl::cl_half'
    }
    alias_dict = {
        'char': 'cl::sycl::char',
        'signed char': 'cl::sycl::schar',
        'unsigned char': 'cl::sycl::uchar',
        'short': 'cl::sycl::short',
        'unsigned short': 'cl::sycl::ushort',
        'int': 'cl::sycl::int',
        'unsigned int': 'cl::sycl::uint',
        'long': 'cl::sycl::long',
        'unsigned long': 'cl::sycl::ulong',
        'long long': 'cl::sycl::longlong',
        'unsigned long long': 'cl::sycl::ulonglong',
        'float': 'cl::sycl::float',
        'double': 'cl::sycl::double',
        'cl::sycl::half': 'cl::sycl::half',
        'cl::sycl::cl_char': 'cl::sycl::cl_char',
        'cl::sycl::cl_uchar': 'cl::sycl::cl_uchar',
        'cl::sycl::cl_short': 'cl::sycl::cl_short',
        'cl::sycl::cl_ushort': 'cl::sycl::cl_ushort',
        'cl::sycl::cl_int': 'cl::sycl::cl_int',
        'cl::sycl::cl_uint': 'cl::sycl::cl_uint',
        'cl::sycl::cl_long': 'cl::sycl::cl_long',
        'cl::sycl::cl_ulong': 'cl::sycl::cl_ulong',
        'cl::sycl::cl_float': 'cl::sycl::cl_float',
        'cl::sycl::cl_double': 'cl::sycl::cl_double',
        'cl::sycl::cl_half': 'cl::sycl::cl_half'
    }
    value_default_dict = defaultdict(lambda: '0', {
        'float': '0.0f',
        'double': '0.0',
        'cl::sycl::half': '0.0f',
        'cl::sycl::cl_float': '0.0f',
        'cl::sycl::cl_double': '0.0',
        'cl::sycl::cl_half': '0.0f'
    })
    vec_name_dict = {
        1: 'One',
        2: 'Two',
        3: 'Three',
        4: 'Four',
        8: 'Eight',
        16: 'Sixteen'
    }
    vals_list_dict = {
        1: ['0'],
        2: ['0', '1'],
        3: ['0', '1', '2'],
        4: ['0', '1', '2', '3'],
        8: ['0', '1', '2', '3', '4', '5', '6', '7'],
        16: [
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12',
            '13', '14', '15'
        ]
    }
    swizzle_xyzw_list_dict = {
        1: ['x'],
        2: ['x', 'y'],
        3: ['x', 'y', 'z'],
        4: ['x', 'y', 'z', 'w']
    }
    swizzle_rgba_list_dict = {
        1: ['r'],
        2: ['r', 'g'],
        3: ['r', 'g', 'b'],
        4: ['r', 'g', 'b', 'a']
    }
    swizzle_elem_list_dict = {
        1: ['cl::sycl::elem::s0'],
        2: ['cl::sycl::elem::s0', 'cl::sycl::elem::s1'],
        3: ['cl::sycl::elem::s0', 'cl::sycl::elem::s1', 'cl::sycl::elem::s2'],
        4: [
            'cl::sycl::elem::s0', 'cl::sycl::elem::s1', 'cl::sycl::elem::s2',
            'cl::sycl::elem::s3'
        ],
        8: [
            'cl::sycl::elem::s0', 'cl::sycl::elem::s1', 'cl::sycl::elem::s2',
            'cl::sycl::elem::s3', 'cl::sycl::elem::s4', 'cl::sycl::elem::s5',
            'cl::sycl::elem::s6', 'cl::sycl::elem::s7'
        ],
        16: [
            'cl::sycl::elem::s0', 'cl::sycl::elem::s1', 'cl::sycl::elem::s2',
            'cl::sycl::elem::s3', 'cl::sycl::elem::s4', 'cl::sycl::elem::s5',
            'cl::sycl::elem::s6', 'cl::sycl::elem::s7', 'cl::sycl::elem::s8',
            'cl::sycl::elem::s9', 'cl::sycl::elem::sA', 'cl::sycl::elem::sB',
            'cl::sycl::elem::sC', 'cl::sycl::elem::sD', 'cl::sycl::elem::sE',
            'cl::sycl::elem::sF'
        ]
    }


kernel_template = Template("""  bool resArray[1] = {true};
  {
    cl::sycl::buffer<bool, 1> boolBuffer(resArray, cl::sycl::range<1>(1));
    testQueue.submit([&](cl::sycl::handler &cgh) {
      auto resAcc = boolBuffer.get_access<cl::sycl::access::mode::write>(cgh);

      cgh.single_task<class ${kernelName}>([=]() {
        ${test}
      });
    });
  }
  if (!resArray[0]) {
    fail_test(log, cl::sycl::string_class("The following vector test failed: ${testName}"));
  }
""")

test_func_template = Template("""
void ${func_name}(util::logger &log) {

  try {
    auto testQueue = util::get_cts_object::queue();
    {
      auto testDevice = testQueue.get_device();
      ${test}
    }
  } catch (const cl::sycl::exception &e) {
    log_exception(log, e);
    cl::sycl::string_class errorMsg =
        "a SYCL exception was caught: " + cl::sycl::string_class(e.what());
    FAIL(log, errorMsg.c_str());
  }
}
""")


def wrap_with_kernel(type_str, kernel_name, test_name, test_string):
    """
    Wraps |test_string| inside a kernel with |kernel_name|.

    Wraps kernels with checks for cl_khr_fp16 and cl_khr_fp64 when appropriate.
    The necessity for extension checks is determined based on |type_str|
    """
    return wrap_with_extension_checks(type_str,
                                      kernel_template.substitute(
                                          kernelName=kernel_name,
                                          testName=test_name,
                                          test=test_string))


def wrap_with_test_func(test_name, type_str, test, additional=''):
    """
    Wraps |test| in a function with name |func_name| and returns the resulting 
    str.
    """
    return test_func_template.substitute(
        func_name=make_func_name(test_name, type_str, additional), test=test)


def make_func_name(test_name, type_str, additional=''):
    """
    Builds a function name of the form

    VECTOR_TEST_|test_name|_|type_str||additional|
    with all ' ' and ':' replace with '_'
    """
    return ('VECTOR_TEST_' + test_name + '_' + type_str + additional).replace(
        ' ', '_').replace(':', '_')


def make_func_call(test_name, type_str, additional=''):
    return make_func_name(test_name, type_str, additional) + '(log);\n'


def wrap_with_half_check(test_string):
    """Wraps test_string with a check for cl_khr_fp16 if appropriate"""
    string = 'if (testDevice.has_extension("cl_khr_fp16")) {\n'
    string += test_string
    string += '}\n'
    return string


def wrap_with_double_check(test_string):
    """Wraps test_string with a check for cl_khr_fp64 if appropriate"""
    string = 'if (testDevice.has_extension("cl_khr_fp64")) {\n'
    string += test_string
    string += '}\n'
    return string


def wrap_with_extension_checks(type_str, test_string):
    if (type_str.count('half') > 0):
        test_string = wrap_with_half_check(test_string)
    if (type_str.count('double') > 0):
        test_string = wrap_with_double_check(test_string)
    test_string = '{\n' + test_string + '}\n'
    return test_string


def append_fp_postfix(type_str, input_val_list):
    """Generates and returns a new list from the input, with .0f or .0 appended
    to each value in the list if type_str is 'float', 'double' or 'cl::sycl::half'"""
    result_val_list = []
    for val in input_val_list:
        if (type_str == 'float' or type_str == 'cl::sycl::cl_float'
                or type_str == 'cl::sycl::half'
                or type_str == 'cl::sycl::cl_half'):
            result_val_list.append(val + '.0f')
        elif type_str == 'double' or type_str == 'cl::sycl::cl_double':
            result_val_list.append(val + '.0')
        else:
            result_val_list.append(val)
    return result_val_list


def generate_value_list(type_str, size):
    """Generates a list incrementing values, up to the given size, with appropriate
    floating point post fixes applied"""
    vec_val_list = []
    for val in Data.vals_list_dict[size]:
        vec_val_list.append(val)
    vec_val_list = append_fp_postfix(type_str, vec_val_list)
    vec_val_string = ', '.join(vec_val_list)
    return vec_val_string


def swap_pairs(input_list):
    """Swaps pairs of elements in a list. The list can be of even or odd length"""
    return_list = []
    return_list.extend(input_list)
    if len(input_list) % 2 == 0:
        return_list[::2], return_list[1::2] = input_list[1::2], input_list[::2]
    else:
        return_list[:len(input_list) - 1:2], return_list[
            1:len(input_list) - 1:2] = input_list[
                1:len(input_list) - 1:2], input_list[:len(input_list) - 1:2]
    return return_list


def get_space_count(line):
    """Gets number of spaces at start of line to preserve indenting"""
    return len(line) - len(line.lstrip())


def add_spaces_to_lines(count, string):
    """Adds a number of spaces to the start of each line"""
    all_lines = string.splitlines(True)
    new_string = all_lines[0]
    for i in range(1, len(all_lines)):
        new_string += ' ' * count + all_lines[i]
    return new_string


def replace_string_in_source_string(source, generated_tests,
                                    replacement_string):
    """Replaces strings in a string with another string"""
    # Get number of spaces to format each line with
    space_count = 0
    for line in source.splitlines(True):
        if replacement_string in line:
            space_count = get_space_count(line)
    # Format each line
    generated_tests = add_spaces_to_lines(space_count, generated_tests)
    # Write lines to source string
    new_source = source.replace(replacement_string, generated_tests)
    # Return new source string
    return new_source


def write_source_file(test_str, func_calls, test_name, input_file, output_file,
                      type_str):

    with open(input_file, 'r') as source_file:
        source = source_file.read()

    source = replace_string_in_source_string(source,
                                             type_str.replace(
                                                 'cl::sycl::', '').replace(
                                                     ' ', '_'), '$TYPE_NAME')
    source = replace_string_in_source_string(source, test_name, '$CATEGORY')
    source = replace_string_in_source_string(source, test_str, '$TEST_FUNCS')
    source = replace_string_in_source_string(source, func_calls, '$FUNC_CALLS')

    with open(
            output_file.strip('.cpp') + '_' + type_str.replace(
                'cl::sycl::', '').replace(' ', '_') + '.cpp', 'w+') as output:
        output.write(source)
