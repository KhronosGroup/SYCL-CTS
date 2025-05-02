# ************************************************************************
#
#   SYCL Conformance Test Suite
#
#
#   Copyright (c) 2018-2022 Codeplay Software LTD. All Rights Reserved.
#   Copyright (c) 2022 The Khronos Group Inc.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# ************************************************************************

from collections import defaultdict
from itertools import product
from math import ceil, floor
from string import Template

class Data:
    signs = [True, False]
    standard_sizes = [1, 2, 3, 4, 8, 16]
    standard_types = [
        'bool', 'char', 'short', 'int', 'long', 'long long', 'float',
        'double', 'sycl::half', 'std::byte'
    ]
    standard_type_dict = {
        (True, 'bool'): 'bool',
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
        (True, 'float'): 'float',
        (True, 'double'): 'double',
        (True, 'sycl::half'): 'sycl::half',
        (False, 'std::byte'): 'std::byte'
    }

    fixed_width_types = [
        'std::int8_t', 'std::int16_t', 'std::int32_t', 'std::int64_t'
    ]

    fixed_width_type_dict = {
        (False, 'std::int8_t'): 'std::uint8_t',
        (True, 'std::int8_t'): 'std::int8_t',
        (False, 'std::int16_t'): 'std::uint16_t',
        (True, 'std::int16_t'): 'std::int16_t',
        (False, 'std::int32_t'): 'std::uint32_t',
        (True, 'std::int32_t'): 'std::int32_t',
        (False, 'std::int64_t'): 'std::uint64_t',
        (True, 'std::int64_t'): 'std::int64_t'
    }

    fixed_width_type_define_dict = {
        ('std::uint8_t'): 'UINT8_MAX',
        ('std::int8_t'): 'INT8_MAX',
        ('std::uint16_t'): 'UINT16_MAX',
        ('std::int16_t'): 'INT16_MAX',
        ('std::uint32_t'): 'UINT32_MAX',
        ('std::int32_t'): 'INT32_MAX',
        ('std::uint64_t'): 'UINT32_MAX',
        ('std::int64_t'): 'INT32_MAX'
    }

    alias_dict = {
        'std::int8_t': 'sycl::char',
        'std::uint8_t': 'sycl::uchar',
        'std::int16_t': 'sycl::short',
        'std::uint16_t': 'sycl::ushort',
        'std::int32_t': 'sycl::int',
        'std::uint32_t': 'sycl::uint',
        'std::int64_t': 'sycl::long',
        'std::uint64_t': 'sycl::ulong',
        'float': 'sycl::float',
        'double': 'sycl::double',
        'sycl::half': 'sycl::half'
    }
    value_default_dict = defaultdict(lambda: '0', {
        'bool': 'false',
        'float': '0.0f',
        'double': '0.0',
        'sycl::half': '0.0f',
        'std::byte': 'std::byte{0}'
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

    vals_list_dict_float = {
        1: ['2.3f'],
        2: ['2.3f', '3.8f'],
        3: ['2.3f', '3.8f', '1.5f'],
        4: ['2.3f', '3.8f', '1.5f', '2.5f'],
        8: ['2.3f', '3.8f', '1.5f', '2.5f', '-2.3f', '-3.8f', '-1.5f', '-2.5f'],
        16: [
            '2.3f', '3.8f', '1.5f', '2.5f', '-2.3f', '-3.8f', '-1.5f', '-2.5f', '2.3f',
            '3.8f', '1.5f', '2.5f', '-2.3f', '-3.8f', '-1.5f', '-2.5f'
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
        1: ['sycl::elem::s0'],
        2: ['sycl::elem::s0', 'sycl::elem::s1'],
        3: ['sycl::elem::s0', 'sycl::elem::s1', 'sycl::elem::s2'],
        4: [
            'sycl::elem::s0', 'sycl::elem::s1', 'sycl::elem::s2',
            'sycl::elem::s3'
        ],
        8: [
            'sycl::elem::s0', 'sycl::elem::s1', 'sycl::elem::s2',
            'sycl::elem::s3', 'sycl::elem::s4', 'sycl::elem::s5',
            'sycl::elem::s6', 'sycl::elem::s7'
        ],
        16: [
            'sycl::elem::s0', 'sycl::elem::s1', 'sycl::elem::s2',
            'sycl::elem::s3', 'sycl::elem::s4', 'sycl::elem::s5',
            'sycl::elem::s6', 'sycl::elem::s7', 'sycl::elem::s8',
            'sycl::elem::s9', 'sycl::elem::sA', 'sycl::elem::sB',
            'sycl::elem::sC', 'sycl::elem::sD', 'sycl::elem::sE',
            'sycl::elem::sF'
        ]
    }

class ReverseData:
    rev_standard_type_dict = { Data.standard_type_dict[k] : k for k in list(Data.standard_type_dict.keys()) }
    rev_fixed_width_type_dict = { Data.fixed_width_type_dict[k] : k for k in list(Data.fixed_width_type_dict.keys()) }


kernel_template = Template("""  bool resArray[1] = {true};
  {
    sycl::buffer<bool, 1> boolBuffer(resArray, sycl::range<1>(1));
    testQueue.submit([&](sycl::handler &cgh) {
      auto resAcc = boolBuffer.get_access<sycl::access_mode::write>(cgh);

      cgh.single_task<class ${kernelName}>([=]() {
        ${test}
      });
    });
  }
  INFO("Checking ${testName}");
  CHECK(resArray[0]);
""")

# Unlike in the template above, some of the checks which are done by tests for
# swizzles are outlined to host code.
# This is done to reduce kernel size and have more information about a problem
# in case of a test failure.
swizzle_kernel_template = Template("""
  {
    auto vecBuffer = sycl::buffer<sycl::vec<${type}, ${size}>, 1>(
        sycl::range<1>(total_per_element_swizzle_test_cases));
    bool resArray[1] = {true};
    {
      sycl::buffer<bool, 1> boolBuffer(resArray, sycl::range<1>(1));
      testQueue.submit([&](sycl::handler &cgh) {
        sycl::accessor resAcc(boolBuffer, cgh, sycl::write_only);
        sycl::accessor vecAcc(vecBuffer, cgh, sycl::write_only);

        cgh.single_task<class ${kernelName}>([=]() {
          ${test}
        });
      });
    }
    INFO("Checking ${testName}");
    CHECK(resArray[0]);

    ${type} in_order_vals[] = {${in_order_vals}};
    ${type} reversed_vals[] = {${reversed_vals}};
    ${type} in_order_reversed_pair_vals[] = {${in_order_pair_vals}};
    ${type} reverse_order_reversed_pair_vals[] = {${reverse_order_pair_vals}};

    auto vecAcc = vecBuffer.get_host_access();
    CHECK(check_vector_values<${type}, ${size}>(vecAcc[in_order], in_order_vals));
    CHECK(check_vector_values<${type}, ${size}>(vecAcc[reverse_order], reversed_vals));
    CHECK(check_vector_values<${type}, ${size}>(vecAcc[in_order_reversed_pair], in_order_reversed_pair_vals));
    CHECK(check_vector_values<${type}, ${size}>(vecAcc[reverse_order_reversed_pair], reverse_order_reversed_pair_vals));
  }
""")

test_func_template = Template("""
void ${func_name}(util::logger &log) {

  {
    auto testQueue = util::get_cts_object::queue();
    {
      auto testDevice = testQueue.get_device();
      ${test}
    }
  }
}
""")

def cast_to_bool(val_list_dict):
    result_list_dict = val_list_dict.copy()
    for key, val in val_list_dict.items():
        for i in range(len(val)):
            val[i] = str(
                (int(val[i]) % 2 == 0).real)
        result_list_dict[key] = val
    return result_list_dict


def remove_namespaces_whitespaces(type_str):
    """
    Clear type name from namespaces and whitespaces
    """
    return type_str.replace('sycl::', '').replace(
            ' ', '_').replace('std::byte', 'std_byte').replace(
            'std::', '')

def wrap_with_kernel(type_str, kernel_name, test_name, test_string):
    """
    Wraps |test_string| inside a kernel with |kernel_name|.

    Wraps kernels with checks for fp16 and fp64 when appropriate.
    The necessity for extension checks is determined based on |type_str|
    """

    return wrap_with_extension_checks(type_str,
                                      kernel_template.substitute(
                                      kernelName=remove_namespaces_whitespaces(kernel_name),
                                      testName=test_name,
                                      test=test_string))

def wrap_with_swizzle_kernel(type_str, vec_size, in_order_vals, reversed_vals,
        in_order_pair_vals, reverse_order_pair_vals, kernel_name, test_name,
        test_string):
    """
    Wraps |test_string| inside a kernel with |kernel_name|.

    Wraps kernels with checks for fp16 and fp64 when appropriate.
    The necessity for extension checks is determined based on |type_str|

    Unlike |wrap_with_kernel| above, this function accepts several extra
    arguments, because kernels for swizzle tests do some of the validation
    checks on host to reduce kernel size and make error messages more detailed.
    """

    return wrap_with_extension_checks(type_str,
                                      swizzle_kernel_template.substitute(
                                      kernelName=remove_namespaces_whitespaces(kernel_name),
                                      testName=test_name,
                                      test=test_string,
                                      type=type_str,
                                      size=vec_size,
                                      in_order_vals=in_order_vals,
                                      reversed_vals=reversed_vals,
                                      in_order_pair_vals=in_order_pair_vals,
                                      reverse_order_pair_vals=reverse_order_pair_vals))



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
    """Wraps test_string with a check for fp16 if appropriate"""
    string = 'if (testDevice.has(sycl::aspect::fp16)) {\n'
    string += test_string
    string += '}\n'
    return string


def wrap_with_double_check(test_string):
    """Wraps test_string with a check for fp64 if appropriate"""
    string = 'if (testDevice.has(sycl::aspect::fp64)) {\n'
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


def make_fp_or_byte_explicit(type_str, input_val_list):
    """Generates and returns a new list from the input, with .0f or .0 appended
    to each value in the list if type_str is 'float', 'double' or 'sycl::half'"""
    result_val_list = []
    for val in input_val_list:
        if (type_str == 'float'
                or type_str == 'sycl::half'):
            result_val_list.append(val + '.0f')
        elif type_str == 'double':
            result_val_list.append(val + '.0')
        elif type_str == 'std::byte':
            result_val_list.append('std::byte{{{}}}'.format(val))
        else:
            result_val_list.append(val)
    return result_val_list


def generate_value_list(type_str, size):
    """Generates a list incrementing values, up to the given size, with appropriate
    floating point post fixes applied"""
    vec_val_list = []
    for val in Data.vals_list_dict[size]:
        vec_val_list.append(val)
    vec_val_list = make_fp_or_byte_explicit(type_str, vec_val_list)
    vec_val_string = ', '.join(vec_val_list)
    return str(vec_val_string)


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
    if not all_lines:
        return ''
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

def get_ifdef_string(source, type_str):
    if type_str == 'std::byte':
        source = source.replace('$IFDEF',
'''
#if SYCL_CTS_COMPILING_WITH_SIMSYCL
  FAIL_CHECK("SimSYCL doesn't support sycl::vec<N, std::byte>");
#else
$IFDEF
''')
        source = source.replace('$ENDIF',
'''
$ENDIF
#endif
''')

    if type_str in ReverseData.rev_fixed_width_type_dict:
        include_string = '#include <cstdint>\n'
        ifdef_string = '#ifdef ' + Data.fixed_width_type_define_dict[type_str]
        source = source.replace('$IFDEF', include_string + ifdef_string)
        source = source.replace('$ENDIF', '#endif // ' + ifdef_string)
    else:
        source = source.replace('$IFDEF', '')
        source = source.replace('$ENDIF', '')
    return source

def write_source_file(test_str, func_calls, test_name, input_file, output_file,
                      type_str):

    with open(input_file, 'r') as source_file:
        source = source_file.read()

    source = replace_string_in_source_string(source,
                                             remove_namespaces_whitespaces(type_str),
                                            '$TYPE_NAME')
    source = replace_string_in_source_string(source, test_name, '$CATEGORY')
    source = replace_string_in_source_string(source, test_str, '$TEST_FUNCS')
    source = replace_string_in_source_string(source, func_calls, '$FUNC_CALLS')

    source = get_ifdef_string(source, type_str)

    with open(output_file, 'w+') as output:
        output.write(source)

def get_types():
    types = ['char', 'sycl::byte']
    for base_type in Data.standard_types:
        for sign in Data.signs:
            if (base_type == 'float' or base_type == 'double' or base_type == 'bool'
                or base_type == 'sycl::half') and sign is False:
                continue
            if base_type == 'std::byte' and sign is True:
                continue
            types.append(Data.standard_type_dict[(sign, base_type)])

    for base_type in Data.fixed_width_types:
        for sign in Data.signs:
            types.append(Data.fixed_width_type_dict[(sign, base_type)])
    return types

# FIXME: We shouldn't accumulate results of *all* checks into a single boolean,
# because it makes debugging failures really hard as there is no indication
# about what exactly went wrong.
# FIXME: vector swizzles should be tested not only on device, but also on host.
# Therefore, the test should ideally perform the same calculations on both host
# and device and then compare to ensure that results are equivalend and match
# reference.
# FIXME: consider reducing amount of calls to 'check_convert_as_all_types'
# swizzle_template invokes 'check_convert_as_all_types' and this template
# is "instantiated" for every simple swizzle there is (such as xyzz, zyxw, etc.)
# which seems overly exessive.
class SwizzleData:
    swizzle_template = Template(
        """        sycl::vec<${type}, ${size}> ${name}DimTestVec = sycl::vec<${type}, ${size}>(${testVecValues});
            sycl::vec<${type}, ${size}> swizzledVec {${name}DimTestVec.${indexes}()};
            ${type} in_order_vals[] = {${in_order_vals}};
            ${type} reversed_vals[] = {${reversed_vals}};
            ${type} in_order_reversed_pair_vals[] = {${in_order_pair_vals}};
            ${type} reverse_order_reversed_pair_vals[] = {${reverse_order_pair_vals}};
            if (!check_equal_type_bool<sycl::vec<${type}, ${size}>>(swizzledVec)) {
                resAcc[0] = false;
            }
            if (!check_vector_size<${type}, ${size}>(swizzledVec)) {
                resAcc[0] = false;
            }
            if (!check_vector_values<${type}, ${size}>(swizzledVec, in_order_vals)) {
                resAcc[0] = false;
            }
            if (!check_vector_size_byte_size<${type}, ${size}>(swizzledVec)) {
                resAcc[0] = false;
            }
#if SYCL_CTS_ENABLE_FULL_CONFORMANCE
            if (!check_convert_as_all_types<${type}, ${size}>(swizzledVec)) {
                resAcc[0] = false;
            }
#endif // SYCL_CTS_ENABLE_FULL_CONFORMANCE
    """)

    lo_hi_odd_even_template = Template(
        """        if (!check_lo_hi_odd_even<${type}>(swizzledVec, in_order_vals)) {
            resAcc[0] = false;
            }
    """)

    swizzle_elem_template = Template(
        """
            vecAcc[in_order] = swizzledVec.template swizzle<${in_order_swiz_indexes}>();
            vecAcc[reverse_order] = swizzledVec.template swizzle<${reverse_order_swiz_indexes}>();
            vecAcc[in_order_reversed_pair] = swizzledVec.template swizzle<${in_order_reversed_pair_swiz_indexes}>();
            vecAcc[reverse_order_reversed_pair] = swizzledVec.template swizzle<${reverse_order_reversed_pair_swiz_indexes}>();
    """)

    swizzle_full_test_template = Template(
        """        sycl::vec<${type}, ${size}> ${name}DimTestVec = sycl::vec<${type}, ${size}>(${testVecValues});
            sycl::vec<${type}, ${size}> inOrderSwizzleFunctionVec {${name}DimTestVec.template swizzle<${in_order_swiz_indexes}>()};
            vecAcc[in_order] = inOrderSwizzleFunctionVec;
            if (!check_equal_type_bool<sycl::vec<${type}, ${size}>>(inOrderSwizzleFunctionVec)) {
                resAcc[0] = false;
            }
            if (!check_vector_size<${type}, ${size}>(inOrderSwizzleFunctionVec)) {
                resAcc[0] = false;
            }
            if (!check_vector_size_byte_size<${type}, ${size}>(inOrderSwizzleFunctionVec)) {
                resAcc[0] = false;
            }
#if SYCL_CTS_ENABLE_FULL_CONFORMANCE
            if (!check_convert_as_all_types<${type}, ${size}>(inOrderSwizzleFunctionVec)) {
                resAcc[0] = false;
            }
#endif // SYCL_CTS_ENABLE_FULL_CONFORMANCE

            sycl::vec<${type}, ${size}> reverseOrderSwizzleFunctionVec {${name}DimTestVec.template swizzle<${reverse_order_swiz_indexes}>()};
            vecAcc[reverse_order] = reverseOrderSwizzleFunctionVec;
            if (!check_equal_type_bool<sycl::vec<${type}, ${size}>>(reverseOrderSwizzleFunctionVec)) {
                resAcc[0] = false;
            }
            if (!check_vector_size<${type}, ${size}>(reverseOrderSwizzleFunctionVec)) {
                resAcc[0] = false;
            }
            if (!check_vector_size_byte_size<${type}, ${size}>(reverseOrderSwizzleFunctionVec)) {
                resAcc[0] = false;
            }
#if SYCL_CTS_ENABLE_FULL_CONFORMANCE
            if (!check_convert_as_all_types<${type}, ${size}>(reverseOrderSwizzleFunctionVec)) {
                resAcc[0] = false;
            }
#endif // SYCL_CTS_ENABLE_FULL_CONFORMANCE

            sycl::vec<${type}, ${size}> inOrderReversedPairSwizzleFunctionVec {${name}DimTestVec.template swizzle<${in_order_reversed_pair_swiz_indexes}>()};
            vecAcc[in_order_reversed_pair] = inOrderReversedPairSwizzleFunctionVec;
            if (!check_equal_type_bool<sycl::vec<${type}, ${size}>>(inOrderReversedPairSwizzleFunctionVec)) {
                resAcc[0] = false;
            }
            if (!check_vector_size<${type}, ${size}>(inOrderReversedPairSwizzleFunctionVec)) {
                resAcc[0] = false;
            }
            if (!check_vector_size_byte_size<${type}, ${size}>(inOrderReversedPairSwizzleFunctionVec)) {
                resAcc[0] = false;
            }
#if SYCL_CTS_ENABLE_FULL_CONFORMANCE
            if (!check_convert_as_all_types<${type}, ${size}>(inOrderReversedPairSwizzleFunctionVec)) {
                resAcc[0] = false;
            }
#endif // SYCL_CTS_ENABLE_FULL_CONFORMANCE

            sycl::vec<${type}, ${size}> reverseOrderReversedPairSwizzleFunctionVec {${name}DimTestVec.template swizzle<${reverse_order_reversed_pair_swiz_indexes}>()};
            vecAcc[reverse_order_reversed_pair] = reverseOrderReversedPairSwizzleFunctionVec;
            if (!check_equal_type_bool<sycl::vec<${type}, ${size}>>(reverseOrderReversedPairSwizzleFunctionVec)) {
                resAcc[0] = false;
            }
            if (!check_vector_size<${type}, ${size}>(reverseOrderReversedPairSwizzleFunctionVec)) {
                resAcc[0] = false;
            }
            if (!check_vector_size_byte_size<${type}, ${size}>(reverseOrderReversedPairSwizzleFunctionVec)) {
                resAcc[0] = false;
            }
#if SYCL_CTS_ENABLE_FULL_CONFORMANCE
            if (!check_convert_as_all_types<${type}, ${size}>(reverseOrderReversedPairSwizzleFunctionVec)) {
                resAcc[0] = false;
            }
#endif // SYCL_CTS_ENABLE_FULL_CONFORMANCE
    """)

def substitute_swizzles_templates(type_str, size, index_subset, value_subset, convert_type_str, as_type_str):
    string = ''
    index_list = []
    val_list = []
    for index, value in zip(index_subset, value_subset):
        index_list.append(index)
        val_list.append(value)
    index_string = ''.join(index_list)
    test_string = SwizzleData.swizzle_template.substitute(
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
        test_string += SwizzleData.lo_hi_odd_even_template.substitute(
            type=type_str, size=size)
    test_string += SwizzleData.swizzle_elem_template.substitute(
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
    string += wrap_with_swizzle_kernel(
            type_str, str(size), ', '.join(val_list), ', '.join(val_list[::-1]),
            ', '.join(swap_pairs(val_list)), ', '.join(swap_pairs(val_list[::-1])),
        'KERNEL_' + type_str + str(size) +
            index_string,
        'vec<' + type_str + ', ' + str(size) + '>.' + index_string,
        test_string)
    return string

def gen_swizzle_test(type_str, convert_type_str, as_type_str, size, num_batches, batch_index):
    string = ''
    val_list = make_fp_or_byte_explicit(type_str, Data.vals_list_dict[size])
    if size > 4:
        test_string = SwizzleData.swizzle_full_test_template.substitute(
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
            in_order_vals=', '.join(val_list),
            reversed_vals=', '.join(val_list[::-1]),
            in_order_pair_vals=', '.join(swap_pairs(val_list)),
            reverse_order_pair_vals=', '.join(swap_pairs(val_list[::-1])))
        string += wrap_with_swizzle_kernel(
                type_str, str(size), ', '.join(val_list),
                ', '.join(val_list[::-1]),
                ', '.join(swap_pairs(val_list)),
                ', '.join(swap_pairs(val_list[::-1])),
            'ELEM_KERNEL_' + type_str + str(size) +
            ''.join(Data.swizzle_elem_list_dict[size][:size]).replace(
                'sycl::elem::', ''),
            'vec<' + type_str + ', ' + str(size) + '> .swizzle<' +
            ', '.join(Data.swizzle_elem_list_dict[size][:size]) + '>',
            test_string)
        return string
   
    # Case when size <=4
    # The test files generated for swizzles of vectors of size <= 4 are enormous and are hurting 
    # compilation times of the suite so we batch the tests according to two command line arguments 
    # in num_batches and batch_index that will dictate how many tests we can put in a single test file. 
    # Specifically, the test cases are to be split in num_batches different groups aka batches 
    # and the batch_index tells the script which batch in particular we want to output to a test file during this run. 
    # Both of these arguments, num_batches and batch_index, are controlled by the cmake test generation script.
    
    total_tests = 0
    for length in range(size, size + 1):
        for index_subset, value_subset in zip(
                product(
                    Data.swizzle_xyzw_list_dict[size][:size],
                    repeat=length),
                product(val_list[:size], repeat=length)):
            total_tests += 1
    batch_size = ceil(total_tests / num_batches)
    cur_index = 0
    cur_batch = 0
    for length in range(size, size + 1):
        for index_subset, value_subset in zip(
                product(
                    Data.swizzle_xyzw_list_dict[size][:size],
                    repeat=length),
                product(val_list[:size], repeat=length)):
            cur_batch = floor(cur_index / batch_size)
            if cur_batch > batch_index:
                break
            if cur_batch == batch_index:
                string += substitute_swizzles_templates(type_str, size,
                        index_subset, value_subset, convert_type_str, as_type_str)
            cur_index += 1

    # Same logic as above repeated for the case when size == 4
    if size == 4:
        total_tests = 0
        for length in range(size, size + 1):
            for index_subset, value_subset in zip(
                    product(
                        Data.swizzle_rgba_list_dict[size][:size],
                        repeat=length),
                    product(
                        val_list[:size], repeat=length)):
                total_tests += 1
        batch_size = ceil(total_tests / num_batches)
        cur_index = 0
        cur_batch = 0
        for length in range(size, size + 1):
            for index_subset, value_subset in zip(
                    product(
                        Data.swizzle_rgba_list_dict[size][:size],
                        repeat=length),
                    product(val_list[:size], repeat=length)):
                cur_batch = floor(cur_index / batch_size)
                if cur_batch > batch_index:
                    break
                if cur_batch == batch_index:
                    string += substitute_swizzles_templates(type_str, size,
                            index_subset, value_subset, convert_type_str, as_type_str)
                cur_index += 1
    return string


def write_swizzle_source_file(swizzles, input_file, output_file, type_str):

    with open(input_file, 'r') as source_file:
        source = source_file.read()

    source = replace_string_in_source_string(source,
                                            remove_namespaces_whitespaces(type_str),
                                            '$TYPE_NAME')

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

    source = get_ifdef_string(source, type_str)

    with open(output_file, 'w+') as output:
        output.write(source)

def get_reverse_type(type_str):
    if type_str == 'char' or type_str == 'sycl::byte':
      return type_str
    if type_str in ReverseData.rev_standard_type_dict:
        type_dict =  Data.standard_type_dict
        rev_type_dict = ReverseData.rev_standard_type_dict
    else:
        type_dict =  Data.fixed_width_type_dict
        rev_type_dict = ReverseData.rev_fixed_width_type_dict
    (sign, base_type) = rev_type_dict[type_str]
    if (not sign, base_type) in type_dict:
        reverse_type_str = type_dict[(not sign, base_type)]
    else:
        reverse_type_str = type_str
    return reverse_type_str

# FIXME: Move this and other functions to generate_vector_swizzles.py
# Reason for the TODO above is that this function and several more it calls are
# not really common and only used to generate vector_swizzles test.
# FIXME: The test (main template and others) should be updated to use Catch2
def make_swizzles_tests(type_str, input_file, output_file, num_batches, batch_index):
    if type_str == 'bool':
        Data.vals_list_dict = cast_to_bool(Data.vals_list_dict)

    swizzles = [None] * 6

    convert_type_str = get_reverse_type(type_str)
    as_type_str = get_reverse_type(type_str)
    swizzles[0] = gen_swizzle_test(type_str, convert_type_str,
                                   as_type_str, 1, num_batches, batch_index)
    swizzles[1] = gen_swizzle_test(type_str, convert_type_str,
                                   as_type_str, 2, num_batches, batch_index)
    swizzles[2] = gen_swizzle_test(type_str, convert_type_str,
                                   as_type_str, 3, num_batches, batch_index)
    swizzles[3] = gen_swizzle_test(type_str, convert_type_str,
                                   as_type_str, 4, num_batches, batch_index)
    swizzles[4] = gen_swizzle_test(type_str, convert_type_str,
                                   as_type_str, 8, num_batches, batch_index)
    swizzles[5] = gen_swizzle_test(type_str, convert_type_str,
                                   as_type_str, 16, num_batches, batch_index)
    write_swizzle_source_file(swizzles, input_file, output_file, type_str)
