# ************************************************************************
#
#   SYCL Conformance Test Suite
#
#   Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
#
# ************************************************************************

class Data:
    standard_types = [
        'char',
        'short',
        'int',
        'long',
        'long long',
        'float',
        'double',
        'half']
    signs = [True, False]
    standard_sizes = [1, 2, 3, 4, 8, 16]
    standard_type_dict = {(False, 'char'): 'unsigned char',
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
                          (False, 'half'): 'half',
                          (True, 'half'): 'half'}
    vals_dict = {
        1: '0',
        2: '0, 1',
        3: '0, 1, 2',
        4: '0, 1, 2, 3',
        8: '0, 1, 2, 3, 4, 5, 6, 7',
        16: '0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15'}
    reverse_vals_dict = {
        1: '0',
        2: '1, 0',
        3: '2, 1, 0',
        4: '3, 2, 1, 0',
        8: '7, 6, 5, 4, 3, 2, 1, 0',
        16: '15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0'}
    swizzle_index_dict = {
        1: 'elem::s0',
        2: 'elem::s0, elem::s1',
        3: 'elem::s0, elem::s1, elem::s2',
        4: 'elem::s0, elem::s1, elem::s2, elem::s3',
        8: 'elem::s0, elem::s1, elem::s2, elem::s3, elem::s4, elem::s5, elem::s6, elem::s7',
        16: 'elem::s0, elem::s1, elem::s2, elem::s3, elem::s4, elem::s5, elem::s6, elem::s7, \
elem::s8, elem::s9, elem::sA, elem::sB, elem::sC, elem::sD, elem::sE, elem::sF'}
    reverse_swizzle_index_dict = {
        1: 'elem::s0',
        2: 'elem::s1, elem::s0',
        3: 'elem::s2, elem::s1, elem::s0',
        4: 'elem::s3, elem::s2, elem::s1, elem::s0',
        8: 'elem::s7, elem::s6, elem::s5, elem::s4, elem::s3, elem::s2, elem::s1, elem::s0',
        16: 'elem::sF, elem::sE, elem::sD, elem::sC, elem::sB, elem::sA, elem::s9, elem::s8, \
elem::s7, elem::s6, elem::s5, elem::s4, elem::s3, elem::s2, elem::s1, elem::s0'}


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


def replace_string_in_source_string(source, generated_tests, replacement_string):
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
