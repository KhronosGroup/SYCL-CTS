#!/usr/bin/python
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
#
#  SYCL Conformance Test Suite
#
#  Copyright: (c) 2014 by Codeplay Software LTD. All Rights Reserved.
#

__author__ = "Codeplay"

import os
import sys
import itertools

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
# global variables
#

g_vector_types = {"2" : "xy", "3" : "xyz", "4": "xyzw"}

g_binary_map = [{ "$VECTOR_SIZE$" : "2", "$NUM_TESTS$" : "0", "$RHS_SWIZZLE_TESTS$" : ""},
                { "$VECTOR_SIZE$" : "3", "$NUM_TESTS$" : "0", "$RHS_SWIZZLE_TESTS$" : ""},
                { "$VECTOR_SIZE$" : "4", "$NUM_TESTS$" : "0", "$RHS_SWIZZLE_TESTS$" : ""}]

                
g_generated_header = \
"""
/************************************************************************************
//
//  This file was AUTOMATICALLY GENERATED via generate_vector_swizzle.py
//
************************************************************************************/
"""

    
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
# generate .cpp files based on template and map list
#
def generate( template_name, template_map ):

    print "generating... " + template_name

    # get path to the template file and generated files from the script location
    pathname = os.path.dirname(sys.argv[0])

    template_file_name = template_name + ".template"
    template_file = os.path.join(pathname, template_file_name)
    # open file
    with open(template_file, "r") as f:
        template_content = f.read()

    if template_content is None:
        print "unable to load template file"
        return
        
    if template_map is None:
        print "substitution map is invalid"
        return
        
    for item in template_map:
        generated_content = template_content
        
        # replace strings
        generated_content = generated_content.replace("$VECTOR_SIZE$", item['$VECTOR_SIZE$'])
        generated_content = generated_content.replace("$NUM_TESTS$", item['$NUM_TESTS$'])
        generated_content = generated_content.replace("$RHS_SIMPLE_SWIZZLE_TESTS$", item['$RHS_SIMPLE_SWIZZLE_TESTS$'])
        generated_content = generated_content.replace("$RHS_SIMPLE_SWIZZLE_TEST_VERIFIERS$",  item['$RHS_SIMPLE_SWIZZLE_TEST_VERIFIERS$'])
        generated_content = generated_content.replace("$LHS_SIMPLE_SWIZZLE_TESTS$", item['$LHS_SIMPLE_SWIZZLE_TESTS$'])
        generated_content = generated_content.replace("$LHS_SIMPLE_SWIZZLE_TEST_VERIFIERS$",  item['$LHS_SIMPLE_SWIZZLE_TEST_VERIFIERS$'])
        generated_content = generated_content.replace("$RHS_TEMPLATE_SWIZZLE_TESTS$", item['$RHS_TEMPLATE_SWIZZLE_TESTS$'])
        generated_content = generated_content.replace("$RHS_TEMPLATE_SWIZZLE_TEST_VERIFIERS$",  item['$RHS_TEMPLATE_SWIZZLE_TEST_VERIFIERS$'])
        generated_content = generated_content.replace("$LHS_TEMPLATE_SWIZZLE_TESTS$", item['$LHS_TEMPLATE_SWIZZLE_TESTS$'])
        generated_content = generated_content.replace("$LHS_TEMPLATE_SWIZZLE_TEST_VERIFIERS$",  item['$LHS_TEMPLATE_SWIZZLE_TEST_VERIFIERS$'])

        # write generated file        
        generated_file_name = template_name + "_" + item['$VECTOR_SIZE$'] + ".cpp"
        generated_file = os.path.join(pathname, generated_file_name)
        with open(generated_file, "w") as f:
            f.write(g_generated_header)
            f.write(generated_content)
    
    return


# generate a list of all possible swizzles given a list of possible elements
def make_swizzle( input, repeated ):
    out = []
    if repeated:
        l = list(itertools.product( input, repeat = len(input) ))
    else:
        l = list(itertools.permutations( input ))
    for i in l:
        prod = ''
        for j in i: prod += j
        out.append( prod )
    return out

        
def generate_swizzle_test_permutations( vector_types, template_map, test_type_str, template_tests = False, repeated = False ):
    # iterate through vector types of 2D, 3D, 4D
    for item in template_map:
        # obtain combination swizzles
        swizzles = make_swizzle( vector_types[item["$VECTOR_SIZE$"]] , repeated )
        test_count = int(item["$NUM_TESTS$"])
        
        # newline with eight spaces
        test_seperator_str = "\n        "
        test_output_str = test_seperator_str
        # newline with twelve spaces
        test_verify_seperator_str = "\n            "
        test_verify_output_str = test_verify_seperator_str

        # for each swizzle, create a test macro and a test verification macro
        for swizzle_item in swizzles:
            # test macro
            if template_tests:
                commaed_swizzle_item = ', '.join(swizzle_item)
                commaed_swizzle_item = commaed_swizzle_item.upper()
                test_str = "%s_SWIZZLE( %d, %s )" % (test_type_str, test_count, commaed_swizzle_item)
            else:
                test_str = "%s_SWIZZLE( %d, %s )" % (test_type_str, test_count, swizzle_item)
            test_output_str = test_output_str + test_str + test_seperator_str
            
            # test verification macro
            commaed_swizzle_item = ', '.join(swizzle_item)
            commaed_swizzle_item = commaed_swizzle_item.upper()
            test_verify_str = "SWIZZLE_VERIFY_EQUALS( %d, %s )" % (test_count, commaed_swizzle_item)
            test_verify_output_str = test_verify_output_str + test_verify_str + test_verify_seperator_str

            test_count = test_count + 1
        
        # save it to the global map to print the tests out
        item["$NUM_TESTS$"] = str( test_count )
        item["$%s_SWIZZLE_TESTS$" % test_type_str ] = test_output_str
        item["$%s_SWIZZLE_TEST_VERIFIERS$" % test_type_str ] = test_verify_output_str


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
# entry point
#
def main( ):
    print "generation script"
    generate_swizzle_test_permutations( g_vector_types, g_binary_map, "RHS_SIMPLE" , repeated = False)
    generate_swizzle_test_permutations( g_vector_types, g_binary_map, "LHS_SIMPLE" )
    generate_swizzle_test_permutations( g_vector_types, g_binary_map, "RHS_TEMPLATE", template_tests = True, repeated = False)
    generate_swizzle_test_permutations( g_vector_types, g_binary_map, "LHS_TEMPLATE", template_tests = True)
    generate( "vector_swizzle", g_binary_map )
    return

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
#
#
if __name__ == '__main__':
    main( )
