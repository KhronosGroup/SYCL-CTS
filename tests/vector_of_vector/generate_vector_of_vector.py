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
from itertools import combinations, chain


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
# global variables
#

g_vector_sizes = [ 2, 3, 4, 8, 16 ]
g_vector_types = [ "char", "uchar", "short", "ushort", "int", "uint", "float", "double" ]
g_binary_map = []
                
g_generated_header = \
"""
/************************************************************************************
//
//  This file was AUTOMATICALLY GENERATED via generate_vector_of_vector.py
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
        generated_content = generated_content.replace("$VECTOR_TYPE$", item['$VECTOR_TYPE$'])
        generated_content = generated_content.replace("$VECTOR_SIZE$", item['$VECTOR_SIZE$'])
        generated_content = generated_content.replace("$NUM_TESTS$", item['$NUM_TESTS$'])        
        generated_content = generated_content.replace("$CONSTRUCTOR_TESTS$", item['$CONSTRUCTOR_TESTS$'])
        generated_content = generated_content.replace("$CONSTRUCTOR_TEST_VERIFIERS$", item['$CONSTRUCTOR_TEST_VERIFIERS$'])

        # write generated file        
        generated_file_name = template_name + "_" + item['$VECTOR_TYPE$'] + item['$VECTOR_SIZE$'] + ".cpp"
        generated_file = os.path.join(pathname, generated_file_name)
        with open(generated_file, "w") as f:
            f.write(g_generated_header)
            f.write(generated_content)
    
    return

       
def generate_test_permutations( vector_types, vector_sizes, template_map ):
    g_set = (1, 2, 3, 4, 8, 16)
     # newline with eight spaces
    test_seperator_str = "\n        "    
    # newline with twelve spaces
    test_verify_seperator_str = "\n            "    
        
    # iterate through vector types of 2D, 3D, 4D
    for type in g_vector_types:
        for size in g_vector_sizes:
        
            test_count = 0
            test_output_str = test_seperator_str
            test_verify_output_str = test_verify_seperator_str
            
            # obtain combinations
            for variation in sum_to_n(size):
                # special case for 16, to limit the number of combinations
                if size == 16:
                    g_set = (2, 3, 4, 8, 16)
                if set(variation).issubset(g_set):
                    test_str = "CONSTRUCTOR_TEST( %d, " % test_count
                    for i in variation:
                            test_str = test_str + "v%d, " % i
                    test_str = test_str[:-2] + " )"
                    test_output_str = test_output_str + test_str + test_seperator_str
                                  
                    test_verify_str = "VERIFY_EQUALS( %d, " % test_count
                    for i in variation:
                            test_verify_str = test_verify_str + "V%d, " % i
                    test_verify_str = test_verify_str[:-2] + " )"
                    test_verify_output_str = test_verify_output_str + test_verify_str + test_verify_seperator_str
                    
                    test_count = test_count + 1
            
            template_map.append({ "$VECTOR_TYPE$" : "%s" % type, "$VECTOR_SIZE$" : "%d" % size, "$NUM_TESTS$" : "%d" % test_count, "$CONSTRUCTOR_TESTS$" : "%s" % test_output_str, "$CONSTRUCTOR_TEST_VERIFIERS$" : "%s" % test_verify_output_str})

            
# Taken from http://stackoverflow.com/questions/2065553/get-all-numbers-that-add-up-to-a-number
def sum_to_n(n):
    'Generate the series of +ve integer lists which sum to a +ve integer, n.'
    from operator import sub
    b, mid, e = [0], list(range(1, n)), [n]
    splits = (d for i in range(n) for d in combinations(mid, i)) 
    return (list(map(sub, chain(s, e), chain(b, s))) for s in splits)

        
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
# entry point
#
def main( ):
    print "generation script"    
    generate_test_permutations( g_vector_types, g_vector_sizes, g_binary_map )
    generate( "vector_of_vector", g_binary_map )    
    return

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
#
#
if __name__ == '__main__':
    main( )
