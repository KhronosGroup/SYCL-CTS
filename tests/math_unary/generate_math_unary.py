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

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
# global variables
#               
g_unary_map  = [{"$TEST_FUNC$" : "cos",      "$MAX_ULPS$" : "4.0f"},
                {"$TEST_FUNC$" : "sin",      "$MAX_ULPS$" : "4.0f"},
                {"$TEST_FUNC$" : "acos",     "$MAX_ULPS$" : "4.0f"},
                {"$TEST_FUNC$" : "acosh",    "$MAX_ULPS$" : "4.0f"},
                {"$TEST_FUNC$" : "acospi",   "$MAX_ULPS$" : "5.0f"},
                {"$TEST_FUNC$" : "asin",     "$MAX_ULPS$" : "4.0f"},
                {"$TEST_FUNC$" : "asinh",    "$MAX_ULPS$" : "4.0f"},
                {"$TEST_FUNC$" : "asinpi",   "$MAX_ULPS$" : "5.0f"},
                {"$TEST_FUNC$" : "atan",     "$MAX_ULPS$" : "5.0f"},
                {"$TEST_FUNC$" : "atanh",    "$MAX_ULPS$" : "5.0f"},
                {"$TEST_FUNC$" : "atanpi",   "$MAX_ULPS$" : "5.0f"},
                {"$TEST_FUNC$" : "cbrt",     "$MAX_ULPS$" : "2.0f"},
                {"$TEST_FUNC$" : "ceil",     "$MAX_ULPS$" : "0.0f"},
                {"$TEST_FUNC$" : "cosh",     "$MAX_ULPS$" : "4.0f"},
                {"$TEST_FUNC$" : "cospi",    "$MAX_ULPS$" : "4.0f"},
                {"$TEST_FUNC$" : "expm1",    "$MAX_ULPS$" : "3.0f"},
                {"$TEST_FUNC$" : "fabs",     "$MAX_ULPS$" : "0.0f"},
                {"$TEST_FUNC$" : "floor",    "$MAX_ULPS$" : "0.0f"},
                {"$TEST_FUNC$" : "lgamma",   "$MAX_ULPS$" : "INFINITY"},
                {"$TEST_FUNC$" : "log10",    "$MAX_ULPS$" : "3.0f"},
                {"$TEST_FUNC$" : "log1p",    "$MAX_ULPS$" : "2.0f"},
                {"$TEST_FUNC$" : "logb",     "$MAX_ULPS$" : "0.0f"},
                {"$TEST_FUNC$" : "rint",     "$MAX_ULPS$" : "0.0f"},
                {"$TEST_FUNC$" : "round",    "$MAX_ULPS$" : "0.0f"},
                {"$TEST_FUNC$" : "rsqrt",    "$MAX_ULPS$" : "2.0f"},
                {"$TEST_FUNC$" : "sinh",     "$MAX_ULPS$" : "4.0f"},
                {"$TEST_FUNC$" : "sinpi",    "$MAX_ULPS$" : "4.0f"},
                {"$TEST_FUNC$" : "tanh",     "$MAX_ULPS$" : "5.0f"},
                {"$TEST_FUNC$" : "tanpi",    "$MAX_ULPS$" : "6.0f"},
                {"$TEST_FUNC$" : "trunc",    "$MAX_ULPS$" : "0.0f"}]
                
g_generated_header = \
"""
/*************************************************************************
//
//  This file was AUTOMATICALLY GENERATED via generate_math_unary.py
//
**************************************************************************/
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
        generated_content = generated_content.replace("$TEST_FUNC$", item['$TEST_FUNC$'])
        generated_content = generated_content.replace("$MAX_ULPS$",  item['$MAX_ULPS$'])

        # write generated file        
        generated_file_name = template_name + "_" + item['$TEST_FUNC$'] + ".cpp"
        generated_file = os.path.join(pathname, generated_file_name)
        with open(generated_file, "w") as f:
            f.write(g_generated_header)
            f.write(generated_content)
    
    return

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
# entry point
#
def main( ):
    print "generation script"
    generate( "math_unary" , g_unary_map  )
    return

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
#
#
if __name__ == '__main__':
    main( )
