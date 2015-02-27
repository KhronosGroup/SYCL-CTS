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
g_ternary_map = [{"$TEST_FUNC$" : "clamp",       "$MAX_ULPS$" : "4.0f"}]
"""
                {"$TEST_FUNC$" : "fma",         "$MAX_ULPS$" : "4.0f"},
                {"$TEST_FUNC$" : "mad",         "$MAX_ULPS$" : "4.0f"},
                {"$TEST_FUNC$" : "remquo",      "$MAX_ULPS$" : "4.0f"},
                {"$TEST_FUNC$" : "mix",         "$MAX_ULPS$" : "4.0f"},
                {"$TEST_FUNC$" : "smoothstep",  "$MAX_ULPS$" : "4.0f"}]
"""

g_generated_header = \
"""
/*************************************************************************
//
//  This file was AUTOMATICALLY GENERATED via generate_math_ternary.py
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
    generate( "math_ternary" , g_ternary_map  )
    return

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
#
#
if __name__ == '__main__':
    main( )
