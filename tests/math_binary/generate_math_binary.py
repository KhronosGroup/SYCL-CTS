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

g_binary_map = [{"$TEST_FUNC$" : "pow",       "$MAX_ULPS$" : "16.0f"},
                {"$TEST_FUNC$" : "fdim",      "$MAX_ULPS$" : "0.0f" },
                {"$TEST_FUNC$" : "copysign",  "$MAX_ULPS$" : "0.0f" },
                {"$TEST_FUNC$" : "atan2pi",   "$MAX_ULPS$" : "6.0f" },
                {"$TEST_FUNC$" : "fmax",      "$MAX_ULPS$" : "0.0f" },
                {"$TEST_FUNC$" : "fmin",      "$MAX_ULPS$" : "0.0f" },
                {"$TEST_FUNC$" : "fmod",      "$MAX_ULPS$" : "0.0f" },
                {"$TEST_FUNC$" : "hypot",     "$MAX_ULPS$" : "4.0f" },
                {"$TEST_FUNC$" : "maxmag",    "$MAX_ULPS$" : "0.0f" },
                {"$TEST_FUNC$" : "minmag",    "$MAX_ULPS$" : "0.0f" },
                {"$TEST_FUNC$" : "atan2",     "$MAX_ULPS$" : "6.0f" },
                {"$TEST_FUNC$" : "powr",      "$MAX_ULPS$" : "16.0f"},
                {"$TEST_FUNC$" : "remainder", "$MAX_ULPS$" : "0.0f" }]
#"""

g_generated_header = \
"""
/*************************************************************************
//
//  This file was AUTOMATICALLY GENERATED via generate_math_binary.py
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
    generate( "math_binary", g_binary_map )
    return

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
#
#
if __name__ == '__main__':
    main( )
