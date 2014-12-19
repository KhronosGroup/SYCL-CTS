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
g_relational_binary_map = \
               [{"$TEST_FUNC$" : "isequal"       },
                {"$TEST_FUNC$" : "isnotequal"    },
                {"$TEST_FUNC$" : "isless"        },
                {"$TEST_FUNC$" : "isgreater"     },
                {"$TEST_FUNC$" : "islessequal"   },
                {"$TEST_FUNC$" : "isgreaterequal"},
                {"$TEST_FUNC$" : "islessgreater" },
                {"$TEST_FUNC$" : "isordered"     },
                {"$TEST_FUNC$" : "isunordered"   }]

g_generated_header = \
"""
/*************************************************************************
//
//  This file was AUTOMATICALLY GENERATED via generate_math_relational.py
//
**************************************************************************/
"""

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
# auto mangle a SPIR function name
#
# _Z[l][name][args]
#   l       : length of name
#   name    : function name
#   args    : 'f' (float), 'ff' (float, float), 'd' (double), ...
#
def mangle( func_name, func_args ):
    return "_Z" + str( len( func_name ) ) + func_name + func_args


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
    generate( "math_relational_binary" , g_relational_binary_map )
    return

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
#
#
if __name__ == '__main__':
    main( )
