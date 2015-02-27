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
g_func_map = [{"$TEST_FUNC$" : "abs_diff"},
              {"$TEST_FUNC$" : "add_sat" },
              {"$TEST_FUNC$" : "hadd"    },
              {"$TEST_FUNC$" : "rhadd"   },
              {"$TEST_FUNC$" : "max"     },
              {"$TEST_FUNC$" : "min"     },
              {"$TEST_FUNC$" : "mul_hi"  },
              {"$TEST_FUNC$" : "rotate"  },
              {"$TEST_FUNC$" : "sub_sat" },
              {"$TEST_FUNC$" : "upsample"},
              {"$TEST_FUNC$" : "mul24"   }]

g_generated_header = \
"""
/*************************************************************************
//
//  This file was AUTOMATICALLY GENERATED via generate_integer_unary.py
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
    generate( "math_integer_binary" , g_func_map  )
    return

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
#
#
if __name__ == '__main__':
    main( )
