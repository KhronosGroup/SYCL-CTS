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

g_binary_map = [{"$ARITHMETIC_OPERATION$" : "plus",       		"$ARITHMETIC_OPERATOR$" : "+",		"$COMPOUND_OPERATION$" : "0"},
                {"$ARITHMETIC_OPERATION$" : "minus",      		"$ARITHMETIC_OPERATOR$" : "-", 		"$COMPOUND_OPERATION$" : "0"},
                {"$ARITHMETIC_OPERATION$" : "multiply",  		"$ARITHMETIC_OPERATOR$" : "*", 		"$COMPOUND_OPERATION$" : "0"},
                {"$ARITHMETIC_OPERATION$" : "divide",			"$ARITHMETIC_OPERATOR$" : "/", 		"$COMPOUND_OPERATION$" : "0"},
                {"$ARITHMETIC_OPERATION$" : "plusequal",      	"$ARITHMETIC_OPERATOR$" : "+=", 	"$COMPOUND_OPERATION$" : "1"},
                {"$ARITHMETIC_OPERATION$" : "minusequal",      	"$ARITHMETIC_OPERATOR$" : "-=", 	"$COMPOUND_OPERATION$" : "1"},
                {"$ARITHMETIC_OPERATION$" : "multiplyequal",	"$ARITHMETIC_OPERATOR$" : "*=", 	"$COMPOUND_OPERATION$" : "1"},
                {"$ARITHMETIC_OPERATION$" : "divideequal",     	"$ARITHMETIC_OPERATOR$" : "/=", 	"$COMPOUND_OPERATION$" : "1"},
                {"$ARITHMETIC_OPERATION$" : "bitwiseor",      	"$ARITHMETIC_OPERATOR$" : "|",  	"$COMPOUND_OPERATION$" : "0"},
                {"$ARITHMETIC_OPERATION$" : "bitwiseand",      	"$ARITHMETIC_OPERATOR$" : "&",  	"$COMPOUND_OPERATION$" : "0"},
                {"$ARITHMETIC_OPERATION$" : "bitwisenot",	    "$ARITHMETIC_OPERATOR$" : "~",  	"$COMPOUND_OPERATION$" : "0"},
                {"$ARITHMETIC_OPERATION$" : "bitwisexor",	    "$ARITHMETIC_OPERATOR$" : "^",  	"$COMPOUND_OPERATION$" : "0"},
                {"$ARITHMETIC_OPERATION$" : "bitwiseorequal",  	"$ARITHMETIC_OPERATOR$" : "|=",  	"$COMPOUND_OPERATION$" : "1"},
                {"$ARITHMETIC_OPERATION$" : "bitwiseandequal", 	"$ARITHMETIC_OPERATOR$" : "&=",  	"$COMPOUND_OPERATION$" : "1"},
                {"$ARITHMETIC_OPERATION$" : "bitwisenotequal",  "$ARITHMETIC_OPERATOR$" : "~=",  	"$COMPOUND_OPERATION$" : "1"},
                {"$ARITHMETIC_OPERATION$" : "bitwisexorequal",  "$ARITHMETIC_OPERATOR$" : "^=",  	"$COMPOUND_OPERATION$" : "1"}]

                
g_generated_header = \
"""
/************************************************************************************
//
//  This file was AUTOMATICALLY GENERATED via generate_vector_arithmetic_operators.py
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
        generated_content = generated_content.replace("$ARITHMETIC_OPERATION$", item['$ARITHMETIC_OPERATION$'])
        generated_content = generated_content.replace("$ARITHMETIC_OPERATOR$", item['$ARITHMETIC_OPERATOR$'])
        generated_content = generated_content.replace("$COMPOUND_OPERATION$",  item['$COMPOUND_OPERATION$'])

        # write generated file        
        generated_file_name = template_name + "_" + item['$ARITHMETIC_OPERATION$'] + ".cpp"
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
    generate( "vector_arithmetic_operators", g_binary_map )
    return

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
#
#
if __name__ == '__main__':
    main( )
