#!/usr/bin/python
################################################################################
##
##  SYCL 1.2.1 Conformance Test Suite
##
##  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
##
################################################################################

import os
from modules import sycl_types
from modules import sycl_functions
from modules import test_generator

class runner:
    def __init__(self):
        self.base_types = ["float", "double", "cl::sycl::half", "char", "short", "int", "long int", "long long int"]
        self.var_types = ["scalar","vector"]
        self.dimensions = [1,2,3,4,8,16]
        self.unsigned = [True, False]

def write_cases_to_file(generated_test_cases, file_name):
    # Determine generator directory
    generatorDirectory = os.path.dirname(os.path.realpath(__file__))

    # Determine input file
    inputFile = os.path.join(generatorDirectory, "math_builtin.template")

    # Determine output file
    outputFile = os.path.join(generatorDirectory, file_name)

    # Load the template from the input file
    with open(inputFile, 'r') as input:
        source = input.read()
    
    newSource = source.replace("$TEST_CASES", generated_test_cases)
    
    # Write the source to the output file
    with open(outputFile, 'w+') as output:
        output.write(newSource)

def create_tests(run, types, signatures, file_name):
    expanded_signatures =  test_generator.expand_signatures(run, types, signatures)
    
    generated_test_cases = test_generator.generate_test_cases(types, expanded_signatures)
    
    write_cases_to_file(generated_test_cases, file_name)

def main():
    run = runner()

    created_types = sycl_types.create_types()

    expanded_types =  test_generator.expand_types(created_types)
    
    integer_signatures = sycl_functions.create_integer_signatures()
    create_tests(run, expanded_types, integer_signatures, "math_builtin_integer.cpp")

    common_signatures = sycl_functions.create_common_signatures()
    create_tests(run, expanded_types, common_signatures, "math_builtin_common.cpp")

    geomteric_signatures = sycl_functions.create_geometric_signatures()
    create_tests(run, expanded_types, geomteric_signatures, "math_builtin_geometric.cpp")

    relational_signatures = sycl_functions.create_relational_signatures()
    create_tests(run, expanded_types, relational_signatures, "math_builtin_relational.cpp")

    float_signatures = sycl_functions.create_float_signatures()
    create_tests(run, expanded_types, float_signatures, "math_builtin_float.cpp")

    native_signatures = sycl_functions.create_native_signatures()
    create_tests(run, expanded_types, native_signatures, "math_builtin_native.cpp")

    half_signatures = sycl_functions.create_half_signatures()
    create_tests(run, expanded_types, half_signatures, "math_builtin_half.cpp")

if __name__ == "__main__":
    main()
