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
        self.base_types = ["float", "double", "char", "short", "int", "long int", "long long int", "int8_t", "int16_t", "int32_t", "int64_t", "cl::sycl::half"]
        self.var_types = ["scalar","vector"]
        self.dimensions = [1,2,3,4,8,16]
        self.unsigned = [True, False]

def contains_base_type(sig, types, base_type):
    data = sig.arg_types[:]
    data.append(sig.ret_type)
    for dt in data:
        for tp in types[dt].keys():
            if types[dt][tp].base_type == base_type:
                return True
    return False

def write_cases_to_file(generated_test_cases, file_name, extension=None):
    # Determine generator directory
    generatorDirectory = os.path.dirname(os.path.realpath(__file__))

    # Determine input file
    inputFile = os.path.join(generatorDirectory, "math_builtin.template")

    # Determine output file
    outputFile = os.path.join(generatorDirectory, file_name)

    # Load the template from the input file
    with open(inputFile, 'r') as input:
        source = input.read()
    
    # Execute the tests if the extensions are supported by target device.
    if extension:
        checkPoint = "\n\nif(makeQueueOnce().get_device().has_extension(\"" + extension + "\")){\n"
        generated_test_cases = checkPoint + generated_test_cases + "\n\n}"

    newSource = source.replace("$TEST_CASES", generated_test_cases)
    newSource = newSource.replace("$math_builtins", file_name[0:-4])
    
    # Write the source to the output file
    with open(outputFile, 'w+') as output:
        output.write(newSource)

def create_tests(test_id, run, types, signatures, file_name):
    expanded_signatures =  test_generator.expand_signatures(run, types, signatures)
    
    # Extensions should be placed on separate files.
    base_signatures = []
    half_signatures = []
    double_signatures = []
    for sig in expanded_signatures:
        if contains_base_type(sig, types, "double"):
            double_signatures.append(sig)
            continue
        if contains_base_type(sig, types, "cl::sycl::half"):
            half_signatures.append(sig)
            continue
        base_signatures.append(sig)

    if base_signatures:
        generated_base_test_cases = test_generator.generate_test_cases(test_id, types, base_signatures)
        write_cases_to_file(generated_base_test_cases, file_name + ".cpp")

    if half_signatures:
        generated_half_test_cases = test_generator.generate_test_cases(test_id + 300000, types, half_signatures)
        write_cases_to_file(generated_half_test_cases, file_name + "_half.cpp", "cl_khr_fp16")

    if half_signatures:
        generated_double_test_cases = test_generator.generate_test_cases(test_id + 600000, types, double_signatures)
        write_cases_to_file(generated_double_test_cases, file_name + "_double.cpp", "cl_khr_fp64")
    
def main():
    run = runner()

    created_types = sycl_types.create_types()

    expanded_types =  test_generator.expand_types(created_types)
    
    integer_signatures = sycl_functions.create_integer_signatures()
    create_tests(0, run, expanded_types, integer_signatures, "math_builtin_integer")

    common_signatures = sycl_functions.create_common_signatures()
    create_tests(1000000, run, expanded_types, common_signatures, "math_builtin_common")

    geomteric_signatures = sycl_functions.create_geometric_signatures()
    create_tests(2000000, run, expanded_types, geomteric_signatures, "math_builtin_geometric")

    relational_signatures = sycl_functions.create_relational_signatures()
    create_tests(3000000, run, expanded_types, relational_signatures, "math_builtin_relational")

    float_signatures = sycl_functions.create_float_signatures()
    create_tests(4000000, run, expanded_types, float_signatures, "math_builtin_float")

    native_signatures = sycl_functions.create_native_signatures()
    create_tests(5000000,run, expanded_types, native_signatures, "math_builtin_native")

    half_signatures = sycl_functions.create_half_signatures()
    create_tests(6000000, run, expanded_types, half_signatures, "math_builtin_half")

if __name__ == "__main__":
    main()
