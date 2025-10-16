#!/usr/bin/env python3
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
import json
import argparse
import shlex

REPORT_HEADER = """<?xml version="1.0" encoding="UTF-8"?>
<?xml-stylesheet xmlns="http://www.w3.org/1999/xhtml" type="text/xsl" href="#stylesheet"?>
<!DOCTYPE Site [
<!ATTLIST ns0:stylesheet
id ID #REQUIRED>
]>
"""


def handle_args(argv):
    """
    Handles the arguements to the script.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--cmake-exe',
        help='Name of the CMake executable',
        type=str,
        default='cmake')
    parser.add_argument(
        '-a',
        '--additional-cmake-args',
        help='Additional args to hand to CMake required by the tested implementation.',
        type=str)
    parser.add_argument(
        '-b',
        '--build-system-name',
        help='The name of the build system as known by CMake, for example \'Ninja\'.',
        type=str,
        required=True)
    parser.add_argument(
        '--build-system-args',
        help='Additional args to pass to the build system through CMake',
        type=str,
        required=True)
    parser.add_argument(
        '--build-dir',
        help='The name of the build directory to use/create',
        type=str,
        default='build',
        required=False)
    parser.add_argument(
        '--build-only',
        help='Whether to perform only a build without any testing.',
        required=False,
        action='store_true')
    parser.add_argument(
        '-e',
        '--exclude-categories',
        help='List of test categories to',
        type=str,
        required=False)
    parser.add_argument(
        '--fast',
        help='Disable full conformance mode to avoid extensive tests.',
        required=False,
        action='store_true')
    parser.add_argument(
        '--disable-deprecated-features',
        help='Disable tests for the deprecated SYCL features.',
        required=False,
        action='store_true')
    parser.add_argument(
        '--device',
        help='Select SYCL device to run CTS on. ECMAScript regular expression syntax can be used.',
        type=str,
        required=True)
    parser.add_argument(
        '-n',
        '--implementation-name',
        help='The name of the implementation to be displayed in the report.',
        type=str,
        required=True)
    parser.add_argument('--additional-ctest-args',
                        '--ctest-args',
                        help='Additional args to hand to CTest.',
                        type=str)
    parser.add_argument('--reduced-feature-set',
                        help='Test the reduced feature set instead of the full feature set.',
                        required=False,
                        action='store_true')
    parser.add_argument(
        '--run-only',
        help='Skip build step and perform only testing for already compiled tests.',
        required=False,
        action='store_true')
    parser.add_argument('--commit-hash',
                        help='Original SYCL-CTS commit hash used for the run',
                        type=str,
                        required=False)
    args = parser.parse_args(argv)

    commit_hash = args.commit_hash if args.commit_hash else 'Not specified'
    full_conformance = 'OFF' if args.fast else 'ON'
    test_deprecated_features = 'OFF' if args.disable_deprecated_features else 'ON'
    full_feature_set = 'OFF' if args.reduced_feature_set else 'ON'

    if (args.build_only and args.run_only):
        print('Fatal error: --build-only and --run-only can not be enabled '
              'together in a single script run.')
        exit(-1)

    return (args.cmake_exe, args.build_system_name, args.build_system_args,
            full_conformance, test_deprecated_features, args.exclude_categories,
            args.implementation_name, args.additional_cmake_args, args.device,
            args.additional_ctest_args, args.build_only, args.run_only,
            commit_hash, full_feature_set, args.build_dir)


def split_additional_args(additional_args):
    """
    Split any 'additional argument' parameter passed to the script into the list
    """

    # shlex doesn't support None
    if additional_args is None:
        return []
    use_posix_mode = os.name != 'nt'
    # shlex was not intended for Windows by design, with non-POSIX Unix shells
    # supported. Still it provides a Windows-compilant solution up to the
    # certain degree, details: https://bugs.python.org/issue1724822
    # The rules for quoting and escaping in non-POSIX mode are different from
    # the native Windows rules, defined for CommandLineToArgvW for example.
    # Still it makes possible to use something like "%ENV_VAR%\subdir" within
    # additional arguments
    return shlex.split(additional_args, posix=use_posix_mode)


def generate_cmake_call(cmake_exe, build_dir, build_system_name,
                        full_conformance,
                        test_deprecated_features, exclude_categories,
                        implementation_name, additional_cmake_args, device,
                        full_feature_set):
    """
    Generates a CMake call based on the input in a form accepted by
    subprocess.call().
    """
    call = [
        cmake_exe,
        '.',
        '-B' + build_dir,
        '-G' + build_system_name,
        '-DSYCL_CTS_ENABLE_FULL_CONFORMANCE=' + full_conformance,
        '-DSYCL_CTS_ENABLE_DEPRECATED_FEATURES_TESTS=' + test_deprecated_features,
        '-DSYCL_IMPLEMENTATION=' + implementation_name,
        '-DSYCL_CTS_CTEST_DEVICE=' + device,
        '-DSYCL_CTS_ENABLE_FEATURE_SET_FULL=' + full_feature_set,
        ]
    if exclude_categories is not None:
        call += ['-DSYCL_CTS_EXCLUDE_TEST_CATEGORIES=' + exclude_categories]
    call += split_additional_args(additional_cmake_args)
    return call


def generate_ctest_call(build_dir, additional_ctest_args):
    """
    Generates a CTest call based on the input in a form accepted by
    subprocess.call().
    """
    return [
        'ctest', '.', '--test-dir', build_dir, '-T', 'Test', '--no-compress-output',
        '--test-output-size-passed', '0', '--test-output-size-failed', '0'
    ] + split_additional_args(additional_ctest_args)


def generate_build_call(cmake_exe, build_dir, build_system_args):
    """
    Generates a CTest call based on the input in a form accepted by
    subprocess.call().
    """
    build_call = [ cmake_exe, '--build', build_dir, '--parallel' ]
    build_system_args = split_additional_args(build_system_args)
    if build_system_args:
        build_call.extend(['--'] + build_system_args)
    return build_call


def subprocess_call(parameter_list):
    """
    Calls subprocess.call() with the parameter list.
    Prints the invocation before doing the call.
    """
    print("subprocess.call:\n  %s" % " ".join(parameter_list))
    return subprocess.call(parameter_list)


def configure_and_run_tests(cmake_call, build_call, build_only,
                            run_only, ctest_call):
    """
    Configures the tests with cmake to produce a ninja.build file.
    Runs the generated ninja file.
    Runs ctest, overwriting any cached results.
    """

    error_code = 0

    if (not run_only):
        subprocess_call(cmake_call)

        error_code = subprocess_call(build_call)

    if (not build_only):
        error_code = subprocess_call(ctest_call)

    return error_code


def collect_info_filenames(build_dir):
    """
    Collects all the .info test result files in the Testing directory.
    Exits the program if no result files are found.
    """

    info_filenames = []

    # Get all the test results in Testing
    testing_dir = os.path.join(build_dir, 'Testing')
    for filename in os.listdir(testing_dir):
        filename_full = os.path.join(testing_dir, filename)
        if filename.endswith('.info'):
            info_filenames.append(filename_full)

    # Exit if we didn't find any test results
    if (len(info_filenames) == 0):
        print("Fatal error: Could not find any device info dumps")
        exit(-1)

    return info_filenames


def get_valid_json_info(info_filenames):
    """
    Ensures that all the .info files have the same data, then returns the
    parsed json.
    """

    reference_info = None
    for info_file in info_filenames:
        with open(info_file, 'r') as info:
            if reference_info is None:
                reference_info = info.read()
            elif info.read() != reference_info:
                print('Fatal error: mismatch in device info dumps between tests')
                exit(-1)

    return json.loads(reference_info)


def get_xml_test_results(build_dir):
    """
    Finds the xml file output by the test and returns the rool of the xml tree.
    """
    test_tag = ""
    with open(os.path.join(build_dir, "Testing", "TAG"), 'r') as tag_file:
        test_tag = tag_file.readline()[:-1]

    test_xml_file = os.path.join(build_dir, "Testing", test_tag, "Test.xml")
    test_xml_tree = ET.parse(test_xml_file)
    return test_xml_tree.getroot()


def update_xml_attribs(info_json, implementation_name, test_xml_root,
                       full_conformance, cmake_call, build_system_name,
                       build_system_call, ctest_call, test_deprecated_features,
                       commit_hash, full_feature_set):
    """
    Adds attributes to the root of the xml trees json required by the
    conformance report.
    These attributes describe the device and platform information used in the
    tests, along with the configuration and execution details of the tests.
    """

    # Set Host Device Information attribs
    test_xml_root.attrib["BuildName"] = implementation_name
    test_xml_root.attrib["PlatformName"] = info_json['platform-name']
    test_xml_root.attrib["PlatformVendor"] = info_json[
        'platform-vendor']
    test_xml_root.attrib["PlatformVersion"] = info_json[
        'platform-version']
    test_xml_root.attrib["DeviceName"] = info_json['device-name']
    test_xml_root.attrib["DeviceVendor"] = info_json['device-vendor']
    test_xml_root.attrib["DeviceVersion"] = info_json[
        'device-version']
    test_xml_root.attrib["DeviceType"] = info_json['device-type']

    # Set Device Extension Support attribs
    # TODO: Revisit this for SYCL 2020 aspects
    test_xml_root.attrib["DeviceFP16"] = info_json['device-fp16']
    test_xml_root.attrib["DeviceFP64"] = info_json['device-fp64']

    # Set Build Information attribs
    test_xml_root.attrib["CommitHash"] = commit_hash
    test_xml_root.attrib["FullConformanceMode"] = full_conformance
    test_xml_root.attrib["CMakeInput"] = ' '.join(cmake_call)
    test_xml_root.attrib["BuildSystemGenerator"] = build_system_name
    test_xml_root.attrib["BuildSystemCall"] = ' '.join(build_system_call)
    test_xml_root.attrib["CTestCall"] = ' '.join(ctest_call)
    test_xml_root.attrib["TestDeprecatedFeatures"] = test_deprecated_features
    test_xml_root.attrib["FullFeatureSet"] = full_feature_set

    return test_xml_root


def main(argv=sys.argv[1:]):

    # Parse and gather all the script args
    (cmake_exe, build_system_name, build_system_args, full_conformance,
     test_deprecated_features, exclude_categories, implementation_name,
     additional_cmake_args, device, additional_ctest_args, build_only, run_only,
     commit_hash, full_feature_set, build_dir) = handle_args(argv)

    # Generate a cmake call in a form accepted by subprocess.call()
    cmake_call = generate_cmake_call(cmake_exe, build_dir, build_system_name,
                                     full_conformance, test_deprecated_features,
                                     exclude_categories, implementation_name,
                                     additional_cmake_args, device,
                                     full_feature_set)

    # Generate a CTest call in a form accepted by subprocess.call()
    ctest_call = generate_ctest_call(build_dir, additional_ctest_args)

    build_call = generate_build_call(cmake_exe, build_dir, build_system_args)

    # Configure the build system with cmake, run the build, and run the tests.
    error_code = configure_and_run_tests(cmake_call, build_call, build_only,
                                         run_only, ctest_call)

    if build_only:
        return error_code

    # Collect the test info files, validate them and get the contents as json.
    info_filenames = collect_info_filenames(build_dir)
    info_json = get_valid_json_info(info_filenames)

    # Get the xml results and update with the necessary information.
    result_xml_root = get_xml_test_results(build_dir)
    result_xml_root = update_xml_attribs(info_json, implementation_name,
                                         result_xml_root, full_conformance,
                                         cmake_call, build_system_name,
                                         build_call, ctest_call,
                                         test_deprecated_features,
                                         commit_hash,
                                         full_feature_set)

    # Get the xml report stylesheet and add it to the results.
    stylesheet_xml_file = os.path.join(
        os.path.abspath(os.path.dirname(__file__)), "tools", "stylesheet.xml"
    )
    stylesheet_xml_tree = ET.parse(stylesheet_xml_file)
    stylesheet_xml_root = stylesheet_xml_tree.getroot()
    result_xml_root.append(stylesheet_xml_root[0])

    # Get the xml results as a string and append them to the report header.
    report = REPORT_HEADER + ET.tostring(result_xml_root).decode("utf-8")

    with open(
        os.path.join(build_dir, "conformance_report.xml"), "w"
    ) as final_conformance_report:
        final_conformance_report.write(report)

    return error_code


if __name__ == "__main__":
    main()
