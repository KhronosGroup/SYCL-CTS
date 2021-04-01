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
        help=
        'Additional args to hand to CMake required by the tested implementation.',
        type=str)
    parser.add_argument(
        '-b',
        '--build-system-name',
        help=
        'The name of the build system as known by CMake, for example \'Ninja\'.',
        type=str,
        required=True)
    parser.add_argument(
        '-c',
        '--build-system-call',
        help='The call to the used build system.',
        type=str,
        required=True)
    parser.add_argument(
        '--build-only',
        help='Whether to perform only a build without any testing.',
        required=False,
        action='store_true')
    parser.add_argument(
        '-f',
        '--conformance-filter',
        help='The conformance filter to use.',
        type=str,
        required=True)
    parser.add_argument(
        '--fast',
        help='Disable full conformance mode to avoid extensive tests.',
        required=False,
        action='store_true')
    parser.add_argument(
        '--host-platform-name',
        help='The name of the host platform to test on.',
        type=str,
        required=True)
    parser.add_argument(
        '--host-device-name',
        help='The name of the host device to test on.',
        type=str,
        required=True)
    parser.add_argument(
        '--opencl-platform-name',
        help='The name of the opencl platform to test on.',
        type=str,
        required=True)
    parser.add_argument(
        '--opencl-device-name',
        help='The name of the opencl device to test on.',
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
    args = parser.parse_args(argv)

    host_names = (args.host_platform_name, args.host_device_name)
    opencl_names = (args.opencl_platform_name, args.opencl_device_name)
    full_conformance = 'OFF' if args.fast else 'ON'

    return (args.cmake_exe, args.build_system_name, args.build_system_call,
            full_conformance, args.conformance_filter, args.implementation_name,
            args.additional_cmake_args, host_names, opencl_names,
            args.additional_ctest_args, args.build_only)

def split_additional_args(additional_args):
    """
    Split any 'additional argument' parameter passed to the script into the list
    """

    # shlex doesn't support None
    if additional_args is None:
        return []
    return shlex.split(shlex.quote(additional_args))

def generate_cmake_call(cmake_exe, build_system_name, full_conformance,
                        conformance_filter, additional_cmake_args, host_names,
                        opencl_names):
    """
    Generates a CMake call based on the input in a form accepted by
    subprocess.call().
    """
    return [
        cmake_exe,
        '..',
        '-G' + build_system_name,
        '-DSYCL_CTS_ENABLE_FULL_CONFORMANCE=' + full_conformance,
        '-DSYCL_CTS_TEST_FILTER=' + conformance_filter,
        '-Dhost_platform_name=' + host_names[0],
        '-Dhost_device_name=' + host_names[1],
        '-Dopencl_platform_name=' + opencl_names[0],
        '-Dopencl_device_name=' + opencl_names[1],
    ] + split_additional_args(additional_cmake_args)


def generate_ctest_call(additional_ctest_args):
    """
    Generates a CTest call based on the input in a form accepted by
    subprocess.call().
    """
    return [
        'ctest', '.', '-T', 'Test', '--no-compress-output',
        '--test-output-size-passed', '0', '--test-output-size-failed', '0'
    ] + split_additional_args(additional_ctest_args)


def subprocess_call(parameter_list):
    """
    Calls subprocess.call() with the parameter list.
    Prints the invocation before doing the call.
    """
    print("subprocess.call:\n  %s" % " ".join(parameter_list))
    return subprocess.call(parameter_list)


def configure_and_run_tests(cmake_call, build_system_call, build_only,
                            ctest_call):
    """
    Configures the tests with cmake to produce a ninja.build file.
    Runs the generated ninja file.
    Runs ctest, overwriting any cached results.
    """

    build_system_call = build_system_call.split()

    subprocess_call(cmake_call)
    error_code = subprocess_call(build_system_call)
    if (not build_only):
        error_code = subprocess_call(ctest_call)
    return error_code


def collect_info_filenames():
    """
    Collects all the .info test result files in the Testing directory.
    Exits the program if no result files are found.
    """

    host_info_filenames = []
    opencl_info_filenames = []

    # Get all the test results in Testing
    for filename in os.listdir('Testing'):
        filename_full = os.path.join('Testing', filename)
        if filename.endswith('host.info'):
            host_info_filenames.append(filename_full)
        if filename.endswith('opencl.info'):
            opencl_info_filenames.append(filename_full)

    # Exit if we didn't find any test results
    if (len(host_info_filenames) == 0 or len(opencl_info_filenames) == 0):
        print("Fatal error: couldn't find any test result files")
        exit(-1)

    return host_info_filenames, opencl_info_filenames


def get_valid_host_json_info(host_info_filenames):
    """
    Ensures that all the host.info files have the same data, then returns the
    parsed json.
    """

    reference_host_info = None
    for host_info_file in host_info_filenames:
        with open(host_info_file, 'r') as host_info:
            if reference_host_info is None:
                reference_host_info = host_info.read()
            elif host_info.read() != reference_host_info:
                print('Fatal error: mismatch in host info between tests')
                exit(-1)

    return json.loads(reference_host_info)


def get_valid_opencl_json_info(opencl_info_filenames):
    """
    Ensures that all the opencl.info files have the same data, then returns the
    parsed json.
    """

    reference_opencl_info = None
    for opencl_info_file in opencl_info_filenames:
        with open(opencl_info_file, 'r') as opencl_info:
            if reference_opencl_info is None:
                reference_opencl_info = opencl_info.read()
            elif opencl_info.read() != reference_opencl_info:
                print('Fatal error: mismatch in OpenCL info between tests')
                exit(-1)

    # Some drivers add \x00 to their output.
    # We have to remove this to parse the json
    reference_opencl_info = reference_opencl_info.replace('\x00', '')
    return json.loads(reference_opencl_info)


def get_xml_test_results():
    """
    Finds the xml file output by the test and returns the rool of the xml tree.
    """
    test_tag = ""
    with open(os.path.join("Testing", "TAG"), 'r') as tag_file:
        test_tag = tag_file.readline()[:-1]

    test_xml_file = os.path.join("Testing", test_tag, "Test.xml")
    test_xml_tree = ET.parse(test_xml_file)
    return test_xml_tree.getroot()


def update_xml_attribs(host_info_json, opencl_info_json, implementation_name,
                       test_xml_root, full_conformance, cmake_call,
                       build_system_name, build_system_call, ctest_call):
    """
    Adds attributes to the root of the xml trees json required by the
    conformance report.
    These attributes describe the device and platform information used in the
    tests, along with the configuration and execution details of the tests.
    """

    # Set Host Device Information attribs
    test_xml_root.attrib["BuildName"] = implementation_name
    test_xml_root.attrib["HostPlatformName"] = host_info_json['platform-name']
    test_xml_root.attrib["HostPlatformVendor"] = host_info_json[
        'platform-vendor']
    test_xml_root.attrib["HostPlatformVersion"] = host_info_json[
        'platform-version']
    test_xml_root.attrib["HostDeviceName"] = host_info_json['device-name']
    test_xml_root.attrib["HostDeviceVendor"] = host_info_json['device-vendor']
    test_xml_root.attrib["HostDeviceVersion"] = host_info_json[
        'device-version']
    test_xml_root.attrib["HostDeviceType"] = host_info_json['device-type']

    # Set Host Device Extension Support attribs
    test_xml_root.attrib["HostDeviceFP16"] = host_info_json['device-fp16']
    test_xml_root.attrib["HostDeviceFP64"] = host_info_json['device-fp64']
    test_xml_root.attrib["HostDeviceInt64Base"] = host_info_json[
        'device-int64-base']
    test_xml_root.attrib["HostDeviceInt64Extended"] = host_info_json[
        'device-int64-extended']
    test_xml_root.attrib["HostDevice3DWrites"] = host_info_json[
        'device-3d-writes']

    # Set OpenCL Device Information attribs
    test_xml_root.attrib["OpenCLPlatformName"] = opencl_info_json[
        'platform-name']
    test_xml_root.attrib["OpenCLPlatformVendor"] = opencl_info_json[
        'platform-vendor']
    test_xml_root.attrib["OpenCLPlatformVersion"] = opencl_info_json[
        'platform-version']
    test_xml_root.attrib["OpenCLDeviceName"] = opencl_info_json['device-name']
    test_xml_root.attrib["OpenCLDeviceVendor"] = opencl_info_json[
        'device-vendor']
    test_xml_root.attrib["OpenCLDeviceVersion"] = opencl_info_json[
        'device-version']

    # Set OpenCL Device Extension Support attribs
    test_xml_root.attrib["OpenCLDeviceType"] = opencl_info_json['device-type']
    test_xml_root.attrib["OpenCLDeviceFP16"] = opencl_info_json['device-fp16']
    test_xml_root.attrib["OpenCLDeviceFP64"] = opencl_info_json['device-fp64']
    test_xml_root.attrib["OpenCLDeviceInt64Base"] = opencl_info_json[
        'device-int64-base']
    test_xml_root.attrib["OpenCLDeviceInt64Extended"] = opencl_info_json[
        'device-int64-extended']
    test_xml_root.attrib["OpenCLDevice3DWrites"] = opencl_info_json[
        'device-3d-writes']

    # Set Build Information attribs
    test_xml_root.attrib["FullConformanceMode"] = full_conformance
    test_xml_root.attrib["CMakeInput"] = ' '.join(cmake_call)
    test_xml_root.attrib["BuildSystemGenerator"] = build_system_name
    test_xml_root.attrib["BuildSystemCall"] = build_system_call
    test_xml_root.attrib["CTestCall"] = ' '.join(ctest_call)

    return test_xml_root


def main(argv=sys.argv[1:]):

    # Parse and gather all the script args
    (cmake_exe, build_system_name, build_system_call, full_conformance,
     conformance_filter, implementation_name, additional_cmake_args, host_names,
     opencl_names, additional_ctest_args, build_only) = handle_args(argv)

    # Generate a cmake call in a form accepted by subprocess.call()
    cmake_call = generate_cmake_call(cmake_exe, build_system_name,
                                     full_conformance, conformance_filter,
                                     additional_cmake_args, host_names,
                                     opencl_names)

    # Generate a CTest call in a form accepted by subprocess.call()
    ctest_call = generate_ctest_call(additional_ctest_args)

    # Make a build directory if required and enter it
    if not os.path.isdir('build'):
        os.mkdir('build')
    os.chdir('build')

    # Configure the build system with cmake, run the build, and run the tests.
    error_code = configure_and_run_tests(cmake_call, build_system_call,
                                         build_only, ctest_call)

    if build_only:
        return error_code

    # Collect the test info files, validate them and get the contents as json.
    host_info_filenames, opencl_info_filenames = collect_info_filenames()
    host_info_json = get_valid_host_json_info(host_info_filenames)
    opencl_info_json = get_valid_opencl_json_info(opencl_info_filenames)

    # Get the xml results and update with the necessary information.
    result_xml_root = get_xml_test_results()
    result_xml_root = update_xml_attribs(host_info_json, opencl_info_json,
                                         implementation_name, result_xml_root,
                                         full_conformance, cmake_call,
                                         build_system_name, build_system_call,
                                         ctest_call)

    # Get the xml report stylesheet and add it to the results.
    stylesheet_xml_file = os.path.join("..", "tools", "stylesheet.xml")
    stylesheet_xml_tree = ET.parse(stylesheet_xml_file)
    stylesheet_xml_root = stylesheet_xml_tree.getroot()
    result_xml_root.append(stylesheet_xml_root[0])

    # Get the xml results as a string and append them to the report header.
    report = REPORT_HEADER + ET.tostring(result_xml_root).decode("utf-8")

    with open("conformance_report.xml", 'w') as final_conformance_report:
        final_conformance_report.write(report)

    return error_code


if __name__ == "__main__":
    main()
