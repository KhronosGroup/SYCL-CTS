==============================================
SYCL 1.2.1 Conformance Test Suite (10/10/2017)
==============================================

Introduction
------------

    This conformance test suite is divided into two parts, a binary
    executable containing all of the tests and a python user interface.
    It is intended that the user only runs the test suite via the python
    frontend.

    The test executable must be compiled before running the test suite, a
    step which requires CMake.  Executables are produced for each category
    and one 'fat' executable that contains the tests from all categories.


Compilation and execution prerequisites
---------------------------------------

    - An implementation of SYCL
    - A conformant implementation of OpenCL
    - Python 2.7
    - GCC 4.8.2 or Visual C++ 2015 Update 3
    - CMake 3.2

    The SYCL conformance test suite assumes that an underlying OpenCL
    implementation has passed the Khronos OpenCL CTS.


Building on Linux
-----------------

    - Checkout the SYCL test suite repository.

    - Run CMake
        - point the source directory to the root of the sycl-cts folder
        - configure the generator to use makefiles
        - set the build directory to a 'build' folder inside the sycl-cts folder
        - set the CMake parameters

    - Run make inside the build folder

        After compilation, the test executables will be placed in the
        sycl-cts 'build/bin' directory.  The 'test_all' executable
        contains all tests in the suite.


Building on Windows
-------------------

    - Checkout the SYCL test suite repository.

    - Run CMake
		- point the source directory to the root of the sycl-cts folder
		- select "Visual Studio 14 2015 Win64" generator
		- set the build directory to a 'build' folder inside the sycl-cts folder
		- set the CMake parameters

    - Open the generated solution in Visual Studio 2015 and rebuild all

        After compilation, the test executables will be placed in the
        sycl-cts 'build/bin' directory.  The 'test_all' executable
        contains all tests in the suite.

CMake flags
-----------

    When configuring CMake, it is possible to use these flags:

    - COMPUTECPP_INSTALL_DIR
        - Required only for ComputeCpp.
    - SYCL_CTS_TEST_FILTER
        - Specify which filter to use when building tests.
    - HOST_COMPILER_FLAGS
        - Flags that will be passed to the host compiler.
    - DEVICE_COMPILER_FLAGS
        - Flags that will be passed to the device compiler.


Launching the test suite
------------------------

    The SYCL test suite can be launched via the following command:

        $ python runtests.py --help
        
        usage: runtests.py [-h] [-b BINPATH] [--csvpath CSVPATH] [--list]
                           [-j JUNIT] [-p {host,intel,amd}]
                           [-d {host,opencl_cpu,opencl_gpu}]

        Khronos SYCL CTS

        optional arguments:

          -h, --help            show this help message and exit
          -b BINPATH, --binpath BINPATH
                                specify path to the cts executable file
          --csvpath CSVPATH     specify path to csv file for filtering tests
          --list                list all tests in a test binary
          -j JUNIT, --junit JUNIT
                                specify output path for a junit xml file
          -p PLATFORM, --platform PLATFORM
                                The platform to run on (where PLATFORM can be
                                host, intel, amd)
          -d DEVICE, --device DEVICE
                                The device to run on (where DEVICE can be host,
                                opencl_cpu, opencl_gpu)
 
    The '--binpath' argument is mandatory and must point to one of the CTS
    test executables built in the previous step.

    An optional CSV file can be given which can be used to narrow the range
    of tests that will be executed.

    The filters work using the principal of partial string matching.  Any
    test in a CTS executable with a name that begins with one of the items
    in CSV file will be scheduled to be run.  Those tests that don't match
    will not be run.

	In the future the CSV file will also be used to specify timeout values
	on a per-test basis.

    The '--list' argument can be used to examine all of the tests that are
    stored in a test executable.  For instance:
    
        $ python runtests.py -b build\bin\test_context.exe --list

        3 tests in executable

          . context_api

          . context_constructors

          . context_getinfo
    
    Passing the '--junit' option will output test results in junit format
    when the test suit has finished executing.

    The '--platform' argument can be used to specify which platform to run the
    tests on.

    The '--device' argument can be used to specify which device to run the
    tests on.
    
    The following command will start a typical test run:

        $ python runtests.py --binpath tests/common/test_all

    During testing any fails will be reported with details about the failure.
    The following failure importantly shows the source file containing the
    test and the line number that signalled the failure.

        platform_api:

         ?   note: sycl exception caught
         ?   note: what - Failed to get platform information.
         + result: fail
         !   file: ../../tests/platform/platform_api.cpp
         !  built: Aug 22 2017, 18:06:45
         !   line: 96

    After the test suit is finished a summary is produced helping programmers
    quickly identify failures and conformance rate.

        16 tests ran in total
         - passed : 13
         - failed : 1
           + platform_api
         - skipped: 2
         - 81% pass rate

Producing a conformance report
------------------------------

    The SYCL test suite can produce a conformance report using the
    run_conformance_test.py script.

    run_conformance_test.py has the following arguments.
    -b, -c, -f and -n are required to run the script.

    -h, --help            show this help message and exit
    -a ADDITIONAL_CMAKE_ARGS, --additional-cmake-args ADDITIONAL_CMAKE_ARGS
                          Additional args to hand to CMake required by the
                          tested implementation.
    -b BUILD_SYSTEM_NAME, --build-system-name BUILD_SYSTEM_NAME
                          The name of the build system as known by CMake, for
                          example 'Ninja'.
    -c BUILD_SYSTEM_CALL, --build-system-call BUILD_SYSTEM_CALL
                          The call to the used build system.
    -f CONFORMANCE_FILTER, --conformance-filter CONFORMANCE_FILTER
                          The conformance filter to use.
    --host-platform-name HOST_PLATFORM_NAME
                          The name of the host platform to test on.
    --host-device-name HOST_DEVICE_NAME
                          The name of the host device to test on.
    --opencl-platform-name OPENCL_PLATFORM_NAME
                          The name of the opencl platform to test on.
    --opencl-device-name OPENCL_DEVICE_NAME
                          The name of the opencl device to test on.
    -n IMPLEMENTATION_NAME, --implementation-name IMPLEMENTATION_NAME
                          The name of the implementation to be displayed in the
                          report.

    --build-system-name is mandatory and must be the name of the CMake generator
     used for the build.
    --build-system-call is mandatory and must be the command line call to the
    build generated by CMake.
    --conformance-filter is mandatory and must be a path to the core.csv
    conformance filter when submitting for conformance.
    --implementation-name is mandatory and must be the name of the tested
    implementation to be displayed on the conformance report.
    --host-platform-name is mandatory and must be the host name used to invoke
    a host platform test.
    This will be used for the CMake variable ${host_platform_name}.
    --host-device-name is mandatory and must be the host name used to invoke
    a host device test.
    This will be used for the CMake variable ${host_device_name}.
    --opencl-platform-name is mandatory and must be the name of the OpenCL
    platform tested on.
    This will be used for the CMake variable ${opencl_platform_name}.
    --opencl-device-name is mandatory and must be the name of the OpenCL
    device tested on.
    This will be used for the CMake variable ${opencl_device_name}.

    The script will produce an xml conformance report detailing the tested
    implementation, host system, host device, opencl device, build configuration
    and the results of each test.

    This report should be packaged with the run tests and sent to Khronos for
    conformance submission.

    Conformance submission requires the use of the core.csv filter.
