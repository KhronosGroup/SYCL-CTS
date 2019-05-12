/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include <stdlib.h>

#include "test_manager.h"
#include "cmdarg.h"
#include "collection.h"
#include "printer.h"
#include "selector.h"
#include "executor.h"
#include "../tests/common/cts_selector.h"

#if defined(_MSC_VER)
extern "C" extern long __stdcall IsDebuggerPresent();
#endif

namespace sycl_cts {
namespace util {

/**
 */
test_manager::test_manager() : m_willExecute(false), m_wimpyMode(false),
  m_infoDump(false), m_infoDumpFile{""} {}

/**
 */
void test_manager::on_start() {
  // prepare the test collection for use
  get<util::collection>().prepare();
}

/**
 */
void test_manager::on_exit() {
  // flush the printer so all output appears on stdout
  get<util::printer>().finish();

// in debug mode, halt before exit
#if defined(_MSC_VER)
  if (IsDebuggerPresent() != 0) {
    get<util::printer>().print("press [enter] to exit...");
    getchar();
  }
#endif
}

/**
 */
bool test_manager::parse(const int argc, const char **args) {
  // get more convenient references to each singleton
  util::cmdarg &cmdarg = get<util::cmdarg>();
  util::printer &printer = get<util::printer>();
  util::collection &collection = get<util::collection>();
  util::selector &selector = get<util::selector>();

  // try to parse all of the command line arguments
  if (!cmdarg.parse(argc, args)) {
    // print an error message
    std::string error;
    if (cmdarg.get_last_error(error)) std::cout << error;
    return false;
  }

  // show the usage information
  if (cmdarg.find_key("--help") || cmdarg.find_key("-h")) {
    print_usage();
    return true;
  }

  // set JSON output formatting
  if (cmdarg.find_key("--json") || cmdarg.find_key("-j")) {
    printer.set_format(sycl_cts::util::printer::ejson);
  }

  // redirect all output to a file
  std::string filePath;
  if (cmdarg.get_value("--file", filePath) ||
      cmdarg.get_value("-f", filePath)) {
    if (!printer.set_file_channel(filePath.c_str())) {
      std::cout << "unable to create output file!" << std::endl;
      return false;
    } else
      std::cout << "writing output to: \'" << filePath << "\'" << std::endl;
  }

  // list all of the tests in this binary
  if (cmdarg.find_key("--list") || cmdarg.find_key("-l")) {
    collection.list();
    return true;
  }

  // load a csv file used for specifying test parameters
  std::string csvfile;
  if (cmdarg.get_value("--csv", csvfile) || cmdarg.get_value("-c", csvfile)) {
    // forward the csv file on to the collection for filtering
    if (!collection.filter_tests_csv(csvfile)) {
      std::cout << "unable to load csv file" << std::endl;
      return false;
    }
  }

  // set text output formatting
  if (cmdarg.find_key("--text")) {
    printer.set_format(sycl_cts::util::printer::etext);
  }

  // set the default sycl cts platform
  std::string platformName;
  if (cmdarg.get_value("--platform", platformName) ||
      cmdarg.get_value("-p", platformName)) {
    selector.set_default_platform(platformName);
  }

  // set the default sycl cts device
  std::string deviceName;
  if (cmdarg.get_value("--device", deviceName) ||
      cmdarg.get_value("-d", deviceName)) {
    selector.set_default_device(deviceName);
  }

  // filter by the given test name
  std::string testName;
  if (cmdarg.get_value("--test", testName)) {
    collection.filter_tests_name(testName);
  }

  // check for wimpy mode being enabled
  if (cmdarg.find_key("--wimpy") || cmdarg.find_key("-w")) {
    m_wimpyMode = true;
  }

  // check for device info dump
  std::string infoFile;
  if (cmdarg.get_value("--info-dump", infoFile) ||
      cmdarg.get_value("-i", infoFile)) {
    m_infoDump = true;
    m_infoDumpFile = infoFile;
  }

  // the test suite will try to execute tests
  m_willExecute = true;
  return true;
}

/**
 */
bool test_manager::run() {
  // execute all tests
  return get<util::executor>().run_all();
}

/**
 */
void test_manager::print_usage() {
  const char *usage = R"(
SYCL 1.2.1 CONFORMANCE TEST SUITE
Usage:
    --help      -h         Show this help message
    --json      -j         Print test results in JSON format
    --text                 Print test results in text format
    --csv       -c         CSV file for specifying tests to run
    --list      -l         List the tests compiled in this executable
    --wimpy     -w         Run with reduced test complexity (faster)
    --platform  -p [name]  Set a platform to target:
                   'host'
                   'amd'
                   'arm'
                   'intel'
                   'nvidia'
    --device    -d [name]  Select a device to target:
                   'host'
                   'opencl_cpu'
                   'opencl_gpu'
    --info-dump -i [file]  Dumps information about the device and platform
                           the tests were executed on to the file specified.
    --test         [name]  Specify a specific test to run by name, eg.
                           'unary_math_sin'
    --file      -f [path]  Redirect test output to a file

)";
  std::cout << usage << std::endl;
}

/**
 */
bool test_manager::will_execute() const { return m_willExecute; }

/**
 */
bool test_manager::wimpy_mode_enabled() const { return m_wimpyMode; }

void test_manager::dump_device_info() {
  if (m_infoDump) {
    cts_selector selector;

    auto chosenDevice = cl::sycl::device(selector);
    auto chosenPlatform = cl::sycl::platform(selector);

    std::fstream infoFile(m_infoDumpFile, std::ios::out);

    auto deviceNameStr = chosenDevice.get_info<cl::sycl::info::device::name>();
    auto deviceVendorStr =
        chosenDevice.get_info<cl::sycl::info::device::vendor>();
    auto deviceType =
        chosenDevice.get_info<cl::sycl::info::device::device_type>();
    auto deviceVersionStr =
        chosenDevice.get_info<cl::sycl::info::device::version>();
    std::string deviceTypeStr;
    switch (deviceType) {
      case cl::sycl::info::device_type::host:
        deviceTypeStr = "device_type::host";
        break;
      case cl::sycl::info::device_type::cpu:
        deviceTypeStr = "device_type::cpu";
        break;
      case cl::sycl::info::device_type::gpu:
        deviceTypeStr = "device_type::gpu";
        break;
      case cl::sycl::info::device_type::accelerator:
        deviceTypeStr = "device_type::accelerator";
        break;
      case cl::sycl::info::device_type::custom:
        deviceTypeStr = "device_type::custom";
        break;
      case cl::sycl::info::device_type::automatic:
        deviceTypeStr = "device_type::automatic";
        break;
      case cl::sycl::info::device_type::all:
        deviceTypeStr = "device_type::all";
        break;
    };
    auto doesDeviceSupportHalf = chosenDevice.has_extension("cl_khr_fp_16")
                                     ? "Supported"
                                     : "Not Supported";
    auto doesDeviceSupportDouble = chosenDevice.has_extension("cl_khr_fp64")
                                       ? "Supported"
                                       : "Not Supported";
    auto doesDeviceSupportBaseAtomics =
        chosenDevice.has_extension("cl_khr_int64_base_atomics")
            ? "Supported"
            : "Not Supported";
    auto doesDeviceSupportExtendedAtomics =
        chosenDevice.has_extension("cl_khr_int64_extended_atomics")
            ? "Supported"
            : "Not Supported";
    auto doesDeviceSupport3DImageWrites =
        chosenDevice.has_extension("cl_khr_3d_image_writes") ? "Supported"
                                                             : "Not Supported";
    auto platformNameStr =
        chosenPlatform.get_info<cl::sycl::info::platform::name>();
    auto platformVendorStr =
        chosenPlatform.get_info<cl::sycl::info::platform::vendor>();
    auto platformVersionStr =
        chosenPlatform.get_info<cl::sycl::info::platform::version>();

    infoFile << "{\"device-name\": \"" << deviceNameStr
             << "\", \"device-vendor\": \"" << deviceVendorStr
             << "\", \"device-type\": \"" << deviceTypeStr
             << "\", \"device-version\": \"" << deviceVersionStr
             << "\", \"device-fp16\": \"" << doesDeviceSupportHalf
             << "\", \"device-fp64\": \"" << doesDeviceSupportDouble
             << "\", \"device-int64-base\": \"" << doesDeviceSupportBaseAtomics
             << "\", \"device-int64-extended\": \""
             << doesDeviceSupportExtendedAtomics
             << "\", \"device-3d-writes\": \"" << doesDeviceSupport3DImageWrites
             << "\", \"platform-name\": \"" << platformNameStr
             << "\", \"platform-vendor\": \"" << platformVendorStr
             << "\", \"platform-version\": \"" << platformVersionStr << "\"}";
  }
}

}  // namespace util
}  // namespace sycl_cts
