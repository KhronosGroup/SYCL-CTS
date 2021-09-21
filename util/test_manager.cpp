/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include <stdlib.h>

#include "../tests/common/cts_selector.h"
#include "cmdarg.h"
#include "collection.h"
#include "executor.h"
#include "printer.h"
#include "test_manager.h"

#if defined(_MSC_VER)
extern "C" extern long __stdcall IsDebuggerPresent();
#endif

namespace sycl_cts {
namespace util {

/**
 */
test_manager::test_manager()
    : m_willExecute(false),
      m_wimpyMode(false),
      m_infoDump(false),
      m_infoDumpFile{""} {}

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

static std::string get_device_type_str(const sycl::device &d) {
  const auto deviceType = d.get_info<sycl::info::device::device_type>();
  switch (deviceType) {
    case sycl::info::device_type::host:
      return "host";
    case sycl::info::device_type::cpu:
      return "cpu";
    case sycl::info::device_type::gpu:
      return "gpu";
    case sycl::info::device_type::accelerator:
      return "accelerator";
    case sycl::info::device_type::custom:
      return "custom";
    case sycl::info::device_type::automatic:
      return "automatic";
    case sycl::info::device_type::all:
      return "all";
    default:
      assert(false);
      return "(unknown)";
  };
}

static void list_devices() {
  const auto all_devices = sycl::device::get_devices();
  const auto cts_device = cts_selector{}.select_device();

  if (all_devices.empty()) {
    printf("No devices available.\n");
    return;
  }

  printf("%zu devices available (> = currently selected):\n",
         all_devices.size());
  printf("  %-12s %s\n", "Type", "Platform / Device");
  for (auto &d : all_devices) {
    const auto deviceType = get_device_type_str(d);
    const auto deviceName = d.get_info<sycl::info::device::name>();
    const auto platformName =
        d.get_platform().get_info<sycl::info::platform::name>();
    printf("%c ", d == cts_device ? '>' : ' ');
    printf("%-12s %s / %s\n", deviceType.c_str(), platformName.c_str(),
           deviceName.c_str());
  }
}

/**
 */
bool test_manager::parse(const int argc, const char **args) {
  // get more convenient references to each singleton
  util::cmdarg &cmdarg = get<util::cmdarg>();
  util::printer &printer = get<util::printer>();
  util::collection &collection = get<util::collection>();

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

  // set the default sycl cts device
  std::string deviceName;
  if (cmdarg.get_value("--device", deviceName) && !deviceName.empty()) {
    device_regex = std::regex(deviceName);
  }

  // list all available devices
  // this has to happen after --device has been parsed
  if (cmdarg.find_key("--list-devices")) {
    list_devices();
    return true;
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
SYCL 2020 CONFORMANCE TEST SUITE
Usage:
    --help      -h         Show this help message
    --json      -j         Print test results in JSON format
    --text                 Print test results in text format
    --csv       -c         CSV file for specifying tests to run
    --list      -l         List the tests compiled in this executable
    --wimpy     -w         Run with reduced test complexity (faster)
    --device       <name>  Select SYCL device to run CTS on.
                           ECMAScript regular expression syntax can be used.
    --list-devices         List all available devices.
    --info-dump -i <file>  Dumps information about the device and platform
                           the tests were executed on to the file specified.
    --test         <name>  Specify a specific test to run by name, eg.
                           'unary_math_sin'
    --file      -f <path>  Redirect test output to a file

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

    auto chosenDevice = sycl::device(selector);
    auto chosenPlatform = sycl::platform(selector);

    std::fstream infoFile(m_infoDumpFile, std::ios::out);

    auto deviceNameStr = chosenDevice.get_info<sycl::info::device::name>();
    auto deviceVendorStr =
        chosenDevice.get_info<sycl::info::device::vendor>();
    auto deviceType =
        chosenDevice.get_info<sycl::info::device::device_type>();
    auto deviceVersionStr =
        chosenDevice.get_info<sycl::info::device::version>();
    std::string deviceTypeStr;
    switch (deviceType) {
      case sycl::info::device_type::host:
        deviceTypeStr = "device_type::host";
        break;
      case sycl::info::device_type::cpu:
        deviceTypeStr = "device_type::cpu";
        break;
      case sycl::info::device_type::gpu:
        deviceTypeStr = "device_type::gpu";
        break;
      case sycl::info::device_type::accelerator:
        deviceTypeStr = "device_type::accelerator";
        break;
      case sycl::info::device_type::custom:
        deviceTypeStr = "device_type::custom";
        break;
      case sycl::info::device_type::automatic:
        deviceTypeStr = "device_type::automatic";
        break;
      case sycl::info::device_type::all:
        deviceTypeStr = "device_type::all";
        break;
    };
    auto doesDeviceSupportHalf = chosenDevice.has(sycl::aspect::fp16)
                                     ? "Supported"
                                     : "Not Supported";
    auto doesDeviceSupportDouble = chosenDevice.has(sycl::aspect::fp64)
                                       ? "Supported"
                                       : "Not Supported";
#if !defined(__COMPUTECPP__)
    auto doesDeviceSupportAtomics = chosenDevice.has(sycl::aspect::atomic64)
                                        ? "Supported"
                                        : "Not Supported";
#else
    auto doesDeviceSupportAtomics =
        (chosenDevice.has_extension("cl_khr_int64_base_atomics") &&
         chosenDevice.has_extension("cl_khr_int64_extended_atomics"))
            ? "Supported"
            : "Not Supported";
#endif
    auto platformNameStr =
        chosenPlatform.get_info<sycl::info::platform::name>();
    auto platformVendorStr =
        chosenPlatform.get_info<sycl::info::platform::vendor>();
    auto platformVersionStr =
        chosenPlatform.get_info<sycl::info::platform::version>();

    infoFile << "{\"device-name\": \"" << deviceNameStr
             << "\", \"device-vendor\": \"" << deviceVendorStr
             << "\", \"device-type\": \"" << deviceTypeStr
             << "\", \"device-version\": \"" << deviceVersionStr
             << "\", \"device-fp16\": \"" << doesDeviceSupportHalf
             << "\", \"device-fp64\": \"" << doesDeviceSupportDouble
             << "\", \"device-atomic64\": \"" << doesDeviceSupportAtomics
             << "\", \"platform-name\": \"" << platformNameStr
             << "\", \"platform-vendor\": \"" << platformVendorStr
             << "\", \"platform-version\": \"" << platformVersionStr << "\"}";
  }
}

}  // namespace util
}  // namespace sycl_cts
