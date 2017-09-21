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

#if defined(_MSC_VER)
extern "C" extern long __stdcall IsDebuggerPresent();
#endif

namespace sycl_cts {
namespace util {

/**
 */
test_manager::test_manager() : m_willExecute(false), m_wimpyMode(false) {}

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
    if (cmdarg.get_last_error(error))
      std::cout << error;
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
  if (cmdarg.get_value("--file", filePath) || cmdarg.find_key("-f")) {
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
  if (cmdarg.get_value("--csv", csvfile) || cmdarg.find_key("-c")) {
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
  if (cmdarg.get_value("--platform", platformName) || cmdarg.find_key("-p")) {
    selector.set_default_platform(platformName);
  }

  // set the default sycl cts device
  std::string deviceName;
  if (cmdarg.get_value("--device", deviceName) || cmdarg.find_key("-d")) {
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
    --help     -h         Show this help message
    --json     -j         Print test results in JSON format
    --text                Print test results in text format 
    --csv      -c         CSV file for specifying tests to run
    --list     -l         List the tests compiled in this executable
    --wimpy    -w         Run with reduced test complexity (faster)
    --platform -p [name]  Set a platform to target:
                  'host'
                  'amd'
                  'arm'
                  'intel'
                  'nvidia'
    --device   -d [name]  Select a device to target:
                  'host'
                  'opencl_cpu'
                  'opencl_gpu'
    --test        [name]   Specify a specific test to run by name, eg.
                          'unary_math_sin'
    --file     -f [path]   Redirect test output to a file

)";
  std::cout << usage << std::endl;
}

/**
 */
bool test_manager::will_execute() const { return m_willExecute; }

/**
 */
bool test_manager::wimpy_mode_enabled() const { return m_wimpyMode; }

} // namespace util
} // namespace sycl_cts
