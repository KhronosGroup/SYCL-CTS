/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2021 The Khronos Group Inc.
//
*******************************************************************************/

#include <regex>
#include <string>

#define CATCH_CONFIG_RUNNER
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/internal/catch_clara.hpp>

#include "./../../util/device_manager.h"
#include "cts_selector.h"

int main(int argc, char** argv) {
  using namespace sycl_cts;

  Catch::Session session;
  session.configData().name = "The SYCL 2020 Conformance Test Suite";

  std::string devicePattern;
  std::string infoDumpFile;
  bool listDevices = false;

  using namespace Catch::Clara;

  // TODO: Look into removing some of Catch2's default options
  //       that we don't need, e.g. for benchmarking.
  auto cli = Opt(devicePattern, "pattern")["--device"](
                 "Select SYCL device to run CTS on. ECMAScript "
                 "regular expression syntax can be used") |
             Opt(listDevices)["--list-devices"]("List all available devices") |
             Opt(infoDumpFile, "file")["--info-dump"](
                 "Dump platform and device info to file") |
             session.cli();

  session.cli(cli);

  const int returnCode = session.applyCommandLine(argc, argv);
  if (returnCode != 0) {
    return returnCode;
  }

  auto& device_mngr = util::get<util::device_manager>();
  if (!devicePattern.empty()) {
    device_mngr.set_device_regex(std::regex(devicePattern));
  }

  if (listDevices) {
    device_mngr.list_devices();
    return EXIT_SUCCESS;
  }

  if (!infoDumpFile.empty()) {
    device_mngr.dump_info(infoDumpFile);
  }

  return session.run();
}
