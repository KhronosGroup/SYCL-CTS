/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provide address_space tests for types that require fp64 extension
//
*******************************************************************************/

#define TEST_NAME address_space_fp64

#include "../common/common.h"
#include "address_space_common.h"

namespace TEST_NAMESPACE {
using namespace sycl_cts;

class TEST_NAME : public sycl_cts::util::test_base {
 public:
  void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  void run(util::logger &log) override {
    auto queue = util::get_cts_object::queue();

    if (!queue.get_device().has(sycl::aspect::fp16)) {
      WARN(
          "Device doesn't support double precision floating point data type - "
          "skipping the test");
      return;
    }

    test_types<double>(log);
  }
};

util::test_proxy<TEST_NAME> proxy;

}  // namespace TEST_NAMESPACE
