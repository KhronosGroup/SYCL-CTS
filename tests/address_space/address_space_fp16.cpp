/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provide address_space tests for types that require fp16 extension
//
*******************************************************************************/

#define TEST_NAME address_space_fp16

#include "../common/common.h"
#include "./../../util/extensions.h"
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

    using availability =
        util::extensions::availability<util::extensions::tag::fp16>;
    if (!availability::check(queue, log))
      return;

    test_types<sycl::half>(log);
  }
};

util::test_proxy<TEST_NAME> proxy;

} // namespace TEST_NAMESPACE
