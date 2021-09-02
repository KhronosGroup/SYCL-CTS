/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME program_info

namespace program_info__ {
using namespace sycl_cts;

/** tests the info for sycl::program
 */
class TEST_NAME : public sycl_cts::util::test_base {
 public:
  /** return information about this test
   */
  void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute the test
   */
  void run(util::logger &log) override {
    auto context = util::get_cts_object::context();
    auto deviceList = context.get_devices();

    try {
      auto program =
          util::get_cts_object::program::built<class TEST_NAME>(context);

      /** check types
       */
      { check_type_existence<sycl::info::program> typeCheck; }
      using vectorDevicesInfo = std::vector<sycl::device>;

      /** check program info parameters
       */
      {
        sycl::context programContext =
            program.get_info<sycl::info::program::context>();
        TEST_TYPE_TRAIT(program, context, program);
      }
      {
        vectorDevicesInfo vectorDevices =
            program.get_info<sycl::info::program::devices>();
        TEST_TYPE_TRAIT(program, devices, program);
      }
      {
        if (!program.is_host()) {
          cl_uint referenceCount =
              program.get_info<sycl::info::program::reference_count>();
          TEST_TYPE_TRAIT(program, reference_count, program);
        }
      }

    } catch (const sycl::feature_not_supported &fnse_build) {
      if (!is_compiler_available(deviceList) ||
          !is_linker_available(deviceList)) {
        log.note("online compiler or linker not available -- skipping check");
      } else {
        throw;
      }
    }
  }
};

// register this test with the test_collection
util::test_proxy<TEST_NAME> proxy;

} /* namespace program_info__ */
