/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME program_info

namespace program_info__ {
using namespace sycl_cts;

/** tests the info for cl::sycl::program
 */
class TEST_NAME : public sycl_cts::util::test_base_opencl {
 public:
  /** return information about this test
   */
  void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute the test
   */
  void run(util::logger &log) override {
    try {
      auto context = util::get_cts_object::context();
      auto program =
          util::get_cts_object::program::built<class TEST_NAME>(context);

      /** check types
      */
      using programInfo = cl::sycl::info::program;
      using vectorDevicesInfo = cl::sycl::vector_class<cl::sycl::device>;

      /** check program info parameters
      */
      {
        cl::sycl::context programContext =
            program.get_info<cl::sycl::info::program::context>();
        TEST_TYPE_TRAIT(program, context, program);
      }
      {
        vectorDevicesInfo vectorDevices =
            program.get_info<cl::sycl::info::program::devices>();
        TEST_TYPE_TRAIT(program, devices, program);
      }
      {
        cl_uint referenceCount =
            program.get_info<cl::sycl::info::program::reference_count>();
        TEST_TYPE_TRAIT(program, reference_count, program);
      }
    } catch (cl::sycl::exception e) {
      log_exception(log, e);
      cl::sycl::string_class errorMsg =
          "a SYCL exception was caught: " + cl::sycl::string_class(e.what());
      FAIL(log, errorMsg.c_str());
    }
  }
};

// register this test with the test_collection
util::test_proxy<TEST_NAME> proxy;

} /* namespace program_info__ */
