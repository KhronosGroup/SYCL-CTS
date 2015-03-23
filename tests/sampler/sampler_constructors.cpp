/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2015 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME sampler_constructors

namespace sampler_constructors__ {
using namespace sycl_cts;

/** tests the constructors for cl::sycl::sampler
*/
class TEST_NAME : public util::test_base {
 public:
  /** return information about this test
   */
  virtual void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute this test
  */
  virtual void run(util::logger &log) override {
    try {
      cts_selector selector;
      cl::sycl::queue queue(selector);

      queue.submit([&](cl::sycl::handler &handler) {
        cl::sycl::sampler sampler(false,
                                  cl::sycl::sampler_addressing_mode::none,
                                  cl::sycl::sampler_filter_mode::nearest);

        /** check (bool, sampler_addressing_mode, sampler_filter_mode)
         * constructor and destructor
        */
        {
          cl::sycl::sampler sampler(false,
                                    cl::sycl::sampler_addressing_mode::clamp,
                                    cl::sycl::sampler_filter_mode::nearest);

          if (!sampler.is_normalized_coordinates()) {
            FAIL(log, "sampler was not constructed correctly.");
          }

          if (sampler.get_addressing_mode() ==
              cl::sycl::sampler_addressing_mode::clamp) {
            FAIL(log, "sampler was not constructed correctly.");
          }

          if (sampler.get_filter_mode() ==
              cl::sycl::sampler_filter_mode::nearest) {
            FAIL(log, "sampler was not constructed correctly.");
          }
        }

        /** check copy constructor
        */
        {
          cl::sycl::sampler samplerA(false,
                                     cl::sycl::sampler_addressing_mode::clamp,
                                     cl::sycl::sampler_filter_mode::nearest);
          cl::sycl::sampler samplerB(samplerA);

          if (samplerA.get() != samplerB.get()) {
            FAIL(log, "sampler was not copied correctly.");
          }

          if (samplerA.is_normalized_coordinates() ==
              samplerB.is_normalized_coordinates()) {
            FAIL(log, "sampler was not copied correctly.");
          }

          if (samplerA.get_addressing_mode() ==
              samplerB.get_addressing_mode()) {
            FAIL(log, "sampler was not copied correctly.");
          }

          if (samplerA.get_filter_mode() == samplerB.get_filter_mode()) {
            FAIL(log, "sampler was not copied correctly.");
          }
        }

        /** check assignment operator
        */
        {
          cl::sycl::sampler samplerA(false,
                                     cl::sycl::sampler_addressing_mode::clamp,
                                     cl::sycl::sampler_filter_mode::nearest);
          cl::sycl::sampler samplerB = samplerA;

          if (samplerA.get() != samplerB.get()) {
            FAIL(log, "sampler was not assigned correctly.");
          }

          if (samplerA.is_normalized_coordinates() ==
              samplerB.is_normalized_coordinates()) {
            FAIL(log, "sampler was not copied correctly.");
          }

          if (samplerA.get_addressing_mode() ==
              samplerB.get_addressing_mode()) {
            FAIL(log, "sampler was not copied correctly.");
          }

          if (samplerA.get_filter_mode() == samplerB.get_filter_mode()) {
            FAIL(log, "sampler was not copied correctly.");
          }
        }
      });
    } catch (cl::sycl::exception e) {
      log_exception(log, e);
      FAIL(log, "a sycl exception was caught");
    }
  }
};

// register this test with the test_collection
util::test_proxy<TEST_NAME> proxy;

} /* namespace sampler_constructors__ */
