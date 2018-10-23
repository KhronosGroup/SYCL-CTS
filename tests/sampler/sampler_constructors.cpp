/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME sampler_constructors

namespace sampler_constructors__ {
using namespace sycl_cts;

/** Create a default sampler object
 */
cl::sycl::sampler defaultSampler() {
  return cl::sycl::sampler(
      cl::sycl::coordinate_normalization_mode::unnormalized,
      cl::sycl::addressing_mode::clamp, cl::sycl::filtering_mode::nearest);
}

/** tests the constructors for cl::sycl::sampler
*/
class TEST_NAME : public util::test_base {
 public:
  /** return information about this test
   */
  void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute this test
  */
  void run(util::logger &log) override {
    try {
      /** check (bool, addressing_mode, filtering_mode)
      * constructor and destructor
      */
      {
        cl::sycl::sampler sampler(
            cl::sycl::coordinate_normalization_mode::unnormalized,
            cl::sycl::addressing_mode::clamp,
            cl::sycl::filtering_mode::nearest);

        if (sampler.get_addressing_mode() != cl::sycl::addressing_mode::clamp) {
          FAIL(log,
               "sampler was not constructed correctly. (get_addressing_mode)");
        }

        if (sampler.get_filtering_mode() != cl::sycl::filtering_mode::nearest) {
          FAIL(log,
               "sampler was not constructed correctly. (get_filtering_mode)");
        }

        if (sampler.get_coordinate_normalization_mode() !=
            cl::sycl::coordinate_normalization_mode::unnormalized) {
          FAIL(log,
               "sampler was not constructed correctly. "
               "(get_coordinate_normalization_mode)");
        }
      }

      /** check copy constructor
      */
      {
        cl::sycl::sampler samplerA = defaultSampler();
        cl::sycl::sampler samplerB(samplerA);

        if (samplerA.get_addressing_mode() != samplerB.get_addressing_mode()) {
          FAIL(log,
               "sampler was not copy constructed correctly. "
               "(get_addressing_mode)");
        }

        if (samplerA.get_filtering_mode() != samplerB.get_filtering_mode()) {
          FAIL(log,
               "sampler was not copy constructed correctly. "
               "(get_filtering_mode)");
        }

        if (samplerA.get_coordinate_normalization_mode() !=
            samplerB.get_coordinate_normalization_mode()) {
          FAIL(log,
               "sampler was not copy constructed correctly. "
               "(get_coordinate_normalization_mode)");
        }
      }

      /** check assignment operator
      */
      {
        cl::sycl::sampler samplerA = defaultSampler();
        cl::sycl::sampler samplerB(
            cl::sycl::coordinate_normalization_mode::normalized,
            cl::sycl::addressing_mode::none, cl::sycl::filtering_mode::linear);
        samplerB = samplerA;

        if (samplerA.get_addressing_mode() != samplerB.get_addressing_mode()) {
          FAIL(log, "sampler was not copied correctly. (get_addressing_mode)");
        }

        if (samplerA.get_filtering_mode() != samplerB.get_filtering_mode()) {
          FAIL(log, "sampler was not copied correctly. (get_filtering_mode)");
        }

        if (samplerA.get_coordinate_normalization_mode() !=
            samplerB.get_coordinate_normalization_mode()) {
          FAIL(log,
               "sampler was not copied correctly. "
               "(get_coordinate_normalization_mode)");
        }
      }

      /** check move constructor
      */
      {
        cl::sycl::sampler samplerA(
            cl::sycl::coordinate_normalization_mode::unnormalized,
            cl::sycl::addressing_mode::clamp,
            cl::sycl::filtering_mode::nearest);
        cl::sycl::sampler samplerB(std::move(samplerA));

        if (samplerB.get_addressing_mode() !=
            cl::sycl::addressing_mode::clamp) {
          FAIL(log,
               "sampler was not move constructed correctly. "
               "(get_addressing_mode)");
        }

        if (samplerB.get_filtering_mode() !=
            cl::sycl::filtering_mode::nearest) {
          FAIL(log,
               "sampler was not move constructed correctly. "
               "(get_filtering_mode)");
        }

        if (samplerB.get_coordinate_normalization_mode() !=
            cl::sycl::coordinate_normalization_mode::unnormalized) {
          FAIL(log,
               "sampler was not move constructed correctly. "
               "(get_coordinate_normalization_mode)");
        }
      }

      /** check move assignment operator
      */
      {
        cl::sycl::sampler samplerA(
            cl::sycl::coordinate_normalization_mode::unnormalized,
            cl::sycl::addressing_mode::clamp,
            cl::sycl::filtering_mode::nearest);
        cl::sycl::sampler samplerB(
            cl::sycl::coordinate_normalization_mode::normalized,
            cl::sycl::addressing_mode::none, cl::sycl::filtering_mode::linear);
        samplerB = std::move(samplerA);

        if (samplerB.get_addressing_mode() !=
            cl::sycl::addressing_mode::clamp) {
          FAIL(
              log,
              "sampler was not move assigned correctly. (get_addressing_mode)");
        }

        if (samplerB.get_filtering_mode() !=
            cl::sycl::filtering_mode::nearest) {
          FAIL(log,
               "sampler was not move assigned correctly. (get_filtering_mode)");
        }

        if (samplerB.get_coordinate_normalization_mode() !=
            cl::sycl::coordinate_normalization_mode::unnormalized) {
          FAIL(log,
               "sampler was not move assigned correctly. "
               "(get_coordinate_normalization_mode)");
        }
      }

      /* check equality operator
      */
      {
        cl::sycl::sampler samplerA = defaultSampler();
        cl::sycl::sampler samplerB(samplerA);
        cl::sycl::sampler samplerC(
            cl::sycl::coordinate_normalization_mode::normalized,
            cl::sycl::addressing_mode::none, cl::sycl::filtering_mode::linear);
        samplerC = samplerA;
        cl::sycl::sampler samplerD(
            cl::sycl::coordinate_normalization_mode::normalized,
            cl::sycl::addressing_mode::none, cl::sycl::filtering_mode::linear);

        if (!(samplerA == samplerB) &&
            ((samplerA.get_addressing_mode() !=
              samplerB.get_addressing_mode()) ||
             (samplerA.get_filtering_mode() != samplerB.get_filtering_mode()) ||
             (samplerA.get_coordinate_normalization_mode() !=
              samplerB.get_coordinate_normalization_mode()))) {
          FAIL(log,
               "sampler equality does not work correctly (copy constructed)");
        }

        if (!(samplerA == samplerC) &&
            ((samplerA.get_addressing_mode() !=
              samplerC.get_addressing_mode()) ||
             (samplerA.get_filtering_mode() != samplerC.get_filtering_mode()) ||
             (samplerA.get_coordinate_normalization_mode() !=
              samplerC.get_coordinate_normalization_mode()))) {
          FAIL(log, "sampler equality does not work correctly (copy assigned)");
        }
        if (samplerA != samplerB) {
          FAIL(log,
               "sampler non-equality does not work correctly"
               "(copy constructed)");
        }
        if (samplerA != samplerC) {
          FAIL(log,
               "sampler non-equality does not work correctly"
               "(copy assigned)");
        }
        if (samplerC == samplerD) {
          FAIL(log,
               "sampler equality does not work correctly"
               "(comparing same)");
        }
        if (!(samplerC != samplerD)) {
          FAIL(log,
               "sampler non-equality does not work correctly"
               "(comparing same)");
        }
      }

      /* check hashing
      */
      {
        cl::sycl::hash_class<cl::sycl::sampler> hasher;

        cl::sycl::sampler samplerA = defaultSampler();
        auto hashA = hasher(samplerA);
        cl::sycl::sampler samplerB = std::move(samplerA);

        if (hashA != hasher(samplerB)) {
          FAIL(log,
               "sampler hashing does not work correctly (hashing of equal "
               "failed)");
        }
      }
    } catch (const cl::sycl::exception &e) {
      log_exception(log, e);
      cl::sycl::string_class errorMsg =
          "a SYCL exception was caught: " + cl::sycl::string_class(e.what());
      FAIL(log, errorMsg.c_str());
    }
  }
};

// register this test with the test_collection
util::test_proxy<TEST_NAME> proxy;

} /* namespace sampler_constructors__ */
