/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
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
sycl::sampler defaultSampler() {
  return sycl::sampler(sycl::coordinate_normalization_mode::unnormalized,
                       sycl::addressing_mode::clamp,
                       sycl::filtering_mode::nearest);
}

/** tests the constructors for sycl::sampler
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
    /** check (bool, addressing_mode, filtering_mode)
     * constructor and destructor
     */
    {
      sycl::sampler sampler(sycl::coordinate_normalization_mode::unnormalized,
                            sycl::addressing_mode::clamp,
                            sycl::filtering_mode::nearest);

      if (sampler.get_addressing_mode() != sycl::addressing_mode::clamp) {
        FAIL(log,
             "sampler was not constructed correctly. (get_addressing_mode)");
      }

      if (sampler.get_filtering_mode() != sycl::filtering_mode::nearest) {
        FAIL(log,
             "sampler was not constructed correctly. (get_filtering_mode)");
      }

      if (sampler.get_coordinate_normalization_mode() !=
          sycl::coordinate_normalization_mode::unnormalized) {
        FAIL(log,
             "sampler was not constructed correctly. "
             "(get_coordinate_normalization_mode)");
      }
    }

    /** check copy constructor
     */
    {
      sycl::sampler samplerA = defaultSampler();
      sycl::sampler samplerB(samplerA);

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
      sycl::sampler samplerA = defaultSampler();
      sycl::sampler samplerB(sycl::coordinate_normalization_mode::normalized,
                             sycl::addressing_mode::none,
                             sycl::filtering_mode::linear);
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
      sycl::sampler samplerA(sycl::coordinate_normalization_mode::unnormalized,
                             sycl::addressing_mode::clamp,
                             sycl::filtering_mode::nearest);
      sycl::sampler samplerB(std::move(samplerA));

      if (samplerB.get_addressing_mode() != sycl::addressing_mode::clamp) {
        FAIL(log,
             "sampler was not move constructed correctly. "
             "(get_addressing_mode)");
      }

      if (samplerB.get_filtering_mode() != sycl::filtering_mode::nearest) {
        FAIL(log,
             "sampler was not move constructed correctly. "
             "(get_filtering_mode)");
      }

      if (samplerB.get_coordinate_normalization_mode() !=
          sycl::coordinate_normalization_mode::unnormalized) {
        FAIL(log,
             "sampler was not move constructed correctly. "
             "(get_coordinate_normalization_mode)");
      }
    }

    /** check move assignment operator
     */
    {
      sycl::sampler samplerA(sycl::coordinate_normalization_mode::unnormalized,
                             sycl::addressing_mode::clamp,
                             sycl::filtering_mode::nearest);
      sycl::sampler samplerB(sycl::coordinate_normalization_mode::normalized,
                             sycl::addressing_mode::none,
                             sycl::filtering_mode::linear);
      samplerB = std::move(samplerA);

      if (samplerB.get_addressing_mode() != sycl::addressing_mode::clamp) {
        FAIL(log,
             "sampler was not move assigned correctly. (get_addressing_mode)");
      }

      if (samplerB.get_filtering_mode() != sycl::filtering_mode::nearest) {
        FAIL(log,
             "sampler was not move assigned correctly. (get_filtering_mode)");
      }

      if (samplerB.get_coordinate_normalization_mode() !=
          sycl::coordinate_normalization_mode::unnormalized) {
        FAIL(log,
             "sampler was not move assigned correctly. "
             "(get_coordinate_normalization_mode)");
      }
    }

    /* check equality operator
     */
    {
      sycl::sampler samplerA = defaultSampler();
      sycl::sampler samplerB(samplerA);
      sycl::sampler samplerC(sycl::coordinate_normalization_mode::normalized,
                             sycl::addressing_mode::none,
                             sycl::filtering_mode::linear);
      samplerC = samplerA;
      sycl::sampler samplerD(sycl::coordinate_normalization_mode::normalized,
                             sycl::addressing_mode::none,
                             sycl::filtering_mode::linear);

      if (!(samplerA == samplerB) &&
          ((samplerA.get_addressing_mode() != samplerB.get_addressing_mode()) ||
           (samplerA.get_filtering_mode() != samplerB.get_filtering_mode()) ||
           (samplerA.get_coordinate_normalization_mode() !=
            samplerB.get_coordinate_normalization_mode()))) {
        FAIL(log,
             "sampler equality does not work correctly (copy constructed)");
      }

      if (!(samplerA == samplerC) &&
          ((samplerA.get_addressing_mode() != samplerC.get_addressing_mode()) ||
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
      std::hash<sycl::sampler> hasher;

      sycl::sampler samplerA = defaultSampler();
      auto hashA = hasher(samplerA);
      sycl::sampler samplerB = std::move(samplerA);

      if (hashA != hasher(samplerB)) {
        FAIL(log,
             "sampler hashing does not work correctly (hashing of equal "
             "failed)");
      }
    }
  }
};

// register this test with the test_collection
util::test_proxy<TEST_NAME> proxy;

} /* namespace sampler_constructors__ */
