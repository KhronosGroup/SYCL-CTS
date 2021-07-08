/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME kernel_args

namespace kernel_args__ {
using namespace sycl_cts;

struct nestedStruct {
  long long a;
};

struct outerStruct {
  int a;
  float b;
  nestedStruct innerStruct;
};

class scalar_struct_kernel;
class sampler_kernel;

bool check_outer_struct_eq(int a, float b, long long c, outerStruct exp) {
  return a == exp.a && c == exp.innerStruct.a &&
         b == exp.b;
}

/** Test kernel args conform to the rules for parameter passing to kernels
 */
class TEST_NAME : public util::test_base {
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
      // Test values
      int testScalar = 1;
      auto testVec = sycl::vec<int, 4>(1, 2, 3, 4);
      outerStruct testStruct{1, 2.0f, nestedStruct{3}};
      int error = 0;

      auto my_queue = util::get_cts_object::queue();

      log.note("Testing kernel args with scalars and struct members");
      {
        sycl::buffer<int, 1> buf(&error, sycl::range<1>(1));
        my_queue.submit([&](sycl::handler &cgh) {
          auto acc =
              buf.template get_access<sycl::access_mode::read_write>(cgh);
          cgh.single_task<class scalar_struct_kernel>([=]() {
            // Test that the values outside kernel have been captured
            int kernelScalar = testScalar;

            sycl::vec<int, 4> kernelVec = sycl::vec<int, 4>(
                testVec.x(), testVec.y(), testVec.z(), testVec.w());
            (void)kernelVec;  // silence warnings

            int a = testStruct.a;
            float b = testStruct.b;
            long long c = testStruct.innerStruct.a;
            // check all value here. Expected value from testVec
            outerStruct exp{1, 2.0f, nestedStruct{3}};
            auto exp_vec = sycl::vec<int, 4>(1, 2, 3, 4);
            if (kernelScalar != 1) {
              acc[0] += 1;
            }
            if (!check_equal_values(kernelVec, exp_vec)) {
              acc[0] += 1 << 1;
            }
            if (!check_outer_struct_eq(a, b, c, exp)) {
              acc[0] += 1 << 2;
            }
          });
        });
      }

      my_queue.wait_and_throw();

      log.note("Testing kernel args with samplers");
      {
        try {
          sycl::sampler sampler(
              sycl::coordinate_normalization_mode::unnormalized,
              sycl::addressing_mode::clamp,
              sycl::filtering_mode::nearest);

          my_queue.submit([&](sycl::handler &cgh) {
            cgh.single_task<class sampler_kernel>([=]() {
              // Test that the value of sampler outside kernel has been captured
              sycl::sampler kernel_sampler = sampler;
            });
          });

          my_queue.wait_and_throw();
          if (error != 0) {
            if (error & 1) {
              FAIL(log, "kernelScalar capture error");
            }
            if (error & 1 << 1) {
              FAIL(log, "kernelVec capture error");
            }
            if (error & 1 << 2) {
              FAIL(log, "outer_struct capture error");
            }
          }
        } catch (const sycl::feature_not_supported &fnse) {
          if (!my_queue.get_device()
                   .get_info<sycl::info::device::image_support>()) {
            log.note("device does not support images -- skipping check");
          } else {
            throw;
          }
        }
      }
    } catch (const sycl::exception &e) {
      log_exception(log, e);
      std::string errorMsg =
          "a SYCL exception was caught: " + std::string(e.what());
      FAIL(log, errorMsg.c_str());
    }
  }
};

// register this test with the test_collection
util::test_proxy<TEST_NAME> proxy;

} /* namespace kernel_args__ */
