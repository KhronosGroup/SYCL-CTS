/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for specialization constants with class with a member
//  function that accesses members
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME specialization_constants_class_with_member_fun

struct sc_class_with_memb {
  int a, b;
  constexpr sc_class_with_memb(int a, int b) : a(a), b(b) {}
  int calculate(int c) const { return a * b * c; }
};

// spec const defined in global namespace
constexpr sycl::specialization_id<sc_class_with_memb> sc_cl_w_mem_fn(
    sc_class_with_memb(0, 0));

namespace TEST_NAMESPACE {
using namespace sycl_cts;

/** test specialization constants
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
    try {
      auto queue = util::get_cts_object::queue();
      sycl::range<1> range(1);
      const int val_A = 3;
      const int val_B = 4;
      const int val_C = 2;
      sc_class_with_memb result(0, 0);
      sc_class_with_memb ref(val_A, val_B);
      int kernel_result_val = 0;
      {
        sycl::buffer<sc_class_with_memb, 1> result_buffer(&result, range);
        sycl::buffer<int, 1> kernel_result_val_buffer(&kernel_result_val,
                                                      range);
        queue.submit([&](sycl::handler &cgh) {
          auto res_acc =
              result_buffer.template get_access<sycl::access_mode::write>(cgh);
          auto kernel_res_val_acc =
              kernel_result_val_buffer
                  .template get_access<sycl::access_mode::write>(cgh);
          cgh.set_specialization_constant<sc_cl_w_mem_fn>(ref);
          cgh.single_task<class sc_cl_w_mem_fn_kernel>(
              [=](sycl::kernel_handler h) {
                res_acc[0] = h.get_specialization_constant<sc_cl_w_mem_fn>();
                kernel_res_val_acc[0] =
                    h.get_specialization_constant<sc_cl_w_mem_fn>().calculate(
                        val_C);
              });
        });
      }
      int ref_val = ref.calculate(val_C);
      int result_val = result.calculate(val_C);
      if (!check_equal_values(ref_val, result_val) ||
          !check_equal_values(ref_val, kernel_result_val))
        FAIL(log,
             "case specialization constants with class with a member function "
             "that accesses members");

    } catch (const sycl::exception &e) {
      log_exception(log, e);
      auto errorMsg =
          "a SYCL exception was caught: " + std::string(e.what());
      FAIL(log, errorMsg);
    } catch (const std::exception &e) {
      auto errorMsg =
          "an exception was caught: " + std::string(e.what());
      FAIL(log, errorMsg);
    }
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

}  // namespace TEST_NAMESPACE
