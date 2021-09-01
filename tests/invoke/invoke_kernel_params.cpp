/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME invoke_kernel_params

namespace invoke_kernel_params__ {
using namespace sycl_cts;

struct basic_type_t {
  uint32_t a;
  uint8_t b;
  uint8_t c;
};

enum {
  ref_a = 1,
  ref_b_a = 10,
  ref_b_b = 12,
  ref_b_c = 5,
};

struct array_type_t {
  int32_t int_array[8];
};

struct test_kernel {
 private:
  typedef sycl::accessor<uint32_t, 1, sycl::access_mode::write,
                         sycl::target::global_buffer>
      acc_uint32_t;

  uint32_t var_a;
  basic_type_t var_b;
  acc_uint32_t out;

  struct {
    int32_t int_array[8];
  } inner;

 public:
  test_kernel(uint32_t a, basic_type_t b, acc_uint32_t acc_out)
      : var_a(a), var_b(b), out(acc_out) {}

  void set(uint32_t i, int32_t v) { inner.int_array[i] = v; }

  void operator()() const {
    uint32_t pass = 1;
    pass &= (var_a == ref_a);
    pass &= (var_b.a == ref_b_a);
    pass &= (var_b.b == ref_b_b);
    pass &= (var_b.c == ref_b_c);

    for (int i = 0; i < 8; i++) {
      pass &= (inner.int_array[i] == i);
    }

    out[0] = pass;
  }
};

class test_kernel_name;

/**
 */
class TEST_NAME : public util::test_base {
 public:
  typedef sycl::accessor<uint32_t, 1, sycl::access_mode::write,
                         sycl::target::global_buffer>
      acc_uint32_t;

  /** return information about this test
   */
  void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute the test
   */
  void run(util::logger &log) override {
    {
      uint32_t result = 0;

      try {
        auto my_queue = util::get_cts_object::queue();

        sycl::buffer<uint32_t> buf_result(&result, sycl::range<1>(1));

        my_queue.submit([&](sycl::handler &cgh) {
          // access the output
          auto acc_pass =
              buf_result.template get_access<sycl::access_mode::write>(cgh);

          const uint32_t var_a = ref_a;
          const basic_type_t var_b = {ref_b_a, ref_b_b, ref_b_c};

          array_type_t my_array;
          for (uint32_t i = 0; i < 8; i++) {
            my_array.int_array[i] = i;
          }

          cgh.single_task<class test_kernel_name>([=]() {
            uint32_t pass = 1;
            pass &= (var_a == ref_a);
            pass &= (var_b.a == ref_b_a);
            pass &= (var_b.b == ref_b_b);
            pass &= (var_b.c == ref_b_c);

            for (int32_t i = 0; i < 8; i++) {
              pass &= (my_array.int_array[i] == i);
            }

            acc_pass[0] = pass;
          });
        });

        my_queue.wait_and_throw();
      } catch (const sycl::exception &e) {
        log_exception(log, e);
        std::string errorMsg =
            "a SYCL exception was caught: " + std::string(e.what());
        FAIL(log, errorMsg.c_str());
      }

      if (!CHECK_VALUE_SCALAR(log, result, static_cast<decltype(result)>(1))) {
        FAIL(log, "incorrect values passed to lambda kernel");
      }
    }

    {
      uint32_t result = 0;

      try {
        auto my_queue = util::get_cts_object::queue();

        sycl::buffer<uint32_t> buf_result(&result, sycl::range<1>(1));

        my_queue.submit([&](sycl::handler &cgh) {
          // construct a basic type
          basic_type_t my_type = {ref_b_a, ref_b_b, ref_b_c};

          // access the output
          auto acc_pass =
              buf_result.template get_access<sycl::access_mode::write,
                                             sycl::target::global_buffer>(cgh);

          // instantiate the kernel
          test_kernel my_kernel(ref_a, my_type, acc_pass);

          for (int i = 0; i < 8; i++) {
            my_kernel.set(i, i);
          }

          // execute the kernel
          cgh.single_task(my_kernel);
        });

        my_queue.wait_and_throw();
      } catch (const sycl::exception &e) {
        log_exception(log, e);
        std::string errorMsg =
            "a SYCL exception was caught: " + std::string(e.what());
        FAIL(log, errorMsg.c_str());
      }

      if (!CHECK_VALUE_SCALAR(log, result, static_cast<decltype(result)>(1))) {
        FAIL(log, "incorrect values passed to functor kernel");
      }
    }
  }
};

util::test_proxy<TEST_NAME> proxy;

} /* namespace invoke_kernel_params__ */
