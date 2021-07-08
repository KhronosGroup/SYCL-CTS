/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provide coverage for global mem_fence
//
*******************************************************************************/

#include "../common/common.h"
#include "../group/group_mem_fence_common.h"

#define TEST_NAME group_global_mem_fence

namespace group_global_mem_fence__ {
using namespace sycl_cts;

template <access_group accessGroup, int dim>
class global_mem_fence_kernel_global;

/**
 * @brief Test mem_fence works for global fence space
 * @param accessGroup Fence group to use for tests
 * @param dim Dimension to use
 * @param log Logger to use
 * @param queue Queue to use
 */
template <access_group accessGroup, int dim>
void test_mem_fence(util::logger &log, sycl::queue &queue) {
  const auto fenceSpace = sycl::access::fence_space::global_space;
  const auto testName = test_name<accessGroup, dim>::get(fenceSpace);

  using globalKernelT = global_mem_fence_kernel_global<accessGroup, dim>;

  const auto fenceCallFactory = make_fence_call_factory(
    [=](sycl::group<dim> item) {
      item.mem_fence(fenceSpace);
    },
    [=](sycl::group<dim> item) {
      item.template mem_fence<sycl::access_mode::read_write>(fenceSpace);
    },
    [=](sycl::group<dim> item) {
      item.template mem_fence<sycl::access_mode::read>(fenceSpace);
    },
    [=](sycl::group<dim> item) {
      item.template mem_fence<sycl::access_mode::write>(fenceSpace);
  });
  const auto access = std::integral_constant<access_group, accessGroup>{};

  const auto readFenceCall = fenceCallFactory.get_read(access);
  const auto writeFenceCall = fenceCallFactory.get_write(access);

  // Verify global mem_fence works for global address space
  {
    const bool passed = test_rw_mem_fence_global_space<globalKernelT, dim>(
        log, queue, readFenceCall, writeFenceCall);

    if (!passed) {
      FAIL(log, testName + "failed for global address space");
    }
  }
}

/** test sycl::group mem_fence functions
*/
class TEST_NAME : public util::test_base {
 public:
  /** return information about this test
  *  @param info, test_base::info structure as output
  */
  void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute the test
  *  @param log, test transcript logging class
  */
  void run(util::logger &log) override {
    try {
      auto queue = util::get_cts_object::queue();

      test_mem_fence<access_group::useDefault, 1>(log, queue);
      test_mem_fence<access_group::useCombined, 1>(log, queue);
      test_mem_fence<access_group::useSeparate, 1>(log, queue);

      queue.wait_and_throw();
    } catch (const sycl::exception &e) {
      log_exception(log, e);
      std::string errorMsg =
          "a SYCL exception was caught: " + std::string(e.what());
      FAIL(log, errorMsg.c_str());
    }
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

}  // namespace group_global_mem_fence__
