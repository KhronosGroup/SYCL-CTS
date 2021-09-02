/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides coverage for global_and_local mem_fence
//
*******************************************************************************/

#include "../common/common.h"
#include "../nd_item/nd_item_mem_fence_common.h"

#define TEST_NAME nd_item_combined_mem_fence

namespace nd_item_combined_mem_fence__ {
using namespace sycl_cts;

template <access_group accessGroup, int dim>
class combined_mem_fence_kernel_local;

template <access_group accessGroup, int dim>
class combined_mem_fence_kernel_global;

/**
 * @brief Test mem_fence works for global_and_local fence space
 * @param accessGroup Fence group to use for tests
 * @param dim Dimension to use
 * @param log Logger to use
 * @param queue Queue to use
 */
template <access_group accessGroup, int dim>
void test_mem_fence(util::logger &log, sycl::queue &queue) {
  const auto fenceSpace = sycl::access::fence_space::global_and_local;
  const auto testName = test_name<accessGroup, dim>::get(fenceSpace);

  using localKernelT = combined_mem_fence_kernel_local<accessGroup, dim>;
  using globalKernelT = combined_mem_fence_kernel_global<accessGroup, dim>;

  const auto fenceCallFactory = make_fence_call_factory(
      [=](sycl::nd_item<dim> item) { item.mem_fence(fenceSpace); },
      [=](sycl::nd_item<dim> item) {
        item.template mem_fence<sycl::access_mode::read_write>(fenceSpace);
      },
      [=](sycl::nd_item<dim> item) {
        item.template mem_fence<sycl::access_mode::read>(fenceSpace);
      },
      [=](sycl::nd_item<dim> item) {
        item.template mem_fence<sycl::access_mode::write>(fenceSpace);
      });
  const auto access = std::integral_constant<access_group, accessGroup>{};

  const auto readFenceCall = fenceCallFactory.get_read(access);
  const auto writeFenceCall = fenceCallFactory.get_write(access);

  // Verify global_and_local mem_fence works for local address space
  {
    const bool passed = test_rw_mem_fence_local_space<localKernelT, dim>(
        log, queue, readFenceCall, writeFenceCall);

    if (!passed) {
      FAIL(log, testName + " failed for local address space");
    }
  }
  // Verify global_and_local mem_fence works for global address space
  {
    const bool passed = test_rw_mem_fence_global_space<globalKernelT, dim>(
        log, queue, readFenceCall, writeFenceCall);

    if (!passed) {
      FAIL(log, testName + " failed for global address space");
    }
  }
}

/** test sycl::nd_item mem_fence functions
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
    auto queue = util::get_cts_object::queue();

    test_mem_fence<access_group::useDefault, 1>(log, queue);
    test_mem_fence<access_group::useCombined, 1>(log, queue);
    test_mem_fence<access_group::useSeparate, 1>(log, queue);

    queue.wait_and_throw();
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

}  // namespace nd_item_combined_mem_fence__
