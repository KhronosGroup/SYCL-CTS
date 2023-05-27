/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  This test calls sycl::handler::use_kernel_bundle and executes provided
//  kernel if this kernel is incompatible for the current device. The test uses
//  secondary queue.
//
//  The test verifies that the exception with sycl::errc::kernel_not_supported
//  was thrown.
//
*******************************************************************************/

#include "../common/common.h"
#include "use_kernel_bundle.h"

#define TEST_NAME \
  use_kernel_bundle_invoke_kernel_with_incompat_dev_with_second_queue

namespace TEST_NAMESPACE {
using namespace sycl_cts;
using namespace sycl_cts::tests::use_kernel_bundle;

/** @brief Struct with overloaded call operator for using in "for_all_types"
 *         function to be able run this test with different user-defined kernels
 *  @tparam KernelDescriptorT Determined user-defined structs with kernels and
 *          restrictions for this kernels
 */
template <typename KernelDescriptorT>
struct run_verification {
  std::string kernel_name;

  void check_exception(sycl::queue& primary_queue,
                       sycl::queue& secondary_queue) {
    using kernel_functor = typename KernelDescriptorT::type;
    sycl::kernel_bundle<sycl::bundle_state::executable> kernel_bundle =
        get_non_empty_bundle<kernel_functor>(primary_queue.get_context());
    bool ex_was_thrown = false;

    unsigned long long data;
    try {
      sycl::buffer<unsigned long long, 1> data_buf(&data, 1);
      primary_queue.submit(
          [&](sycl::handler& cgh) {
            auto data_acc =
                data_buf.get_access<sycl::access_mode::read_write>(cgh);
            cgh.use_kernel_bundle(kernel_bundle);
            cgh.parallel_for(sycl::range(1), kernel_functor{data_acc});
          },
          secondary_queue);
    } catch (const sycl::exception& e) {
      if (e.code() != sycl::errc::kernel_not_supported) {
        FAIL(log, unexpected_exception_msg);
        throw;
      }
      ex_was_thrown = true;
    }

    if (!ex_was_thrown) {
      FAIL(log, "Exception was not thrown for kernel name: " + kernel_name);
    }
  }
  /** @brief Call sycl::handler::use_kernel_bundle with user-defined kernel for
   *         incompatible device and verify that exception with
   *         sycl::errc::kernel_not_supported code was thrown with using
   *         secondary queue
   *  @param log sycl_cts::util::logger class object
   *  @param ctx Context that will used for sycl::queue and kernel bundle
   *  @param kernel_name String with tested kernel
   */
  void operator()(util::logger &log, const sycl::context &ctx,
                  const std::string &kernel_name) {
    this->kernel_name = kernel_name;

    auto restrictions{KernelDescriptorT::get_restrictions()};
    std::vector<sycl::device> compatible_devs;
    std::vector<sycl::device> incompatible_devs;

    for (auto& dev : ctx.get_devices()) {
      if (restrictions.is_compatible(dev))
        compatible_devs.push_back(dev);
      else
        incompatible_devs.push_back(dev);
    }

    if (!compatible_devs.empty() && !incompatible_devs.empty()) {
      sycl::queue first_queue_incompat_dev(ctx, incompatible_devs[0]);
      sycl::queue second_queue_incompat_dev(ctx, incompatible_devs[0]);

      sycl::queue first_queue_compat_dev(ctx, compatible_devs[0]);
      sycl::queue second_queue_compat_dev(ctx, compatible_devs[0]);

      // Check when devices of both queues are not compatible with kernel bundle
      check_exception(first_queue_incompat_dev, second_queue_incompat_dev);
      // Check when device of secondary queue is not compatible with kernel
      // bundle
      check_exception(first_queue_compat_dev, second_queue_incompat_dev);
      // Check when device of primary queue is not compatible with kernel bundle
      check_exception(first_queue_incompat_dev, second_queue_compat_dev);
    }
  }
};

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
    sycl::device dev = util::get_cts_object::device();

    if (dev.get_platform().get_devices().size() < 2) {
      SKIP(
          "Not enough devices on the platform used. Required at least two "
          "devices to test kernel"
          "on not compatible device. In case of single available device a "
          "kernel_bundle with incompatible"
          "kernel for available device can't be gotten as there is no devices "
          "which support the kernel,"
          "only device that is not compatible");
    }

    sycl::context ctx(dev.get_platform().get_devices());

    for_all_types<run_verification>(user_def_kernels, log, ctx);
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

}  // namespace TEST_NAMESPACE
