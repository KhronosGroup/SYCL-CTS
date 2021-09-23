/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provide verification to cooperation USM functions get_pointer_type and
//  get_pointer_device with four memory allocation types: host, device, shared
//  and non USM allocation.
//
*******************************************************************************/

#include "../../util/usm_helper.h"
#include <cstring>

#define TEST_NAME usm_get_pointer_queries

namespace TEST_NAMESPACE {
using namespace sycl_cts;

template <sycl::usm::alloc alloc>
void run_check(const sycl::queue &queue, sycl_cts::util::logger &log) {
  const auto &device{queue.get_device()};
  const auto &context{queue.get_context()};

  auto str_usm_alloc_type{usm_helper::get_allocation_description<alloc>()};

  if (device.has(usm_helper::get_aspect<alloc>())) {
    auto allocated_memory = usm_helper::allocate_usm_memory<alloc, int>(queue);
    const auto value = sycl::get_pointer_type(allocated_memory.get(), context);

    if (alloc != value) {
      FAIL(log, "sycl::get_pointer_type return " +
                    std::to_string(to_integral(value)) +
                    " type, expected sycl::usm::alloc::" +
                    std::string(str_usm_alloc_type) + " type");
    }
    if (sycl::get_pointer_device(allocated_memory.get(), context) != device) {
      FAIL(log, "sycl::get_pointer_device return invalid device for " +
                    std::string(str_usm_alloc_type) + " type");
    }
  } else {
    log.note("Device does not support " + std::string(str_usm_alloc_type) +
             " allocation. Tests were skipped.");
  }
}

/** Test instance
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
      auto queue{util::get_cts_object::queue()};

      {
        int non_usm_allocated_memory{};
        if (sycl::usm::alloc::unknown !=
            sycl::get_pointer_type(&non_usm_allocated_memory,
                                   queue.get_context())) {
          FAIL(log,
               "sycl::get_pointer_type return not sycl::usm::alloc::unknown "
               "type");
        }
      }

      run_check<sycl::usm::alloc::shared>(queue, log);
      run_check<sycl::usm::alloc::device>(queue, log);
      run_check<sycl::usm::alloc::host>(queue, log);

    } catch (const cl::sycl::exception &e) {
      log_exception(log, e);
      auto errorMsg = "a SYCL exception was caught: " + std::string(e.what());
      FAIL(log, errorMsg);
    }
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;
}  // namespace TEST_NAMESPACE
