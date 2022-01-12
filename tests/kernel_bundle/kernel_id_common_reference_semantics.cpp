/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for sycl::kernel_id common reference semantics
//
*******************************************************************************/

#include "../../util/exceptions.h"
#include "../common/common.h"
#include "../common/common_by_reference.h"
#include "get_kernel_id.h"
#include "kernel_bundle.h"

#define TEST_NAME kernel_id_common_reference_semantics

namespace TEST_NAMESPACE {
using namespace sycl_cts;
using namespace sycl_cts::tests::get_kernel_id;
using namespace sycl_cts::tests::kernel_bundle;

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
    using k_id_vector = std::vector<sycl::kernel_id>;
    /** Check that sycl::kernel_id follows common reference semantics
     */
    run_verification<by_queue>(log, [&](sycl::queue &queue) {
      using k_name = class kernel_id_comm_ref_sem;
      using k_name_other = class kernel_id_other_comm_ref_sem;
      sycl::kernel_id k_id = sycl::get_kernel_id<k_name>();
      sycl::kernel_id k_id_other = sycl::get_kernel_id<k_name_other>();
      common_by_reference::check_on_host<sycl::kernel_id, kernel_id_storage>(
          log, k_id, k_id_other, "sycl::kernel_id",
          common_by_reference::no_mutation{});
      define_kernel<k_name>(queue);
      define_kernel<k_name_other>(queue);
    });

    /** Check that two sycl::kernel_ids referring to the same kernel name are
     *  equal
     */
    run_verification<by_handler>(log, [&](sycl::handler &cgh) {
      using k_name = class two_kernel_ids_same_kernel;
      sycl::kernel_id k_id = sycl::get_kernel_id<k_name>();
      sycl::kernel_id k_id_same = sycl::get_kernel_id<k_name>();
      if (k_id != k_id_same) {
        FAIL(log,
             "Two kernel_ids referring to the same kernel name should be"
             " equal");
      }
      // Dummy kernel
      cgh.single_task<k_name>([] {});
    });
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

}  // namespace TEST_NAMESPACE
