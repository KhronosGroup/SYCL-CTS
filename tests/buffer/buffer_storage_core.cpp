/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
// Provides buffer storage methods tests
//
*******************************************************************************/

#include "../common/common.h"
#include "../common/type_coverage.h"
#include "../common/type_list.h"
#include "buffer_type_list.h"
#include "buffer_storage_common.h"

#define TEST_NAME buffer_storage_core

namespace TEST_NAMESPACE {
using namespace sycl_cts;

/**
 * test cl::sycl::buffer storage methods
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
#ifdef SYCL_CTS_ENABLE_FULL_CONFORMANCE
      for_all_types_and_vectors<
          buffer_storage_common::check_buffer_storage_for_type>(
          get_cts_types::vector_types, log);
#ifdef INT8_MAX
      for_type_and_vectors<buffer_storage_common::check_buffer_storage_for_type,
                           std::int8_t>(log, "std::int8_t");
#endif
#ifdef UINT8_MAX
      for_type_and_vectors<buffer_storage_common::check_buffer_storage_for_type,
                           std::uint8_t>(log, "std::uint8_t");
#endif
#ifdef INT16_MAX
      for_type_and_vectors<buffer_storage_common::check_buffer_storage_for_type,
                           std::int16_t>(log, "std::int16_t");
#endif
#ifdef UINT16_MAX
      for_type_and_vectors<buffer_storage_common::check_buffer_storage_for_type,
                           std::uint16_t>(log, "std::uint16_t");
#endif
#ifdef INT32_MAX
      for_type_and_vectors<buffer_storage_common::check_buffer_storage_for_type,
                           std::int32_t>(log, "std::int32_t");
#endif
#ifdef UINT32_MAX
      for_type_and_vectors<buffer_storage_common::check_buffer_storage_for_type,
                           std::uint32_t>(log, "std::uint32_t");
#endif
#ifdef INT64_MAX
      for_type_and_vectors<buffer_storage_common::check_buffer_storage_for_type,
                           std::int64_t>(log, "std::int64_t");
#endif
#ifdef UINT64_MAX
      for_type_and_vectors<buffer_storage_common::check_buffer_storage_for_type,
                           std::uint64_t>(log, "std::uint64_t");
#endif
#else
      for_all_types_and_vectors<
          buffer_storage_common::check_buffer_storage_for_type>(
          get_buffer_types::vector_types, log);
#endif // SYCL_CTS_ENABLE_FULL_CONFORMANCE
      for_all_types<buffer_storage_common::check_buffer_storage_for_type>(
          get_buffer_types::scalar_types, log);
    } catch (cl::sycl::exception e) {
      log_exception(log, e);
      cl::sycl::string_class errorMsg =
          "a SYCL exception was caught: " + cl::sycl::string_class(e.what());
      FAIL(log, errorMsg.c_str());
    }
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

} // namespace TEST_NAMESPACE
