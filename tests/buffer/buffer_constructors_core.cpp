/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
// Provides buffer constructors tests
//
*******************************************************************************/

#include "../common/common.h"
#include "../common/type_coverage.h"
#include "../common/type_list.h"
#include "buffer_type_list.h"
#include "buffer_constructors_common.h"

#define TEST_NAME buffer_constructors_core

namespace TEST_NAMESPACE {
using namespace sycl_cts;
/**
 * test sycl::buffer initialization
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

#ifdef SYCL_CTS_ENABLE_FULL_CONFORMANCE
      for_all_types_and_vectors<
          buffer_constructors_common::check_buffer_ctors_for_type>(
          get_cts_types::vector_types, log);
#ifdef INT8_MAX
      for_type_and_vectors<
          buffer_constructors_common::check_buffer_ctors_for_type, std::int8_t>(
          log, "std::int8_t");
#endif
#ifdef UINT8_MAX
      for_type_and_vectors<
          buffer_constructors_common::check_buffer_ctors_for_type,
          std::uint8_t>(log, "std::uint8_t");
#endif
#ifdef INT16_MAX
      for_type_and_vectors<
          buffer_constructors_common::check_buffer_ctors_for_type,
          std::int16_t>(log, "std::int16_t");
#endif
#ifdef UINT16_MAX
      for_type_and_vectors<
          buffer_constructors_common::check_buffer_ctors_for_type,
          std::uint16_t>(log, "std::uint16_t");
#endif
#ifdef INT32_MAX
      for_type_and_vectors<
          buffer_constructors_common::check_buffer_ctors_for_type,
          std::int32_t>(log, "std::int32_t");
#endif
#ifdef UINT32_MAX
      for_type_and_vectors<
          buffer_constructors_common::check_buffer_ctors_for_type,
          std::uint32_t>(log, "std::uint32_t");
#endif
#ifdef INT64_MAX
      for_type_and_vectors<
          buffer_constructors_common::check_buffer_ctors_for_type,
          std::int64_t>(log, "std::int64_t");
#endif
#ifdef UINT64_MAX
      for_type_and_vectors<
          buffer_constructors_common::check_buffer_ctors_for_type,
          std::uint64_t>(log, "std::uint64_t");
#endif
#else
      for_all_types_and_vectors<
          buffer_constructors_common::check_buffer_ctors_for_type>(
          get_buffer_types::vector_types, log);
#endif // SYCL_CTS_ENABLE_FULL_CONFORMANCE
      for_all_types<buffer_constructors_common::check_buffer_ctors_for_type>(
          get_buffer_types::scalar_types, log);
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

} /* namespace TEST_NAMESPACE */
