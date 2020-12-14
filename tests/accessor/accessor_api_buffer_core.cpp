/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2018 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"
#include "../common/type_coverage.h"
#include "./../../util/math_helper.h"
#include "accessor_api_buffer_common.h"

#define TEST_NAME accessor_api_buffer_core

namespace TEST_NAMESPACE {

using namespace sycl_cts;
using namespace accessor_utility;

/** tests the api for cl::sycl::accessor
*/
class TEST_NAME : public util::test_base {
 public:
  /** return information about this test
  */
  void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute this test
  */
  void run(util::logger &log) override {
    try {
      auto queue = util::get_cts_object::queue();

#ifndef SYCL_CTS_EXTENSIVE_MODE
      // Specific set of types to cover during ordinary compilation

      const auto vector_types = named_type_pack<int>({"int"});
      const auto scalar_types =
          named_type_pack<float,
                          std::size_t,
                          user_struct>({
                          "float",
                          "std::size_t",
                          "user_struct"});
#else
      // Extended type coverage

      const auto vector_types =
          named_type_pack<bool,
                          char, signed char, unsigned char,
                          short, unsigned short,
                          int, unsigned int,
                          long, unsigned long,
                          long long, unsigned long long,
                          float, cl::sycl::cl_float,
                          cl::sycl::byte,
                          cl::sycl::cl_bool,
                          cl::sycl::cl_char, cl::sycl::cl_uchar,
                          cl::sycl::cl_short, cl::sycl::cl_ushort,
                          cl::sycl::cl_int, cl::sycl::cl_uint,
                          cl::sycl::cl_long, cl::sycl::cl_ulong>{
                          "bool",
                          "char", "signed char", "unsigned char",
                          "short", "unsigned short",
                          "int", "unsigned int",
                          "long", "unsigned long",
                          "long long", "unsigned long long",
                          "float", "cl::sycl::cl_float",
                          "cl::sycl::byte",
                          "cl::sycl::cl_bool",
                          "cl::sycl::cl_char", "cl::sycl::cl_uchar",
                          "cl::sycl::cl_short", "cl::sycl::cl_ushort",
                          "cl::sycl::cl_int", "cl::sycl::cl_uint",
                          "cl::sycl::cl_long", "cl::sycl::cl_ulong"};
      const auto scalar_types =
          named_type_pack<std::size_t, user_struct>{
                          "std::size_t", "user_struct"};

#ifdef INT8_MAX
      if (!std::is_same<std::int8_t, cl::sycl::cl_char>::value) {
        for_type_and_vectors<check_buffer_accessor_api_type, std::int8_t>(
            log, queue, "std::int8_t");
      }
#endif
#ifdef UINT8_MAX
      if (!std::is_same<std::uint8_t, cl::sycl::cl_uchar>::value) {
        for_type_and_vectors<check_buffer_accessor_api_type, std::uint8_t>(
            log, queue, "std::uint8_t");
      }
#endif
#ifdef INT16_MAX
      if (!std::is_same<std::int16_t, cl::sycl::cl_short>::value) {
        for_type_and_vectors<check_buffer_accessor_api_type, std::int16_t>(
            log, queue, "std::int16_t");
      }
#endif
#ifdef UINT16_MAX
      if (!std::is_same<std::uint16_t, cl::sycl::cl_ushort>::value) {
        for_type_and_vectors<check_buffer_accessor_api_type, std::uint16_t>(
            log, queue, "std::uint16_t");
      }
#endif
#ifdef INT32_MAX
      if (!std::is_same<std::int32_t, cl::sycl::cl_int>::value) {
        for_type_and_vectors<check_buffer_accessor_api_type, std::int32_t>(
            log, queue, "std::int32_t");
      }
#endif
#ifdef UINT32_MAX
      if (!std::is_same<std::uint32_t, cl::sycl::cl_uint>::value) {
        for_type_and_vectors<check_buffer_accessor_api_type, std::uint32_t>(
            log, queue, "std::uint32_t");
      }
#endif
#ifdef INT64_MAX
      if (!std::is_same<std::int64_t, cl::sycl::cl_long>::value) {
        for_type_and_vectors<check_buffer_accessor_api_type, std::int64_t>(
            log, queue, "std::int64_t");
      }
#endif
#ifdef UINT64_MAX
      if (!std::is_same<std::uint64_t, cl::sycl::cl_ulong>::value) {
        for_type_and_vectors<check_buffer_accessor_api_type, std::uint64_t>(
            log, queue, "std::uint64_t");
      }
#endif

#endif // SYCL_CTS_EXTENSIVE_MODE

      for_all_types_and_vectors<check_buffer_accessor_api_type>(
           vector_types, log, queue);

      for_all_types<check_buffer_accessor_api_type>(
          scalar_types, log, queue);

      queue.wait_and_throw();
    } catch (const cl::sycl::exception &e) {
      log_exception(log, e);
      cl::sycl::string_class errorMsg =
          "a SYCL exception was caught: " + cl::sycl::string_class(e.what());
      FAIL(log, errorMsg.c_str());
    }
  }
};

/** register this test with the test_collection
*/
util::test_proxy<TEST_NAME> proxy;

}  // namespace TEST_NAMESPACE
