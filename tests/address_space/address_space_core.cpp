/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provide address_space tests for types that do not require extension
//
*******************************************************************************/

#define TEST_NAME address_space_core

#include "../common/common.h"
#include "address_space_common.h"

#include <cstddef>
#include <cstdint>

namespace TEST_NAMESPACE {
using namespace sycl_cts;

class TEST_NAME : public sycl_cts::util::test_base {
 public:
  void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  void run(util::logger &log) override {
    test_types<int>(log);
    test_types<char>(log);
    test_types<unsigned char>(log);
    test_types<short int>(log);
    test_types<unsigned short int>(log);
    test_types<int>(log);
    test_types<unsigned int>(log);
    test_types<long int>(log);
    test_types<unsigned long int>(log);
    test_types<long long int>(log);
    test_types<unsigned long long int>(log);
    test_types<float>(log);

    test_types<std::byte>(log);
    test_types<std::size_t>(log);

#ifdef INT8_MAX
    test_types<std::int8_t>(log);
#endif
#ifdef UINT8_MAX
    test_types<std::uint8_t>(log);
#endif
#ifdef INT16_MAX
    test_types<std::int16_t>(log);
#endif
#ifdef UINT16_MAX
    test_types<std::uint16_t>(log);
#endif
#ifdef INT32_MAX
    test_types<std::int32_t>(log);
#endif
#ifdef UINT32_MAX
    test_types<std::uint32_t>(log);
#endif
#ifdef INT64_MAX
    test_types<std::int64_t>(log);
#endif
#ifdef UINT64_MAX
    test_types<std::uint64_t>(log);
#endif
  }
};

util::test_proxy<TEST_NAME> proxy;

}  // namespace TEST_NAMESPACE
