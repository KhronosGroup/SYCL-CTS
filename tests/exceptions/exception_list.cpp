/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for sycl::exception_list
//
*******************************************************************************/

#include <catch2/catch_template_test_macros.hpp>

#include "../common/common.h"

using namespace sycl_cts;

TEST_CASE("Check sycl::exception_list member types", "[exception]") {
  using sycl::exception_list;
  CHECK(std::is_same_v<exception_list::value_type, std::exception_ptr>);
  CHECK(std::is_same_v<exception_list::reference, exception_list::value_type&>);
  CHECK(std::is_same_v<exception_list::const_reference,
                       const exception_list::value_type&>);
  CHECK(std::is_same_v<exception_list::size_type, std::size_t>);
}

TEST_CASE(
    "Check that sycl::exception_list iterators satisfy LegacyForwardIterator "
    "requirements",
    "[exception]") {
  using It = sycl::exception_list::iterator;
  CHECK(std::is_base_of_v<std::forward_iterator_tag,
                          std::iterator_traits<It>::iterator_category>);
  using ConstIt = sycl::exception_list::iterator;
  CHECK(std::is_base_of_v<std::forward_iterator_tag,
                          std::iterator_traits<ConstIt>::iterator_category>);
}

TEST_CASE("Check default-constructed sycl::exception_list", "[exception]") {
  sycl::exception_list list;
  CHECK(list.size() == 0);
  CHECK(list.begin() == list.end());
}
