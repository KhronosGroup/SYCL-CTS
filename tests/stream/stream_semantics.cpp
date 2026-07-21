/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2023 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
*******************************************************************************/

#include "../common/common.h"
#include "../common/semantics_reference.h"

struct storage {
  std::size_t size;
  std::size_t max_statement_size;

  explicit storage(const sycl::stream& stream)
      : size(stream.size()),
        max_statement_size(stream.get_max_statement_size()) {}

  bool check(const sycl::stream& stream) const {
    return (stream.size() == size &&
            stream.get_max_statement_size() == max_statement_size);
  }
};

TEST_CASE("stream common reference semantics (host)", "[stream]") {
  auto queue = sycl_cts::util::get_cts_object::queue();
  queue.submit([&](sycl::handler& cgh) {
    sycl::stream stream{0, 0, cgh};
    common_reference_semantics::check_host<storage>(stream, "stream");
  });
}

TEST_CASE("stream common reference semantics (kernel)", "[stream]") {
  common_reference_semantics::check_kernel<storage, sycl::stream>(
      [&](sycl::handler& cgh) {
        return sycl::stream{0, 0, cgh};
      },
      "stream");
}
