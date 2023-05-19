/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2023 The Khronos Group Inc.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
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
