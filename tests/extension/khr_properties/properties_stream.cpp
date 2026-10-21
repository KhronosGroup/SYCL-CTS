/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2026 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
//  Tests for the stream constructor added by the sycl_khr_properties
//  extension.
//
//  The extension does not define any property that can be used with the
//  stream class and applications cannot define their own properties, so the
//  only PropertyOrList argument that satisfies the constraints is the empty
//  properties list.
//
*******************************************************************************/

#include "khr_properties_common.h"

#include <cstdio>
#include <string>

#if defined(__linux__)
#include <unistd.h>
#endif

namespace khr_properties::tests {

class khr_props_stream_kernel;

constexpr std::size_t total_buffer_size = 1024;
constexpr std::size_t work_item_buffer_size = 256;

template <typename P>
inline constexpr bool can_construct_stream_v =
    std::is_constructible_v<sycl::stream, std::size_t, std::size_t,
                            sycl::handler&, P>;

TEST_CASE("the new stream constructor is constrained on PropertyOrList",
          "[khr_properties][stream]") {
  STATIC_CHECK(can_construct_stream_v<khr::empty_properties_t>);
  STATIC_CHECK_FALSE(can_construct_stream_v<prop::in_order>);
  STATIC_CHECK_FALSE(can_construct_stream_v<prop::enable_profiling>);
  STATIC_CHECK_FALSE(can_construct_stream_v<prop::initialize_to_identity>);
  STATIC_CHECK_FALSE(can_construct_stream_v<queue_props_t>);
  STATIC_CHECK_FALSE(can_construct_stream_v<reduction_props_t>);
  STATIC_CHECK_FALSE(can_construct_stream_v<not_a_property>);
}

#if defined(__linux__)
/**
 * Redirects stdout to a temporary file for the lifetime of the object, so
 * that the output of a sycl::stream can be inspected.
 */
class stdout_capture {
 public:
  stdout_capture() {
    std::fflush(stdout);
    m_file = std::tmpfile();
    m_saved = dup(fileno(stdout));
    if (m_file && m_saved != -1) dup2(fileno(m_file), fileno(stdout));
  }
  ~stdout_capture() { restore(); }

  /** Restores stdout and returns everything that was written to it. */
  std::string restore() {
    if (m_saved == -1 || !m_file) return {};
    std::fflush(stdout);
    dup2(m_saved, fileno(stdout));
    close(m_saved);
    m_saved = -1;
    std::string result;
    std::rewind(m_file);
    char buf[256];
    std::size_t len;
    while ((len = std::fread(buf, 1, sizeof(buf), m_file)) > 0) {
      result.append(buf, len);
    }
    std::fclose(m_file);
    m_file = nullptr;
    return result;
  }

 private:
  std::FILE* m_file = nullptr;
  int m_saved = -1;
};
#endif

TEST_CASE("a stream constructed with an empty properties list",
          "[khr_properties][stream]") {
  auto q = sycl_cts::util::get_cts_object::queue();
  std::size_t size = 0;
  std::size_t wi_size = 0;
  {
#if defined(__linux__)
    stdout_capture capture;
#endif
    q.submit([&](sycl::handler& cgh) {
      sycl::stream os{total_buffer_size, work_item_buffer_size, cgh,
                      khr::empty_properties_t{}};
      size = os.get_size();
      wi_size = os.get_work_item_buffer_size();
      cgh.single_task<khr_props_stream_kernel>(
          [=] { os << "khr_properties stream " << 42 << sycl::endl; });
    });
    q.wait_and_throw();
#if defined(__linux__)
    const std::string output = capture.restore();
    CHECK(output.find("khr_properties stream 42") != std::string::npos);
#endif
  }
  CHECK(size == total_buffer_size);
  CHECK(wi_size == work_item_buffer_size);
}

}  // namespace khr_properties::tests
