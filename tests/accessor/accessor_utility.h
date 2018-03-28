/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
//  This file is a common utility for the implementation of
//  accessor_constructors.cpp and accessor_api.cpp.
//
**************************************************************************/
#ifndef SYCL_1_2_1_TESTS_ACCESSOR_ACCESSOR_UTILITY_H
#define SYCL_1_2_1_TESTS_ACCESSOR_ACCESSOR_UTILITY_H

#include "../common/common.h"
#include <stdexcept>
#include <utility>

namespace TEST_NAMESPACE {
/**
 * @brief Common error string.
 */
constexpr const char* internal_cts_error =
    "Internal CTS error. Please report this to Khronos immediately, "
    "including the source that generated this diagnostic, and the section "
    "that generates this diagnostic.";

namespace detail {
namespace sycl = cl::sycl;
namespace access = cl::sycl::access;

template <typename T, int dims, access::mode mode, access::target target,
          access::placeholder placeholder = access::placeholder::false_t>
class accessor_factory {
  using accessor_t = sycl::accessor<T, dims, mode, target, placeholder>;
  using buffer_t = sycl::buffer<T, dims ? dims : 1>;
  using range_t = sycl::range<dims ? dims : 1>;
  using id_t = sycl::id<dims ? dims : 1>;

  template <access::target target2>
  using expected_target = std::integral_constant<access::target, target2>;

  using requires_global_buffer = expected_target<access::target::global_buffer>;
  using requires_constant_buffer =
      expected_target<access::target::constant_buffer>;
  using requires_local = expected_target<access::target::local>;
  using requires_image = expected_target<access::target::image>;
  using requires_host_buffer = expected_target<access::target::host_buffer>;
  using requires_host_image = expected_target<access::target::host_image>;
  using requires_image_array = expected_target<access::target::image_array>;

  template <access::placeholder placeholder2>
  using expected_placeholder =
      std::integral_constant<access::placeholder, placeholder2>;
  using requires_non_placeholder_t =
      expected_placeholder<access::placeholder::false_t>;

  static constexpr auto actual_target = expected_target<target>{};
  static constexpr auto actual_placeholder =
      expected_placeholder<placeholder>{};

 public:
  accessor_t operator()(buffer_t& b) const
  // requires target == access::target::host_buffer
  {
    return impl(b, actual_target, actual_placeholder);
  }

  accessor_t operator()(buffer_t& b, sycl::handler& h) const
  // requires target == access::target::host_buffer ||
  //   target == access::target::constant_buffer
  {
    return impl(b, h, actual_target, actual_placeholder);
  }

  accessor_t operator()(buffer_t& b, range_t r, id_t o = {}) const {
    return impl(b, r, o, actual_target, actual_placeholder);
  }

  accessor_t operator()(buffer_t& b, sycl::handler& h, range_t r,
                        id_t o = {}) const {
    return impl(b, h, r, o, actual_target, actual_placeholder);
  }

  accessor_t operator()(sycl::handler& h) const {
    return impl(h, actual_target, actual_placeholder);
  }

  accessor_t operator()(range_t r, sycl::handler& h) const {
    return impl(r, h, actual_target, actual_placeholder);
  }

  template <typename A>
  accessor_t operator()(sycl::image<dims, A>& i) const {
    return impl(i, actual_target);
  }

  template <typename A>
  accessor_t operator()(sycl::image<dims, A>& i, sycl::handler& h) const {
    return impl(i, actual_target);
  }

  template <typename A>
  accessor_t operator()(sycl::image<dims + 1, A>& i, sycl::handler& h) const {
    return impl(i, h, actual_target);
  }

 private:
  accessor_t impl(buffer_t& b, requires_host_buffer,
                  requires_non_placeholder_t) const {
    return accessor_t(b);
  }

  template <typename B, typename P,
            REQUIRES(std::is_same<B, requires_constant_buffer>::value ||
                     std::is_same<B, requires_global_buffer>::value)>
  accessor_t impl(buffer_t& b, sycl::handler& h, B, P) {
    return accessor_t(b, h);
  }

  accessor_t impl(buffer_t& b, range_t r, id_t& o, requires_host_buffer,
                  requires_non_placeholder_t) const {
    return accessor_t(b, r, o);
  }

  template <typename B, typename P,
            REQUIRES(std::is_same<B, requires_global_buffer>::value ||
                     std::is_same<B, requires_constant_buffer>::value)>
  accessor_t impl(buffer_t& b, sycl::handler& h, range_t r, id_t o, B,
                  P) const {
    return accessor_t(b, h, r, o);
  }

  template <int d, REQUIRES(d == 0)>
  accessor_t impl(sycl::handler& h, requires_local,
                  requires_non_placeholder_t) const {
    return accessor_t(h);
  }

  template <int d, REQUIRES(d > 0)>
  accessor_t impl(range_t& r, sycl::handler& h, requires_local,
                  requires_non_placeholder_t) const {
    return accessor_t(r, h);
  }

  template <typename A>
  accessor_t impl(sycl::image<dims, A>& i, requires_host_image) const {
    return accessor_t(i);
  }

  template <typename A>
  accessor_t impl(sycl::image<dims, A>& i, sycl::handler& h,
                  requires_image) const {
    return accessor_t(i, h);
  }

  template <int d, typename A, REQUIRES(dims <= 3)>
  accessor_t impl(sycl::image<d, A>& i, sycl::handler& h,
                  requires_image_array) const {
    return accessor_t(i, h);
  }

  // metaprogramming should prevent this from being compiled (or executed if it
  // is compiled). The exception is merely a safeguard.
  template <typename... Args>
  [[noreturn]] accessor_t impl(Args&&...) const {
    throw std::logic_error{internal_cts_error};
  }
};
}  // namespace detail

template <typename T, int dims, cl::sycl::access::mode mode,
          cl::sycl::access::target target,
          cl::sycl::access::placeholder placeholder =
              cl::sycl::access::placeholder::false_t,
          typename... Args>
cl::sycl::accessor<T, dims, mode, target, placeholder> make_accessor(
    Args&&... args) {
  static const auto factory =
      detail::accessor_factory<T, dims, mode, target, placeholder>{};
  return factory(std::forward<Args>(args)...);
}

/** eliminates the need for the ugly `typename T::value_type`
  * taken from Ranges TS
  */
template <typename T>
using value_type_t = typename T::value_type;

/** Convenient compile-time evaluation to determine if an accessor is an image
 *  accessor (of sorts)
 */
template <cl::sycl::access::target target>
struct is_image {
  static constexpr auto value =
      target == cl::sycl::access::target::image ||
      target == cl::sycl::access::target::host_image ||
      target == cl::sycl::access::target::image_array;
};

/** Convenient compile-time evaluation to determine if an accessor is an local
 *  accessor
 */
template <cl::sycl::access::target target>
struct is_local {
  static constexpr auto value = target == cl::sycl::access::target::local;
};

/** Convenient compile-time evaluation to determine if an accessor is an buffer
 *  accessor (of sorts)
 */
template <cl::sycl::access::target target>
struct is_buffer {
  static constexpr auto value =
      !is_image<target>::value && !is_local<target>::value;
};

}  // namespace accessor_utility__

#endif  // SYCL_1_2_1_TESTS_ACCESSOR_ACCESSOR_UTILITY_H
