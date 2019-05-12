/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2018 by Codeplay Software LTD. All Rights Reserved.
//
//  This file is a common utility for the implementation of
//  accessor_constructors.cpp and accessor_api.cpp.
//
**************************************************************************/
#ifndef SYCL_1_2_1_TESTS_ACCESSOR_ACCESSOR_UTILITY_H
#define SYCL_1_2_1_TESTS_ACCESSOR_ACCESSOR_UTILITY_H

#include "../common/common.h"
#include <stdexcept>
#include <sstream>
#include <utility>
#include <vector>

namespace accessor_utility {
struct user_struct;
}  // namespace accessor_utility

namespace {
/**
 * @brief Display the name of the user_struct type
 * @param Dummy parameter
 * @return String representing the user_struct type
 */
inline cl::sycl::string_class type_to_string(
    const accessor_utility::user_struct&) {
  return "user_struct";
}
}  // namespace

namespace accessor_utility {

/**
 * @brief Tag used in cases where the dimension doesn't matter
 */
struct generic_dim_tag {};

/**
 * @brief Constructs a type that can determine whether (dim == 0)
 * @tparam dim Number of dimensions to check
 */
template <int dim>
struct is_zero_dim : std::integral_constant<bool, (dim == 0)>,
                     generic_dim_tag {};

/**
 * @brief Tag to use when (dim == 0)
 */
using zero_dim_tag = is_zero_dim<0>;

/**
 * @brief Helper alias for identifying the number of dimensions
 *        required for storing data
 * @tparam dim Number of dimensions. Can be zero, in which case the number of
 *         data dimensions is 1.
 */
template <int dim>
using data_dim = std::integral_constant<int, ((dim == 0) ? 1 : dim)>;

/**
 * @brief Alias to SYCL id type to use
 * @tparam dim Number of dimensions. Can be zero, in which case the number of
 *         dimensions is increased to 1.
 */
template <int dim>
using sycl_id_t = cl::sycl::id<data_dim<dim>::value>;

/**
 * @brief Alias to SYCL range type to use
 * @tparam dim Number of dimensions. Can be zero, in which case the number of
 *         dimensions is increased to 1.
 */
template <int dim>
using sycl_range_t = cl::sycl::range<data_dim<dim>::value>;

/**
 * @brief Alias to SYCL buffer type to use
 * @tparam dim Number of dimensions. Can be zero, in which case the number of
 *         dimensions is increased to 1.
 */
template <typename T, int dim>
using buffer_t = cl::sycl::buffer<T, data_dim<dim>::value>;

/**
 * @brief The SYCL access mode to use for storing errors inside kernels
 */
static constexpr auto errorMode = cl::sycl::access::mode::write;

/**
 * @brief The SYCL buffer type to use for storing errors inside kernels
 */
using error_buffer_t = cl::sycl::buffer<int, 1>;

/** user defined struct that is used in accessor tests
*/
struct user_struct {
  float a;
  int b;
  char c;

  using element_type = int;

  user_struct() : a(0), b(0), c(0){};

  user_struct(int val) : a(0), b(val), c(0) {}

  element_type operator[](size_t index) const { return b; }

  friend bool operator==(const user_struct& lhs, const user_struct& rhs) {
    static constexpr auto eps = 1e-4f;
    return (((lhs.a + eps > rhs.a) && (lhs.a < rhs.a + eps)) &&
            (lhs.b == rhs.b) && (lhs.c == rhs.c));
  }
};

/**
 * @brief Namespace that defines access target tags
 */
namespace acc_target_tag {
struct generic {};
struct host : generic {};
struct local : generic {};
struct constant : generic {};

template <cl::sycl::access::target target>
struct get_helper {
  using type = generic;
};
template <>
struct get_helper<cl::sycl::access::target::host_buffer> {
  using type = host;
};
template <>
struct get_helper<cl::sycl::access::target::host_image> {
  using type = host;
};
template <>
struct get_helper<cl::sycl::access::target::local> {
  using type = local;
};
template <>
struct get_helper<cl::sycl::access::target::constant_buffer> {
  using type = constant;
};

template <cl::sycl::access::target target>
using get_helper_t = typename get_helper<target>::type;

/**
 * @brief Retrieves the tag associated with the access target
 * @tparam target The SYCL access target to get the tag for
 * @return Instance of the tag
 */
template <cl::sycl::access::target target>
auto get() -> decltype(get_helper_t<target>{}) {
  return get_helper_t<target>{};
}

}  // namespace acc_target_tag

/**
 * @brief Namespace that defines access mode tags
 */
namespace acc_mode_tag {
struct generic {};
struct write_only : generic {};
struct read_only : generic {};

template <cl::sycl::access::mode mode>
struct get_helper {
  using type = generic;
};
template <>
struct get_helper<cl::sycl::access::mode::read> {
  using type = read_only;
};
template <>
struct get_helper<cl::sycl::access::mode::write> {
  using type = write_only;
};
template <>
struct get_helper<cl::sycl::access::mode::discard_write> {
  using type = write_only;
};

template <cl::sycl::access::mode mode>
using get_helper_t = typename get_helper<mode>::type;

/**
 * @brief Retrieves the tag associated with the access mode
 * @tparam mode The SYCL access mode to get the tag for
 * @return Instance of the tag
 */
template <cl::sycl::access::mode mode>
auto get() -> decltype(get_helper_t<mode>{}) {
  return get_helper_t<mode>{};
}

}  // namespace acc_mode_tag

/**
 * @brief Namespace that defines tags based on the access target
 *        and whether the accessor is a placeholder
 */
namespace acc_type_tag {
struct generic {};
struct buffer : generic {};
struct constant_buffer : buffer {};
struct image : generic {};
struct image_array : image {};
struct local : acc_target_tag::local, generic {};
struct host : acc_target_tag::host, generic {};
struct placeholder : buffer {};
struct constant_placeholder : placeholder {};
struct global_placeholder : placeholder {};
struct invalid {};

template <cl::sycl::access::target target,
          cl::sycl::access::placeholder placeholder>
struct get_helper {
  using type = generic;
};
template <cl::sycl::access::placeholder placeholder>
struct get_helper<cl::sycl::access::target::global_buffer, placeholder> {
  using type = buffer;
};
template <cl::sycl::access::placeholder placeholder>
struct get_helper<cl::sycl::access::target::constant_buffer, placeholder> {
  using type = constant_buffer;
};
template <>
struct get_helper<cl::sycl::access::target::host_buffer,
                  cl::sycl::access::placeholder::false_t> {
  using type = host;
};
template <>
struct get_helper<cl::sycl::access::target::host_image,
                  cl::sycl::access::placeholder::false_t> {
  using type = host;
};
template <>
struct get_helper<cl::sycl::access::target::image,
                  cl::sycl::access::placeholder::false_t> {
  using type = image;
};
template <>
struct get_helper<cl::sycl::access::target::image_array,
                  cl::sycl::access::placeholder::false_t> {
  using type = image_array;
};
template <>
struct get_helper<cl::sycl::access::target::local,
                  cl::sycl::access::placeholder::false_t> {
  using type = local;
};
template <cl::sycl::access::target target>
struct get_helper<target, cl::sycl::access::placeholder::true_t> {
  using type = invalid;
};
template <>
struct get_helper<cl::sycl::access::target::global_buffer,
                  cl::sycl::access::placeholder::true_t> {
  using type = global_placeholder;
};
template <>
struct get_helper<cl::sycl::access::target::constant_buffer,
                  cl::sycl::access::placeholder::true_t> {
  using type = constant_placeholder;
};

template <cl::sycl::access::target target,
          cl::sycl::access::placeholder placeholder>
using get_helper_t = typename get_helper<target, placeholder>::type;

/**
 * @brief Retrieves the tag associated with the access mode
 *        and whether the accessor is a placeholder
 * @tparam mode The SYCL target mode to get the tag for
 * @tparam placeholder Whether the accessor is a placeholder or not
 * @return Instance of the tag
 */
template <cl::sycl::access::target target,
          cl::sycl::access::placeholder placeholder =
              cl::sycl::access::placeholder::false_t>
auto get() -> decltype(get_helper_t<target, placeholder>{}) {
  return get_helper_t<target, placeholder>{};
};

}  // namespace acc_type_tag

/**
 * @brief Namespace that defines tags based on the number of dimensions
 */
namespace acc_dims_tag {
/**
 * @brief Tag that accepts any number of dimensions
 */
struct generic {};

/**
 * @brief Unique tag based on the number of dimensions
 * @tparam dims number of dimensions
 */
template <int dims>
struct num_dims : std::integral_constant<int, dims>, generic {};

/**
 * @brief Retrieves the tag associated with the number of dimensions
 * @tparam dims number of dimensions
 * @return Instance of the tag
 */
template <int dims>
num_dims<dims> get() {
  return {};
}

}  // namespace acc_dims_tag

namespace detail {
namespace sycl = cl::sycl;
namespace access = cl::sycl::access;

/**
 * @brief Constant type indicating whether the access target defines host access
 * @tparam target Accessor target to inspect
 */
template <cl::sycl::access::target target>
using is_host_target =
    std::integral_constant<bool,
                           ((target == cl::sycl::access::target::host_buffer) ||
                            (target == cl::sycl::access::target::host_image))>;

/**
 * @brief Helper struct for constructing an accessor
 * @tparam T Underlying type of the accessor
 * @tparam dims Number of accessor dimensions
 * @tparam mode Access mode used
 * @tparam target Access target used
 * @tparam placeholder Whether the accessor is a placeholder
 */
template <typename T, int dims, cl::sycl::access::mode mode,
          cl::sycl::access::target target,
          cl::sycl::access::placeholder placeholder>
struct accessor_factory {
  using acc_t = cl::sycl::accessor<T, dims, mode, target, placeholder>;

  static acc_t make_local_generic(const sycl_range_t<dims>& rng,
                                  cl::sycl::handler& cgh, generic_dim_tag) {
    return acc_t(rng, cgh);
  }

  static acc_t make_local_generic(const sycl_range_t<dims>& rng,
                                  cl::sycl::handler& cgh, zero_dim_tag) {
    return acc_t(cgh);
  }

  template <typename buf_t>
  static acc_t make_generic(buf_t& buf, const sycl_range_t<dims>* rng,
                            const sycl_id_t<dims>* accessOffset,
                            cl::sycl::handler* cgh, acc_type_tag::generic,
                            generic_dim_tag) {
    assert(cgh != nullptr);
    if ((rng == nullptr) && (accessOffset == nullptr)) {
      return acc_t(buf, *cgh);
    } else if ((rng != nullptr) && (accessOffset == nullptr)) {
      return acc_t(buf, *cgh, *rng);
    } else {
      assert(rng != nullptr);
      assert(accessOffset != nullptr);
      return acc_t(buf, *cgh, *rng, *accessOffset);
    }
  }

  template <typename buf_t>
  static acc_t make_generic(buf_t& buf, const sycl_range_t<dims>* /*rng*/,
                            const sycl_id_t<dims>* /*accessOffset*/,
                            cl::sycl::handler* cgh, acc_type_tag::generic,
                            zero_dim_tag) {
    assert(cgh != nullptr);
    return acc_t(buf, *cgh);
  }

  template <typename buf_t>
  static acc_t make_generic(buf_t& buf, const sycl_range_t<dims>* rng,
                            const sycl_id_t<dims>* accessOffset,
                            cl::sycl::handler* cgh, acc_type_tag::host,
                            generic_dim_tag) {
    if ((rng == nullptr) && (accessOffset == nullptr)) {
      return acc_t(buf);
    } else if ((rng != nullptr) && (accessOffset == nullptr)) {
      return acc_t(buf, *rng);
    } else {
      assert(rng != nullptr);
      assert(accessOffset != nullptr);
      return acc_t(buf, *rng, *accessOffset);
    }
  }

  template <typename buf_t>
  static acc_t make_generic(buf_t& buf, const sycl_range_t<dims>* /*rng*/,
                            const sycl_id_t<dims>* /*accessOffset*/,
                            cl::sycl::handler* cgh, acc_type_tag::host,
                            zero_dim_tag) {
    return acc_t(buf);
  }

  static acc_t make_generic(buffer_t<T, dims>& buf,
                            const sycl_range_t<dims>* rng,
                            const sycl_id_t<dims>* accessOffset,
                            cl::sycl::handler* cgh, acc_type_tag::placeholder,
                            generic_dim_tag) {
    // Placeholder accessors use the same constructors as host ones
    return make_generic(buf, rng, accessOffset, cgh, acc_type_tag::host{},
                        generic_dim_tag{});
  }

  static acc_t make_generic(buffer_t<T, dims>& buf,
                            const sycl_range_t<dims>* rng,
                            const sycl_id_t<dims>* /*accessOffset*/,
                            cl::sycl::handler* cgh, acc_type_tag::placeholder,
                            zero_dim_tag) {
    return acc_t(buf);
  }

  static acc_t make_generic(buffer_t<T, dims>& buf,
                            const sycl_range_t<dims>* rng,
                            const sycl_id_t<dims>* /*accessOffset*/,
                            cl::sycl::handler* cgh, acc_type_tag::local,
                            generic_dim_tag) {
    assert(rng != nullptr);
    assert(cgh != nullptr);
    return acc_t(*rng, *cgh);
  }
};

}  // namespace detail

/**
 * @brief Helper function for constructing an accessor
 * @tparam T Underlying type of the accessor
 * @tparam dims Number of accessor dimensions
 * @tparam mode Access mode used
 * @tparam target Access target used
 * @tparam placeholder Whether the accessor is a placeholder
 * @tparam Args Types of arguments passed to the accessor constructor
 * @param args Arguments passed to the accessor constructor
 * @return Fully constructed accessor
 */
template <typename T, int dims, cl::sycl::access::mode mode,
          cl::sycl::access::target target,
          cl::sycl::access::placeholder placeholder =
              cl::sycl::access::placeholder::false_t,
          typename... Args>
cl::sycl::accessor<T, dims, mode, target, placeholder> make_accessor(
    Args&&... args) {
  return cl::sycl::accessor<T, dims, mode, target, placeholder>(
      std::forward<Args>(args)...);
}

/**
 * @brief Helper function for constructing a local accessor
 * @tparam T Underlying type of the accessor
 * @tparam dims Number of accessor dimensions
 * @tparam mode Access mode used
 * @tparam Args Types of arguments passed to the accessor constructor
 * @param args Arguments passed to the accessor constructor
 * @return Fully constructed local accessor
 */
template <typename T, int dims, cl::sycl::access::mode mode>
cl::sycl::accessor<T, dims, mode, cl::sycl::access::target::local>
make_local_accessor_generic(const sycl_range_t<dims>& rng,
                            cl::sycl::handler& cgh) {
  return detail::accessor_factory<T, dims, mode,
                                  cl::sycl::access::target::local,
                                  cl::sycl::access::placeholder::false_t>::
      make_local_generic(rng, cgh, is_zero_dim<dims>{});
}

/**
 * @brief Helper function for constructing a buffer accessor
 *        in a generic way using an optional range and handler
 * @tparam dims Number of accessor dimensions
 * @tparam mode Access mode used
 * @tparam target Access target used
 * @tparam placeholder Whether the accessor is a placeholder
 * @tparam T Underlying type of the accessor, deduced from the buffer argument
 * @param buf The buffer to construct the accessor from
 * @param rng Optional range to use on construction, can be null
 * @param accessOffset Optional access offset to use on construction,
 *        can be null
 * @param cgh Optional handler to use on construction, can be null
 * @return Fully constructed buffer accessor
 */
template <int dims, cl::sycl::access::mode mode,
          cl::sycl::access::target target,
          cl::sycl::access::placeholder placeholder =
              cl::sycl::access::placeholder::false_t,
          typename T>
cl::sycl::accessor<T, dims, mode, target, placeholder> make_accessor_generic(
    buffer_t<T, dims>& buf,
    typename std::add_pointer<const sycl_range_t<dims>>::type rng,
    typename std::add_pointer<const sycl_id_t<dims>>::type accessOffset,
    cl::sycl::handler* cgh) {
  return detail::accessor_factory<T, dims, mode, target, placeholder>::
      make_generic(buf, rng, accessOffset, cgh,
                   acc_type_tag::get<target, placeholder>(),
                   is_zero_dim<dims>{});
}

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
  static constexpr auto value = (target == cl::sycl::access::target::local);
};

/** Convenient compile-time evaluation to determine if an accessor is an buffer
 *  accessor (of sorts)
 */
template <cl::sycl::access::target target>
struct is_buffer {
  static constexpr auto value =
      !is_image<target>::value && !is_local<target>::value;
};

/**
 * @brief Logs the type of the accessor currently under test
 * @tparam T Underlying type of the accessor
 * @tparam dims Number of accessor dimensions
 * @tparam mode Access mode used
 * @tparam target Access target used
 * @tparam placeholder Whether the accessor is a placeholder
 * @param functionName String indicating what is being tested
 * @param log The logger object
 */
template <typename T, int dims, cl::sycl::access::mode mode,
          cl::sycl::access::target target,
          cl::sycl::access::placeholder placeholder =
              cl::sycl::access::placeholder::false_t>
void log_accessor(const cl::sycl::string_class& functionName,
                  sycl_cts::util::logger& log) {
  std::stringstream stream;
  if (!functionName.empty()) {
    stream << functionName << " -> ";
  }
  stream << "accessor<" << type_to_string(T{}) << ", " << dims << ", mode{"
         << static_cast<int>(mode) << "}, target{" << static_cast<int>(target)
         << "}";
  if (!is_image<target>::value) {
    stream << ", placeholder{"
           << (placeholder == cl::sycl::access::placeholder::true_t) << "}";
  }
  stream << ">";
  const auto message = stream.str();
  log.note(message.c_str());
}

/**
 * @brief Logs the current test check
 * @tparam T Underlying type being tested
 * @tparam dims Number of dimensions being tested
 * @param functionName String indicating what is being tested
 * @param log The logger object
 */
template <typename T, int dims>
void log_check(const cl::sycl::string_class& functionName,
               sycl_cts::util::logger& log) {
  std::stringstream stream;
  stream << functionName << " -> accessor<" << type_to_string(T{}) << ", "
         << dims << ", ...>";
  const auto message = stream.str();
  log.note(message.c_str());
}

}  // namespace accessor_utility

namespace {

using namespace accessor_utility;

/** creates a list of ids
*/
template <int dims>
cl::sycl::vector_class<cl::sycl::id<dims>> create_id_list(
    const cl::sycl::range<dims>& r);

/** creates a list of ids (specialization for 1 dimension)
*/
template <>
cl::sycl::vector_class<cl::sycl::id<1>> create_id_list<1>(
    const cl::sycl::range<1>& r) {
  cl::sycl::vector_class<cl::sycl::id<1>> ret;
  for (size_t i = 0; i < r[0]; ++i) {
    ret.emplace_back(i);
  }
  return ret;
}

/** creates a list of ids (specialization for 2 dimension)
*/
template <>
cl::sycl::vector_class<cl::sycl::id<2>> create_id_list<2>(
    const cl::sycl::range<2>& r) {
  cl::sycl::vector_class<cl::sycl::id<2>> ret;
  for (size_t i = 0; i < r[0]; ++i) {
    for (size_t j = 0; j < r[1]; ++j) {
      ret.emplace_back(i, j);
    }
  }
  return ret;
}

/** creates a list of ids (specialization for 3 dimension)
*/
template <>
cl::sycl::vector_class<cl::sycl::id<3>> create_id_list<3>(
    const cl::sycl::range<3>& r) {
  cl::sycl::vector_class<cl::sycl::id<3>> ret;
  for (size_t i = 0; i < r[0]; ++i) {
    for (size_t j = 0; j < r[1]; ++j) {
      for (size_t k = 0; k < r[2]; ++k) {
        ret.emplace_back(i, j, k);
      }
    }
  }
  return ret;
}

/** tests that two values are equal
*/
template <typename T1>
bool check_elems_equal(const T1& actual, const T1& expected) {
  return (actual == expected);
}

/**
 * @brief Checks if two values are equal, overload for floating point values
 */
inline bool check_elems_equal(float actual, float expected) {
  static constexpr float eps = 1e-4f;
  return ((actual < expected + eps) && (actual + eps > expected));
}

/**
 * @brief Checks if two values are equal, overload for SYCL vectors
 */
template <typename T1, int N>
bool check_elems_equal(const cl::sycl::vec<T1, N>& actual,
                       const cl::sycl::vec<T1, N>& expected) {
  for (int i = 0; i < N; i++) {
    if (!check_elems_equal(getElement(actual, i), getElement(expected, i))) {
      return false;
    }
  }
  return true;
}

/**
 * @brief Checks if two values are equal, overload for comparing a SYCL vec
 *        to a scalar
 */
template <typename T1, int N>
bool check_elems_equal(const cl::sycl::vec<T1, N>& actual, const T1& expected) {
  for (int i = 0; i < N; i++) {
    if (!check_elems_equal(getElement(actual, i), expected)) {
      return false;
    }
  }
  return true;
}

/** tests that an array of linear ids is correct
*/
template <typename T>
bool check_linear_index(sycl_cts::util::logger& log, T* data, size_t size,
                        int mul = 1) {
  for (size_t i = 0; i < size; i++) {
    if (!CHECK_VALUE(log, check_elems_equal(data[i], static_cast<T>(i * mul)),
                     true, i)) {
      return false;
    }
  }
  return true;
}

/**
 * @brief Retrieves the common value for testing a zero-dim accessor
 * @tparam T Underlying type of the accessor
 * @return Some value
 */
template <typename T>
T get_zero_dim_buffer_value() {
  return static_cast<T>(47);
}

/**
 * @brief Retrieves the input data for a SYCL buffer
 * @tparam T Underlying type of the buffer
 * @param count Number of elements of type T to allocate
 * @param dims Number of accessor dimensions
 * @param useIndexes Whether to initialize data with indexes.
 *        If (dims == 0), the data will be initialized using the common value
 *        for testing a zero-dim accessor.
 *        If false, data will be zero initialized.
 * @return Initialized data container
 */
template <typename T>
cl::sycl::vector_class<T> get_buffer_input_data(size_t count, int dims,
                                                bool useIndexes = true) {
  auto data = cl::sycl::vector_class<T>(count);
  if (useIndexes) {
    for (size_t i = 0; i < count; ++i) {
      data[i] = T(i);
    }
    if (dims == 0) {
      data[0] = get_zero_dim_buffer_value<T>();
    }
  } else {
    std::fill(std::begin(data), std::end(data), 0);
  }
  return data;
}

/**
 * @brief Retrieves the input data for a SYCL buffer that will be used
 *        for storing error in a kernel
 * @param count Number of error categories
 * @return Zero-initialized data container
 */
cl::sycl::vector_class<int> get_error_data(size_t count) {
  static constexpr int dims = 1;
  static constexpr bool useIndexes = false;
  return get_buffer_input_data<int>(count, dims, useIndexes);
}

}  // namespace

#endif  // SYCL_1_2_1_TESTS_ACCESSOR_ACCESSOR_UTILITY_H
