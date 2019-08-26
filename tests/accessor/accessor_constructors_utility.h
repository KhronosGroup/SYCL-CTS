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
#ifndef SYCL_1_2_1_TESTS_ACCESSOR_ACCESSOR_CONSTRUCTORS_UTILITY_H
#define SYCL_1_2_1_TESTS_ACCESSOR_ACCESSOR_CONSTRUCTORS_UTILITY_H

#include "../common/common.h"
#include <array>
#include <sstream>

namespace TEST_NAMESPACE {

/** unique dummy_functor per file
 *  this is a hack until the CMake script is fixed; kill both the alias and the
 *  dummy class once it is fixed
 */
template <typename T, cl::sycl::access::target kTarget>
class dummy_accessor_api_buffer {};
template <typename T, cl::sycl::access::target kTarget>
using dummy_functor = ::dummy_functor<dummy_accessor_api_buffer<T, kTarget>>;

using namespace sycl_cts;

/** Helper function that calculates the amount of elements
 *  of a range
 */
template <size_t dims>
size_t getElementsCount(const cl::sycl::range<dims> &range);

/** Specializations of for getElementsCount each supported
 *  dimensionality
 */
template <>
size_t getElementsCount<1>(const cl::sycl::range<1> &range) {
  return range[0];
}

template <>
size_t getElementsCount<2>(const cl::sycl::range<2> &range) {
  return range[0] * range[1];
}

template <>
size_t getElementsCount<3>(const cl::sycl::range<3> &range) {
  return range[0] * range[1] * range[2];
}

/** Helper function that calculates an id from a size so
 *  that each dimension equals size
 */
template <size_t dims>
cl::sycl::id<dims> getId(const size_t &size);

/** Specializations of for getId each supported
 *  dimensionality
 */
template <>
cl::sycl::id<1> getId<1>(const size_t &size) {
  return cl::sycl::id<1>(size);
}
template <>
cl::sycl::id<2> getId<2>(const size_t &size) {
  return cl::sycl::id<2>(size, size);
}
template <>
cl::sycl::id<3> getId<3>(const size_t &size) {
  return cl::sycl::id<3>(size, size, size);
}

/** Returns the string representation of a SYCL access mode
 */
std::string access_mode_to_string(cl::sycl::access::mode kMode) {
  switch (kMode) {
    case cl::sycl::access::mode::read:
      return "read";
    case cl::sycl::access::mode::write:
      return "write";
    case cl::sycl::access::mode::read_write:
      return "read_write";
    case cl::sycl::access::mode::discard_write:
      return "discard_write";
    case cl::sycl::access::mode::discard_read_write:
      return "discard_read_write";
    case cl::sycl::access::mode::atomic:
      return "atomic";
    default:
      return "";  // or throw an exception here
  }
}

/** Returns the string representation of a SYCL access target
 */
std::string access_target_to_string(cl::sycl::access::target kTarget) {
  switch (kTarget) {
    case cl::sycl::access::target::global_buffer:
      return "global_buffer";
    case cl::sycl::access::target::constant_buffer:
      return "constant_buffer";
    case cl::sycl::access::target::local:
      return "local";
    case cl::sycl::access::target::image:
      return "image";
    case cl::sycl::access::target::host_buffer:
      return "host_buffer";
    case cl::sycl::access::target::host_image:
      return "host_image";
    case cl::sycl::access::target::image_array:
      return "image_array";
    default:
      return "";  // or throw an exception here
  }
}

/** generates an error message containing all required information to trace a
 * failure
 */
std::string get_error_message(size_t dims, cl::sycl::access::mode kMode,
                              cl::sycl::access::target kTarget,
                              cl::sycl::access::placeholder isPlaceholder,
                              std::string operation) {
  std::stringstream ss;
  if (isPlaceholder == cl::sycl::access::placeholder::true_t) {
    ss << "placeholder accessor ";
  } else {
    ss << "accessor ";
  }
  ss << access_target_to_string(kTarget) << " for "
     << access_mode_to_string(kMode) << " " << dims << "d failed in "
     << operation;
  return ss.str();
}

/** checks all available accessor members to ensure no failure occured
 */
template <typename T, size_t dims, cl::sycl::access::mode kMode,
          cl::sycl::access::target kTarget,
          cl::sycl::access::placeholder isPlaceholder>
class check_accessor_members {
 public:
  static void check(
      const cl::sycl::accessor<T, dims, kMode, kTarget, isPlaceholder> a,
      size_t size, size_t count, cl::sycl::id<dims> offset,
      cl::sycl::range<dims> range, std::string operation, util::logger &log) {
    std::string error_message =
        get_error_message(dims, kMode, kTarget, isPlaceholder, operation);

    bool placeholderValue =
        (isPlaceholder == cl::sycl::access::placeholder::true_t);

    if (a.get_size() != size) {
      FAIL(log, (error_message + " check(get_size)").c_str());
    }

    if (a.get_count() != count) {
      FAIL(log, (error_message + " check(get_count)").c_str());
    }

    if (a.get_offset() != offset) {
      FAIL(log, (error_message + " check(get_offset)").c_str());
    }

    if (a.get_range() != range) {
      FAIL(log, (error_message + " check(get_range)").c_str());
    }

    if (a.is_placeholder() != placeholderValue) {
      FAIL(log, (error_message + " check(is_placeholder)").c_str());
    }
  }
};

/** specialization of check_accessor_members for 0 dimensional buffer accessors
 */
template <typename T, cl::sycl::access::mode kMode,
          cl::sycl::access::target kTarget,
          cl::sycl::access::placeholder isPlaceholder>
class check_accessor_members<T, 0, kMode, kTarget, isPlaceholder> {
 public:
  static void check(
      const cl::sycl::accessor<T, 0, kMode, kTarget, isPlaceholder> a,
      size_t size, size_t count, std::string operation, util::logger &log) {
    std::string error_message =
        get_error_message(0, kMode, kTarget, isPlaceholder, operation);

    bool placeholderValue =
        (isPlaceholder == cl::sycl::access::placeholder::true_t);

    if (a.get_size() != size) {
      FAIL(log, (error_message + " check(get_size)").c_str());
    }

    if (a.get_count() != count) {
      FAIL(log, (error_message + " check(get_count)").c_str());
    }

    if (a.is_placeholder() != placeholderValue) {
      FAIL(log, (error_message + " check(is_placeholder)").c_str());
    }
  }
};

/** specialization of check_accessor_members for n dimensional local accessors
 */
template <typename T, size_t dims, cl::sycl::access::mode kMode>
class check_accessor_members<T, dims, kMode, cl::sycl::access::target::local,
                             cl::sycl::access::placeholder::false_t> {
 public:
  static void check(
      const cl::sycl::accessor<T, dims, kMode, cl::sycl::access::target::local,
                               cl::sycl::access::placeholder::false_t>
          a,
      size_t size, size_t count, std::string operation, util::logger &log) {
    std::string error_message =
        get_error_message(dims, kMode, cl::sycl::access::target::local,
                          cl::sycl::access::placeholder::false_t, operation);

    if (a.get_size() != size) {
      FAIL(log, (error_message + " check(get_size)").c_str());
    }

    if (a.get_count() != count) {
      FAIL(log, (error_message + " check(get_count)").c_str());
    }
  }
};

/** specialization of check_accessor_members for 0 dimensional local accessors
 */
template <typename T, cl::sycl::access::mode kMode>
class check_accessor_members<T, 0, kMode, cl::sycl::access::target::local,
                             cl::sycl::access::placeholder::false_t> {
 public:
  static void check(
      const cl::sycl::accessor<T, 0, kMode, cl::sycl::access::target::local,
                               cl::sycl::access::placeholder::false_t>
          a,
      std::string operation, util::logger &log) {
    std::string error_message =
        get_error_message(0, kMode, cl::sycl::access::target::local,
                          cl::sycl::access::placeholder::false_t, operation);

    if (a.get_size() != sizeof(T)) {
      FAIL(log, (error_message + " check(get_size)").c_str());
    }

    if (a.get_count() != 1) {
      FAIL(log, (error_message + " check(get_count)").c_str());
    }
  }
};

/** specialization of check_accessor_members for image accessors
 */
template <typename T, size_t dims, cl::sycl::access::mode kMode>
class check_accessor_members<T, dims, kMode, cl::sycl::access::target::image,
                             cl::sycl::access::placeholder::false_t> {
 public:
  static void check(
      const cl::sycl::accessor<T, dims, kMode, cl::sycl::access::target::image,
                               cl::sycl::access::placeholder::false_t>
          a,
      size_t size, size_t count, std::string operation, util::logger &log) {
    std::string error_message =
        get_error_message(dims, kMode, cl::sycl::access::target::image,
                          cl::sycl::access::placeholder::false_t, operation);

    if (a.get_size() < size) {
      FAIL(log, (error_message + " check(get_size)").c_str());
    }

    if (a.get_count() != count) {
      FAIL(log, (error_message + " check(get_count)").c_str());
    }
  }
};

/** specialization of check_accessor_members for host_images accessors
 */
template <typename T, size_t dims, cl::sycl::access::mode kMode>
class check_accessor_members<T, dims, kMode,
                             cl::sycl::access::target::host_image,
                             cl::sycl::access::placeholder::false_t> {
 public:
  static void check(const cl::sycl::accessor<
                        T, dims, kMode, cl::sycl::access::target::host_image,
                        cl::sycl::access::placeholder::false_t>
                        a,
                    size_t size, size_t count, std::string operation,
                    util::logger &log) {
    std::string error_message =
        get_error_message(dims, kMode, cl::sycl::access::target::host_image,
                          cl::sycl::access::placeholder::false_t, operation);

    if (a.get_size() < size) {
      FAIL(log, (error_message + " check(get_size)").c_str());
    }

    if (a.get_count() != count) {
      FAIL(log, (error_message + " check(get_count)").c_str());
    }
  }
};

/** specialization of check_accessor_members for image_array accessors
 */
template <typename T, size_t dims, cl::sycl::access::mode kMode>
class check_accessor_members<T, dims, kMode,
                             cl::sycl::access::target::image_array,
                             cl::sycl::access::placeholder::false_t> {
 public:
  static void check(const cl::sycl::accessor<
                        T, dims, kMode, cl::sycl::access::target::image_array,
                        cl::sycl::access::placeholder::false_t>
                        a,
                    size_t size, size_t count, std::string operation,
                    util::logger &log) {
    std::string error_message =
        get_error_message(dims, kMode, cl::sycl::access::target::image_array,
                          cl::sycl::access::placeholder::false_t, operation);

    if (a.get_size() < size) {
      FAIL(log, (error_message + " check(get_size)").c_str());
    }

    if (a.get_count() != count) {
      FAIL(log, (error_message + " check(get_count)").c_str());
    }
  }
};

}  // namespace accessor_utility__

#endif  // SYCL_1_2_1_TESTS_ACCESSOR_ACCESSOR_UTILITY_H
