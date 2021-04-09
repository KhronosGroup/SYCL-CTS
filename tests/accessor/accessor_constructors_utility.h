/*************************************************************************
//
//  SYCL 2020 Conformance Test Suite
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
#include "accessor_utility_common.h"

#include <array>
#include <sstream>
#include <type_traits>

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
size_t getElementsCount(const cl::sycl::range<dims> &range) {
  return range.size();
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
std::string access_mode_to_string(cl::sycl::access::mode mode) {
  switch (mode) {
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
std::string access_target_to_string(cl::sycl::access::target target) {
  switch (target) {
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

template <cl::sycl::access::target target>
constexpr bool is_buffer_accessor() {
  return
      (target == cl::sycl::access::target::global_buffer) ||
      (target == cl::sycl::access::target::constant_buffer) ||
      (target == cl::sycl::access::target::host_buffer);
}
template <cl::sycl::access::target target>
constexpr bool is_local_accessor() {
  return
      (target == cl::sycl::access::target::local);
}
template <cl::sycl::access::target target>
constexpr bool is_image_accessor() {
  return
      (target == cl::sycl::access::target::image) ||
      (target == cl::sycl::access::target::host_image) ||
      (target == cl::sycl::access::target::image_array);
}

template <typename T, size_t kDims, cl::sycl::access::mode kMode,
          cl::sycl::access::target kTarget,
          cl::sycl::access::placeholder kPlaceholder =
              cl::sycl::access::placeholder::false_t>
struct accessor_type_info {
  using type = cl::sycl::accessor<T, kDims, kMode, kTarget, kPlaceholder>;

  using dataT = T;
  static constexpr size_t dims = kDims;
  static constexpr cl::sycl::access::mode mode = kMode;
  static constexpr cl::sycl::access::target target = kTarget;
  static constexpr cl::sycl::access::placeholder placeholder = kPlaceholder;
};

template <typename accTag>
void fail_for_accessor(sycl_cts::util::logger& log,
                       const std::string& dataType,
                       const std::string& message) {
  accessor_utility::fail_for_accessor<typename accTag::dataT, accTag::dims,
                                      accTag::mode, accTag::target,
                                      accTag::placeholder>(
    log, dataType, message);
}

namespace accessor_members {
  class size {
  public:
    using type = size_t;
    const type value;

    template <typename accTag>
    static inline bool constexpr enabled() {
      return is_buffer_accessor<accTag::target>() ||
             is_local_accessor<accTag::target>();
    }

    template <typename accTag>
    typename std::enable_if<enabled<accTag>(), size>::type
        static from(const typename accTag::type& a) {
      return size{a.get_size()};
    }
    template <typename accTag>
    typename std::enable_if<!enabled<accTag>(), size>::type
        static from(const typename accTag::type& a) {
      return size{0U};
    }

    template <typename accTag>
    typename std::enable_if<enabled<accTag>(), void>::type
        check(const typename accTag::type& a, sycl_cts::util::logger &log,
              const std::string& operation, const std::string& typeName) {

      if (a.get_size() != value) {
        fail_for_accessor<accTag>(log, typeName,
                                  operation + " check(get_size)");
      }
    }
    template <typename accTag>
    typename std::enable_if<!enabled<accTag>(), void>::type
        check(const typename accTag::type&, sycl_cts::util::logger &,
              const std::string&, const std::string&) {
      // do nothing: accessor has no get_size()
    }
  };

  class count {
  public:
    using type = size_t;
    const type value;

    template <typename accTag>
    static count from(const typename accTag::type& a) {
      return count{a.get_count()};
    }

    template <typename accTag>
    void check(const typename accTag::type& a, sycl_cts::util::logger &log,
               const std::string& operation, const std::string& typeName) {
      if (a.get_count() != value) {
        fail_for_accessor<accTag>(log, typeName,
                                  operation + " check(get_count)");
      }
    }
  };

  template <size_t dims>
  class offset {
  public:
    using type = cl::sycl::id<dims>;
    const type value;

    template <typename accTag>
    static inline bool constexpr enabled() {
      return is_buffer_accessor<accTag::target>() && (accTag::dims != 0);
    }

    template <typename accTag>
    typename std::enable_if<enabled<accTag>(), offset<dims>>::type
        static from(const typename accTag::type& a) {
      return offset<dims>{a.get_offset()};
    }
    template <typename accTag>
    typename std::enable_if<!enabled<accTag>(), offset>::type
        static from(const typename accTag::type& a) {
      auto zero = getId<dims>(0U);
      return offset<dims>{zero};
    }

    template <typename accTag>
    typename std::enable_if<enabled<accTag>(), void>::type
        check(const typename accTag::type& a, sycl_cts::util::logger &log,
              const std::string& operation, const std::string& typeName) {

      if (a.get_offset() != value) {
        fail_for_accessor<accTag>(log, typeName,
                                  operation + " check(get_offset)");
      }
    }
    template <typename accTag>
    typename std::enable_if<!enabled<accTag>(), void>::type
        check(const typename accTag::type&, sycl_cts::util::logger &,
              const std::string&, const std::string&) {
      // do nothing: accessor has no get_offset()
    }
  };

  template <size_t dims>
  class range {
  public:
    using type = cl::sycl::range<dims>;
    const type value;

    template <typename accTag>
    static inline bool constexpr enabled() {
      return is_image_accessor<accTag::target>();
    }

    template <typename accTag>
    typename std::enable_if<enabled<accTag>(), range<dims>>::type
        static from(const typename accTag::type& a) {
      return range<dims>{a.get_range()};
    }
    template <typename accTag>
    typename std::enable_if<!enabled<accTag>(), range<dims>>::type
        static from(const typename accTag::type& a) {
      auto zero = sycl_cts::util::get_cts_object::range<dims>::get(0, 0, 0);
      return range<dims>{zero};
    }

    template <typename accTag>
    typename std::enable_if<enabled<accTag>(), void>::type
        check(const typename accTag::type& a, sycl_cts::util::logger &log,
              const std::string& operation, const std::string& typeName) {

      if (a.get_range() != value) {
        fail_for_accessor<accTag>(log, typeName,
                                  operation + " check(get_range)");
      }
    }
    template <typename accTag>
    typename std::enable_if<!enabled<accTag>(), void>::type
        check(const typename accTag::type&, sycl_cts::util::logger &,
              const std::string&, const std::string&) {
      // do nothing: accessor has no get_range()
    }
  };

  class placeholder {
  public:
    using type = cl::sycl::access::placeholder;
    const type value;

    template <typename accTag>
    static inline bool constexpr enabled() {
      return is_buffer_accessor<accTag::target>();
    }

    template <typename accTag>
    typename std::enable_if<enabled<accTag>(), placeholder>::type
        static from(const typename accTag::type& a) {
      const auto v = a.is_placeholder() ?
                     cl::sycl::access::placeholder::true_t :
                     cl::sycl::access::placeholder::false_t;
      return placeholder{v};
    }
    template <typename accTag>
    typename std::enable_if<!enabled<accTag>(), placeholder>::type
        static from(const typename accTag::type& a) {
      return placeholder{};
    }

    template <typename accTag>
    typename std::enable_if<enabled<accTag>(), void>::type
        check(const typename accTag::type& a, sycl_cts::util::logger &log,
              const std::string& operation, const std::string& typeName) {

      const bool expectedValue =
        (value == cl::sycl::access::placeholder::true_t);

      if (a.is_placeholder() != expectedValue) {
        fail_for_accessor<accTag>(log, typeName,
                                  operation + " check(is_placeholder)");
      }
    }
    template <typename accTag>
    typename std::enable_if<!enabled<accTag>(), void>::type
        check(const typename accTag::type&, sycl_cts::util::logger &,
              const std::string&, const std::string&) {
      // do nothing: accessor has no is_placeholder()
    }
  };
};

/** checks all available accessor members to ensure no failure occured
 */
template <typename accTag>
class check_accessor_members {
  using acc_t = typename accTag::type;
public:
  template <typename ... membersT>
  static void check(sycl_cts::util::logger &log, const acc_t& accessor,
                    const std::string& operation, const std::string& typeName,
                    membersT ... members) {
    /** run verification for each member tag passed
    */
    int packExpansion[] = {(
      members.template check<accTag>(accessor, log, operation, typeName),
      0 // Dummy initialization value
    )...};
    static_cast<void>(packExpansion);
  }
};

}  // namespace accessor_utility__

#endif  // SYCL_1_2_1_TESTS_ACCESSOR_ACCESSOR_UTILITY_H
