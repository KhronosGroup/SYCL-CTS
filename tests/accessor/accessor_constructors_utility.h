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

#ifndef TEST_NAME
#error Invalid test namespace
#endif

namespace TEST_NAMESPACE {

/** unique dummy_functor per file
 *  this is a hack until the CMake script is fixed; kill both the alias and the
 *  dummy class once it is fixed
 */
template <typename kernelName, sycl::target kTarget>
struct dummy_accessor_ctors {};

template <typename kernelName, sycl::target kTarget>
using dummy_functor =
    ::dummy_functor<dummy_accessor_ctors<kernelName, kTarget>>;

using namespace sycl_cts;

/** @brief Helper functor to retrieve underlying data dimensionality
 *         for buffer and local accessors
 */
template <sycl::target target, size_t dims>
struct acc_data_dims {
  static constexpr size_t get() {
    return (dims == 0) ? 1 : dims;
  }
};
/** @brief Helper functor specialization to retrieve image dimensionality
 *         for image accessors
 */
template <size_t dims>
struct acc_data_dims<sycl::target::image, dims> {
  static constexpr size_t get() {
    return dims;
  }
};
/** @brief Helper functor specialization to retrieve image dimensionality
 *         for host_image accessors
 */
template <size_t dims>
struct acc_data_dims<sycl::target::host_image, dims> {
  static constexpr size_t get() {
    return dims;
  }
};
/** @brief Helper functor specialization to retrieve image dimensionality
 *         for image_array accessors
 */
template <size_t dims>
struct acc_data_dims<sycl::target::image_array, dims> {
  static constexpr size_t get() {
    return dims + 1;
  }
};

/** @brief Helper type trait for buffer accessors
 */
template <sycl::target target>
constexpr bool is_buffer_accessor() {
  return
      (target == sycl::target::global_buffer) ||
      (target == sycl::target::constant_buffer) ||
      (target == sycl::target::host_buffer);
}
/** @brief Helper type trait for local accessors
 */
template <sycl::target target>
constexpr bool is_local_accessor() {
  return
      (target == sycl::target::local);
}
/** @brief Helper type trait for image accessors
 */
template <sycl::target target>
constexpr bool is_image_accessor() {
  return
      (target == sycl::target::image) ||
      (target == sycl::target::host_image) ||
      (target == sycl::target::image_array);
}

/** @brief Helper tag to store accessor type information
 */
template <typename T, size_t kDims, sycl::access_mode kMode,
          sycl::target kTarget,
          sycl::access::placeholder kPlaceholder =
              sycl::access::placeholder::false_t>
struct accessor_type_info {
  using type = sycl::accessor<T, kDims, kMode, kTarget, kPlaceholder>;

  using dataT = T;
  static constexpr size_t dims = kDims;
  static constexpr sycl::access_mode mode = kMode;
  static constexpr sycl::target target = kTarget;
  static constexpr sycl::access::placeholder placeholder = kPlaceholder;

  static constexpr size_t dataDims = acc_data_dims<target, dims>::get();
};

/** @brief Syntax sugar use accessor type info tags for failure messages
 */
template <typename accTag>
inline void fail_for_accessor(sycl_cts::util::logger& log,
                              const std::string& dataType,
                              const std::string& message) {
  accessor_utility::fail_for_accessor<typename accTag::dataT, accTag::dims,
                                      accTag::mode, accTag::target,
                                      accTag::placeholder>(
      log, dataType, message);
}

/** @brief Namespace to store accessor member tags
 */
namespace accessor_members {
  /** @brief Helper struct to verify get_size() call for appropriate accessors
   */
  class size {
  public:
    using type = size_t;
    const type value;

    /** @brief Type trait to disable verification for image accessors
     */
    template <typename accTag>
    static inline bool constexpr enabled() {
      return is_buffer_accessor<accTag::target>() ||
             is_local_accessor<accTag::target>();
    }

    /** @brief Factory method to generate member tag in case accessor has
     *         get_size() method
     */
    template <typename accTag>
    typename std::enable_if<enabled<accTag>(), size>::type
        static from(const typename accTag::type& a) {
      return size{a.get_size()};
    }
    /** @brief Factory method to generate member tag in case accessor has no
     *         get_size() method
     */
    template <typename accTag>
    typename std::enable_if<!enabled<accTag>(), size>::type
        static from(const typename accTag::type& a) {
      return size{0U};
    }

    /** @brief Test for get_size() call
     */
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

  /** @brief Helper struct to verify size() call for all accessors
   */
  class count {
  public:
    using type = size_t;
    const type value;

    /** @brief Factory method to generate member tag
     */
    template <typename accTag>
    static count from(const typename accTag::type& a) {
      return count{a.size()};
    }

    /** @brief Test for size() call
     */
    template <typename accTag>
    void check(const typename accTag::type& a, sycl_cts::util::logger &log,
               const std::string& operation, const std::string& typeName) {
      if (a.size() != value) {
        fail_for_accessor<accTag>(log, typeName,
                                  operation + " check(size)");
      }
    }
  };

  /** @brief Helper struct to verify get_offset() call for appropriate accessors
   */
  template <size_t dims>
  class offset {
  public:
    using type = sycl::id<dims>;
    const type value;

    /** @brief Type trait to enable verification only if accessor has
     *         get_offset() method
     */
    template <typename accTag>
    static inline bool constexpr enabled() {
      return is_buffer_accessor<accTag::target>() && (accTag::dims != 0);
    }

    /** @brief Factory method to generate member tag in case accessor has
     *         get_offset() method
     */
    template <typename accTag>
    typename std::enable_if<enabled<accTag>(), offset<dims>>::type
        static from(const typename accTag::type& a) {
      return offset<dims>{a.get_offset()};
    }
    /** @brief Factory method to generate member tag in case accessor has no
     *         get_offset() method
     */
    template <typename accTag>
    typename std::enable_if<!enabled<accTag>(), offset>::type
        static from(const typename accTag::type& a) {
      const auto zero =
          sycl_cts::util::get_cts_object::id<dims>::get(0, 0, 0);
      return offset<dims>{zero};
    }

    /** @brief Test for get_offset() call
     */
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

  /** @brief Helper struct to verify get_range() call for appropriate accessors
   */
  template <size_t dims>
  class range {
  public:
    using type = sycl::range<dims>;
    const type value;

    /** @brief Type trait to enable verification only if accessor has
     *         get_range() method
     */
    template <typename accTag>
    static inline bool constexpr enabled() {
      return is_image_accessor<accTag::target>();
    }

    /** @brief Factory method to generate member tag in case accessor has
     *         get_range() method
     */
    template <typename accTag>
    typename std::enable_if<enabled<accTag>(), range<dims>>::type
        static from(const typename accTag::type& a) {
      return range<dims>{a.get_range()};
    }
    /** @brief Factory method to generate member tag in case accessor has no
     *         get_range() method
     */
    template <typename accTag>
    typename std::enable_if<!enabled<accTag>(), range<dims>>::type
        static from(const typename accTag::type& a) {
      auto zero = sycl_cts::util::get_cts_object::range<dims>::get(0, 0, 0);
      return range<dims>{zero};
    }

    /** @brief Test for get_range() call
     */
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

  /** @brief Helper struct to verify is_placeholder() call for buffer accessors
   */
  class placeholder {
  public:
    using type = sycl::access::placeholder;
    const type value;

    /** @brief Type trait to enable verification only if accessor has
     *         is_placeholder() method
     */
    template <typename accTag>
    static inline bool constexpr enabled() {
      return is_buffer_accessor<accTag::target>();
    }

    /** @brief Factory method to generate member tag in case accessor has
     *         is_placeholder() method
     */
    template <typename accTag>
    typename std::enable_if<enabled<accTag>(), placeholder>::type
        static from(const typename accTag::type& a) {
      const auto v = a.is_placeholder() ?
                     sycl::access::placeholder::true_t :
                     sycl::access::placeholder::false_t;
      return placeholder{v};
    }
    /** @brief Factory method to generate member tag in case accessor has no
     *         is_placeholder() method
     */
    template <typename accTag>
    typename std::enable_if<!enabled<accTag>(), placeholder>::type
        static from(const typename accTag::type& a) {
      return placeholder{};
    }

    /** @brief Test for is_placeholder() call
     */
    template <typename accTag>
    typename std::enable_if<enabled<accTag>(), void>::type
        check(const typename accTag::type& a, sycl_cts::util::logger &log,
              const std::string& operation, const std::string& typeName) {

      const bool expectedValue =
        (value == sycl::access::placeholder::true_t);

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

/** @brief Check all available accessor members to ensure no failure occured
 */
template <typename accTag>
class check_accessor_members {
  using acc_t = typename accTag::type;
public:
  /** @brief Run verification for each member tag passed
   *  @tparam membersT Deduced parameter pack type with tags
   *                   from accessor_members namespace
   */
  template <typename ... membersT>
  static void check(sycl_cts::util::logger &log, const acc_t& accessor,
                    const std::string& operation, const std::string& typeName,
                    membersT ... members) {
    int packExpansion[] = {(
      members.template check<accTag>(accessor, log, operation, typeName),
      0 // Dummy initialization value
    )...};
    static_cast<void>(packExpansion);
  }
};

/** @brief Check accessor is Copy Constructible
 */
template <typename accTag>
class check_accessor_copy_constructable {
public:
  static void check(const typename accTag::type& a,
                    sycl_cts::util::logger &log,
                    const std::string& typeName) {
    auto b{a};

    check_accessor_members<accTag>::check(
        log, b, "copy construction", typeName,
        accessor_members::size::from<accTag>(a),
        accessor_members::count::from<accTag>(a),
        accessor_members::offset<accTag::dataDims>::template from<accTag>(a),
        accessor_members::range<accTag::dataDims>::template from<accTag>(a),
        accessor_members::placeholder::from<accTag>(a));

    // check operator ==
    if (!(a == b)) {
      fail_for_accessor<accTag>(
          log, typeName,
          "accessor is not equality-comparable (operator==)");
    }
    if (!(b == a)) {
      fail_for_accessor<accTag>(
          log, typeName,
          "accessor is not equality-comparable (operator== symmetry failed)");
    }
    if (a != b) {
      fail_for_accessor<accTag>(
          log, typeName,
          "accessor is not equality-comparable (operator!=)");
    }
    if (b != a) {
      fail_for_accessor<accTag>(
          log, typeName,
          "accessor is not equality-comparable (operator!= symmetry failed)");
    }

    // check std::hash<accessor<>>
    std::hash<typename accTag::type> hasher;

    if (hasher(a) != hasher(b)) {
      fail_for_accessor<accTag>(
          log, typeName,
          "accessor hashing of equal failed");
    }
  }
};

/** @brief Check accessor is Copy Assignable
 */
template <typename accTag>
class check_accessor_copy_assignable {
public:
  static void check(const typename accTag::type& a,
                    typename accTag::type& b,
                    sycl_cts::util::logger &log,
                    const std::string& typeName) {
    b = a;

    check_accessor_members<accTag>::check(
        log, b, "copy assignment", typeName,
        accessor_members::size::from<accTag>(a),
        accessor_members::count::from<accTag>(a),
        accessor_members::offset<accTag::dataDims>::template from<accTag>(a),
        accessor_members::range<accTag::dataDims>::template from<accTag>(a),
        accessor_members::placeholder::from<accTag>(a));
  }
};

/** @brief Check accessor is Move Constructible
 */
template <typename accTag>
class check_accessor_move_constructable {
public:
  static void check(const typename accTag::type& a,
                    sycl_cts::util::logger &log,
                    const std::string& typeName) {
    auto b{std::move(a)};

    check_accessor_members<accTag>::check(
        log, b, "move construction", typeName,
        accessor_members::size::from<accTag>(a),
        accessor_members::count::from<accTag>(a),
        accessor_members::offset<accTag::dataDims>::template from<accTag>(a),
        accessor_members::range<accTag::dataDims>::template from<accTag>(a),
        accessor_members::placeholder::from<accTag>(a));
  }
};

/** @brief Check accessor is Move Assignable
 */
template <typename accTag>
class check_accessor_move_assignable {
public:
  static void check(const typename accTag::type& a,
                    typename accTag::type& b,
                    sycl_cts::util::logger &log,
                    const std::string& typeName) {
    b = std::move(a);

    check_accessor_members<accTag>::check(
        log, b, "move assignment", typeName,
        accessor_members::size::from<accTag>(a),
        accessor_members::count::from<accTag>(a),
        accessor_members::offset<accTag::dataDims>::template from<accTag>(a),
        accessor_members::range<accTag::dataDims>::template from<accTag>(a),
        accessor_members::placeholder::from<accTag>(a));
  }
};

}  // namespace accessor_utility__

#endif  // SYCL_1_2_1_TESTS_ACCESSOR_ACCESSOR_UTILITY_H
