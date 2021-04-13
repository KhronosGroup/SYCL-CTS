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
#ifndef SYCL_1_2_1_TESTS_ACCESSOR_ACCESSOR_CONSTRUCTORS_LOCAL_UTILITY_H
#define SYCL_1_2_1_TESTS_ACCESSOR_ACCESSOR_CONSTRUCTORS_LOCAL_UTILITY_H

#include "../common/common.h"
#include "accessor_constructors_utility.h"

namespace TEST_NAMESPACE {

using namespace sycl_cts;

/** Creates a local accessor and checks all its members for correctness.
 */
template <typename accTag>
class check_accessor_constructor_local {
  static constexpr size_t rangeDims = ((accTag::dims == 0) ? 1 : accTag::dims);
public:
  static void check(sycl_cts::util::logger &log,
                    const std::string& constructorName,
                    const std::string& typeName,
                    cl::sycl::handler& handler) {
    // construct the accessor
    typename accTag::type accessor(handler);

    // check the accessor
    check_accessor_members<accTag>::check(
        log, accessor, constructorName, typeName,
        accessor_members::size{sizeof(typename accTag::dataT)},
        accessor_members::count{1});
  }
  static void check(cl::sycl::range<rangeDims> range,
                    sycl_cts::util::logger &log,
                    const std::string& constructorName,
                    const std::string& typeName,
                    cl::sycl::handler& handler) {
    // construct the accessor
    typename accTag::type accessor(range, handler);

    // check the accessor
    check_accessor_members<accTag>::check(
        log, accessor, constructorName, typeName,
        accessor_members::size{range.size() * sizeof(typename accTag::dataT)},
        accessor_members::count{range.size()},
        accessor_members::range<rangeDims>{range});
  }
};

 /** check accessor is Copy Constructible
  */
template <typename accTag>
class check_local_accessor_copy_constructable {
public:
  static void check(const typename accTag::type& a,
                    sycl_cts::util::logger &log,
                    const std::string& typeName) {
    auto b{a};

    check_accessor_members<accTag>::check(
        log, b, "copy construction", typeName,
        accessor_members::size::from<accTag>(a),
        accessor_members::count::from<accTag>(a));

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

/** check accessor is Copy Assignable
 */
template <typename accTag>
class check_local_accessor_copy_assignable {
public:
  static void check(const typename accTag::type& a,
                    typename accTag::type& b,
                    sycl_cts::util::logger &log,
                    const std::string& typeName) {
    b = a;

    check_accessor_members<accTag>::check(
        log, b, "copy assignment", typeName,
        accessor_members::size::from<accTag>(a),
        accessor_members::count::from<accTag>(a));
  }
};

/** check accessor is Move Constructible
 */
template <typename accTag>
class check_local_accessor_move_constructable {
public:
  static void check(const typename accTag::type& a,
                    sycl_cts::util::logger &log,
                    const std::string& typeName) {
    auto b{std::move(a)};

    check_accessor_members<accTag>::check(
        log, b, "move construction", typeName,
        accessor_members::size::from<accTag>(a),
        accessor_members::count::from<accTag>(a));
  }
};

/** check accessor is Move Assignable
 */
template <typename accTag>
class check_local_accessor_move_assignable {
public:
  static void check(const typename accTag::type& a,
                    typename accTag::type& b,
                    sycl_cts::util::logger &log,
                    const std::string& typeName) {
    b = std::move(a);

    check_accessor_members<accTag>::check(
        log, b, "move assignment", typeName,
        accessor_members::size::from<accTag>(a),
        accessor_members::count::from<accTag>(a));
  }
};

/** Used to test the local accessor combinations for n > 0 dimensions
 */
template <typename T, size_t dims>
class local_accessor_dims {
public:
  static void check(util::logger &log, cl::sycl::queue &queue,
                    const std::string& typeName) {
    int size = 32;
    cl::sycl::range<dims> range = getRange<dims>(size);

    /** check buffer accessor constructors for local
     */
    {
      constexpr auto target = cl::sycl::access::target::local;

      queue.submit([&](cl::sycl::handler &h) {
        /** check constructor for reading local buffer
         */
        {
          constexpr auto mode = cl::sycl::access::mode::read;
          using accTag = accessor_type_info<T, dims, mode, target>;
          using verifier = check_accessor_constructor_local<accTag>;

          verifier::check(range,
                          log, "constructor(range, handler)",
                          typeName, h);
        }
        /** check constructor for atomic local buffer
         */
        {
          constexpr auto mode = cl::sycl::access::mode::atomic;
          using accTag = accessor_type_info<T, dims, mode, target>;
          using verifier = check_accessor_constructor_local<accTag>;

          verifier::check(range,
                          log, "constructor(range, handler)",
                          typeName, h);
        }

        /** check common-by-reference semantics
         */
        {
          constexpr auto mode = cl::sycl::access::mode::read_write;
          using accTag = accessor_type_info<T, dims, mode, target>;
          {
            using verifier = check_local_accessor_copy_constructable<accTag>;

            typename accTag::type srcAccessor(range, h);
            verifier::check(srcAccessor, log, typeName);
          }
          {
            using verifier = check_local_accessor_copy_assignable<accTag>;

            typename accTag::type srcAccessor(range, h);
            typename accTag::type dstAccessor(range, h);

            verifier::check(srcAccessor, dstAccessor, log, typeName);
          }
          {
            using verifier = check_local_accessor_move_constructable<accTag>;

            typename accTag::type srcAccessor(range, h);

            verifier::check(srcAccessor, log, typeName);
          }
          {
            using verifier = check_local_accessor_move_assignable<accTag>;

            typename accTag::type srcAccessor(range, h);
            typename accTag::type dstAccessor(range, h);

            verifier::check(srcAccessor, dstAccessor, log, typeName);
          }
        }

        /** dummy kernel as no kernel is required for these checks
         */
        h.single_task(dummy_functor<T, cl::sycl::access::target::local>{});
      });
      queue.wait_and_throw();
    }
  }
};

/** Used to test the 0 dimensional local accessor combinations
*/
template <typename T>
class local_accessor_dims<T, 0> {
 public:
  static void check(util::logger &log, cl::sycl::queue &queue,
                    const std::string& typeName) {
    /** check buffer accessor constructors for local
     */
    {
      constexpr auto target = cl::sycl::access::target::local;
      constexpr size_t dims = 0;

      queue.submit([&](cl::sycl::handler &h) {
        /** check constructor for reading local buffer
         */
        {
          constexpr auto mode = cl::sycl::access::mode::read;
          using accTag = accessor_type_info<T, dims, mode, target>;
          using verifier = check_accessor_constructor_local<accTag>;

          verifier::check(log, "constructor(handler)",
                          typeName, h);
        }
        /** check constructor for atomic local buffer
         */
        {
          constexpr auto mode = cl::sycl::access::mode::atomic;
          using accTag = accessor_type_info<T, dims, mode, target>;
          using verifier = check_accessor_constructor_local<accTag>;

          verifier::check(log, "constructor(handler)",
                          typeName, h);
        }

        /** check common-by-reference semantics
         */
        {
          constexpr auto mode = cl::sycl::access::mode::read_write;
          using accTag = accessor_type_info<T, dims, mode, target>;
          {
            using verifier = check_local_accessor_copy_constructable<accTag>;

            typename accTag::type srcAccessor(h);
            verifier::check(srcAccessor, log, typeName);
          }
          {
            using verifier = check_local_accessor_copy_assignable<accTag>;

            typename accTag::type srcAccessor(h);
            typename accTag::type dstAccessor(h);

            verifier::check(srcAccessor, dstAccessor, log, typeName);
          }
          {
            using verifier = check_local_accessor_move_constructable<accTag>;

            typename accTag::type srcAccessor(h);

            verifier::check(srcAccessor, log, typeName);
          }
          {
            using verifier = check_local_accessor_move_assignable<accTag>;

            typename accTag::type srcAccessor(h);
            typename accTag::type dstAccessor(h);

            verifier::check(srcAccessor, dstAccessor, log, typeName);
          }
        }

        /** dummy kernel as no kernel is required for these checks
         */
        h.single_task(dummy_functor<T, cl::sycl::access::target::local>{});
      });
      queue.wait_and_throw();
    }
  }
};

}  // namespace accessor_utility__

#endif  // SYCL_1_2_1_TESTS_ACCESSOR_ACCESSOR_CONSTRUCTORS_LOCAL_UTILITY_H
