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
template <typename T, size_t dims, cl::sycl::access::mode kMode>
class check_accessor_constructor_local {
 public:
  static void check(cl::sycl::range<dims> range, cl::sycl::handler &h,
                    util::logger &log) {
    // construct the accessor
    cl::sycl::accessor<T, dims, kMode, cl::sycl::access::target::local,
                       cl::sycl::access::placeholder::false_t>
        a(range, h);

    // check the accessor
    check_accessor_members<T, dims, kMode, cl::sycl::access::target::local,
                           cl::sycl::access::placeholder::false_t>::
        check(a, getElementsCount<dims>(range) * sizeof(T),
              getElementsCount<dims>(range), "constructor(handler)", log);
  }
};

/** Creates a 0 dimensional local accessor and checks all its members for
 * correctness.
 */
template <typename T, cl::sycl::access::mode kMode>
class check_accessor_constructor_local<T, 0, kMode> {
 public:
  static void check(cl::sycl::handler &h, util::logger &log) {
    // construct the accessor
    cl::sycl::accessor<T, 0, kMode, cl::sycl::access::target::local,
                       cl::sycl::access::placeholder::false_t>
        a(h);

    // check the accessor
    check_accessor_members<T, 0, kMode, cl::sycl::access::target::local,
                           cl::sycl::access::placeholder::false_t>::
        check(a, "constructor(buffer, handler)", log);
  }
};

/** Used to test the local accessor combinations
 */
template <typename T, size_t dims>
class local_accessor_dims {
 public:
  static void check(util::logger &log, cl::sycl::queue &queue) {
    int size = 32;

    /** check buffer accessor constructors for n > 0 dimensions
     */

    cl::sycl::range<dims> range = getRange<dims>(size);

    /** check buffer accessor constructors for local
     */
    {
      queue.submit([&](cl::sycl::handler &h) {
        /** check (handler, range) constructor for reading
         * local buffer
         */
        check_accessor_constructor_local<
            T, dims, cl::sycl::access::mode::read_write>::check(range, h, log);

        /** check (handler, range) constructor for atomic local buffer
         */
        check_accessor_constructor_local<
            T, dims, cl::sycl::access::mode::atomic>::check(range, h, log);

        /** check accessor is Copy Constructible
         */
        {
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::read_write,
                             cl::sycl::access::target::local,
                             cl::sycl::access::placeholder::false_t>
              a(range, h);
          auto b{a};

          check_accessor_members<T, dims, cl::sycl::access::mode::read_write,
                                 cl::sycl::access::target::local,
                                 cl::sycl::access::placeholder::false_t>::
              check(b, a.get_size(), a.get_count(), "copy construction", log);

          // check operator ==
          if (!(a == b)) {
            FAIL(log, "accessor is not equality-comparable (operator==)");
          }
          if (!(b == a)) {
            FAIL(log,
                 "accessor is not equality-comparable (operator== symmetry "
                 "failed)");
          }
          if (a != b) {
            FAIL(log, "accessor is not equality-comparable (operator!=)");
          }
          if (b != a) {
            FAIL(log,
                 "accessor is not equality-comparable (operator!= symmetry "
                 "failed)");
          }

          // check std::hash<accessor<>>
          std::hash<decltype(a)> hasher;

          if (hasher(a) != hasher(b)) {
            FAIL(log, "accessor hashing of equal failed");
          }
        }

        /** check accessor is Copy Assignable
         */
        {
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::read_write,
                             cl::sycl::access::target::local,
                             cl::sycl::access::placeholder::false_t>
              a(range, h);
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::read_write,
                             cl::sycl::access::target::local,
                             cl::sycl::access::placeholder::false_t>
              b(range, h);
          b = a;

          check_accessor_members<
              T, dims, cl::sycl::access::mode::read_write,
              cl::sycl::access::target::local,
              cl::sycl::access::placeholder::false_t>::check(b, a.get_size(),
                                                             a.get_count(),
                                                             "copy assignment",
                                                             log);
        }

        /** check accessor is Move Constructible
         */
        {
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::read_write,
                             cl::sycl::access::target::local,
                             cl::sycl::access::placeholder::false_t>
              a(range, h);
          auto b{std::move(a)};

          check_accessor_members<T, dims, cl::sycl::access::mode::read_write,
                                 cl::sycl::access::target::local,
                                 cl::sycl::access::placeholder::false_t>::
              check(b, getElementsCount<dims>(range) * sizeof(T),
                    getElementsCount<dims>(range), "move construction", log);
        }

        /** check accessor is Move Assignable
         */
        {
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::read_write,
                             cl::sycl::access::target::local,
                             cl::sycl::access::placeholder::false_t>
              a(range, h);
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::read_write,
                             cl::sycl::access::target::local,
                             cl::sycl::access::placeholder::false_t>
              b(range, h);
          b = std::move(a);

          check_accessor_members<T, dims, cl::sycl::access::mode::read_write,
                                 cl::sycl::access::target::local,
                                 cl::sycl::access::placeholder::false_t>::
              check(b, getElementsCount<dims>(range) * sizeof(T),
                    getElementsCount<dims>(range), "move assignment", log);
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
  static void check(util::logger &log, cl::sycl::queue &queue) {
    /** check buffer accessor constructors for local
     */
    {
      queue.submit([&](cl::sycl::handler &h) {
        /** check (handler, range) constructor for reading
         * local buffer
         */
        check_accessor_constructor_local<
            T, 0, cl::sycl::access::mode::read_write>::check(h, log);

        /** check (handler, range) constructor for atomic access of local buffer
         */
        check_accessor_constructor_local<
            T, 0, cl::sycl::access::mode::atomic>::check(h, log);

        /** check accessor is Copy Constructible
         */
        {
          cl::sycl::accessor<T, 0, cl::sycl::access::mode::read_write,
                             cl::sycl::access::target::local,
                             cl::sycl::access::placeholder::false_t>
              a(h);
          auto b{a};

          check_accessor_members<T, 0, cl::sycl::access::mode::read_write,
                                 cl::sycl::access::target::local,
                                 cl::sycl::access::placeholder::false_t>::
              check(b, "copy construction", log);

          // check operator ==
          if (!(a == b)) {
            FAIL(log, "accessor is not equality-comparable (operator==)");
          }
          if (!(b == a)) {
            FAIL(log,
                 "accessor is not equality-comparable (operator== symmetry "
                 "failed)");
          }
          if (a != b) {
            FAIL(log, "accessor is not equality-comparable (operator!=)");
          }
          if (b != a) {
            FAIL(log,
                 "accessor is not equality-comparable (operator!= symmetry "
                 "failed)");
          }

          // check std::hash<accessor<>>
          std::hash<decltype(a)> hasher;

          if (hasher(a) != hasher(b)) {
            FAIL(log, "accessor hashing of equal failed");
          }
        }

        /** check accessor is Copy Assignable
         */
        {
          cl::sycl::accessor<T, 0, cl::sycl::access::mode::read_write,
                             cl::sycl::access::target::local,
                             cl::sycl::access::placeholder::false_t>
              a(h);
          cl::sycl::accessor<T, 0, cl::sycl::access::mode::read_write,
                             cl::sycl::access::target::local,
                             cl::sycl::access::placeholder::false_t>
              b(h);
          b = a;

          check_accessor_members<
              T, 0, cl::sycl::access::mode::read_write,
              cl::sycl::access::target::local,
              cl::sycl::access::placeholder::false_t>::check(b,
                                                             "copy assignment",
                                                             log);
        }

        /** check accessor is Move Constructible
         */
        {
          cl::sycl::accessor<T, 0, cl::sycl::access::mode::read_write,
                             cl::sycl::access::target::local,
                             cl::sycl::access::placeholder::false_t>
              a(h);
          auto b{std::move(a)};

          check_accessor_members<T, 0, cl::sycl::access::mode::read_write,
                                 cl::sycl::access::target::local,
                                 cl::sycl::access::placeholder::false_t>::
              check(b, "move construction", log);
        }

        /** check accessor is Move Assignable
         */
        {
          cl::sycl::accessor<T, 0, cl::sycl::access::mode::read_write,
                             cl::sycl::access::target::local,
                             cl::sycl::access::placeholder::false_t>
              a(h);
          cl::sycl::accessor<T, 0, cl::sycl::access::mode::read_write,
                             cl::sycl::access::target::local,
                             cl::sycl::access::placeholder::false_t>
              b(h);
          b = std::move(a);

          check_accessor_members<
              T, 0, cl::sycl::access::mode::read_write,
              cl::sycl::access::target::local,
              cl::sycl::access::placeholder::false_t>::check(b,
                                                             "move assignment",
                                                             log);
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
