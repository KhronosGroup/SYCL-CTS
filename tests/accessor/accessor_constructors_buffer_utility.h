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
#ifndef SYCL_1_2_1_TESTS_ACCESSOR_ACCESSOR_CONSTRUCTORS_BUFFER_UTILITY_H
#define SYCL_1_2_1_TESTS_ACCESSOR_ACCESSOR_CONSTRUCTORS_BUFFER_UTILITY_H

#include "../common/common.h"
#include "accessor_constructors_utility.h"

namespace TEST_NAMESPACE {

using namespace sycl_cts;

/** Creates a buffer accessor and checks all its members for correctness.
 */
template <typename T, size_t dims, cl::sycl::access::mode kMode,
          cl::sycl::access::target kTarget,
          cl::sycl::access::placeholder isPlaceholder>
class check_accessor_constructor_buffer {
 public:
  static void check(cl::sycl::buffer<T, dims> &buffer, cl::sycl::handler &h,
                    util::logger &log) {
    // construct the accessor
    cl::sycl::accessor<T, dims, kMode, kTarget, isPlaceholder> a(buffer, h);

    // check the accessor
    check_accessor_members<T, dims, kMode, kTarget, isPlaceholder>::check(
        a, buffer.get_size(), buffer.get_count(), getId<dims>(0),
        buffer.get_range(), "constructor(buffer, handler)", log);
  }
};

/** Creates a 0 dimensional buffer accessor and checks all its members for
 * correctness.
 */
template <typename T, cl::sycl::access::mode kMode,
          cl::sycl::access::target kTarget,
          cl::sycl::access::placeholder isPlaceholder>
class check_accessor_constructor_buffer<T, 0, kMode, kTarget, isPlaceholder> {
 public:
  static void check(cl::sycl::buffer<T, 1> &buffer, cl::sycl::handler &h,
                    util::logger &log) {
    // construct the accessor
    cl::sycl::accessor<T, 0, kMode, kTarget, isPlaceholder> a(buffer, h);

    // check the accessor
    check_accessor_members<T, 0, kMode, kTarget, isPlaceholder>::check(
        a, buffer.get_size(), buffer.get_count(),
        "constructor(buffer, handler)", log);
  }
};

/** Creates a buffer host accessor and checks all its members for correctness.
 */
template <typename T, size_t dims, cl::sycl::access::mode kMode,
          cl::sycl::access::placeholder isPlaceholder>
class check_accessor_constructor_buffer<
    T, dims, kMode, cl::sycl::access::target::host_buffer, isPlaceholder> {
 public:
  static void check(cl::sycl::buffer<T, dims> &buffer, util::logger &log) {
    // construct the accessor
    cl::sycl::accessor<T, dims, kMode, cl::sycl::access::target::host_buffer,
                       isPlaceholder>
        a(buffer);

    // check the accessor
    check_accessor_members<
        T, dims, kMode, cl::sycl::access::target::host_buffer,
        isPlaceholder>::check(a, buffer.get_size(), buffer.get_count(),
                              getId<dims>(0), buffer.get_range(),
                              "constructor(buffer, handler)", log);
  }
};

/** Creates a 0 dimensional buffer host accessor and checks all its members for
 * correctness.
 */
template <typename T, cl::sycl::access::mode kMode,
          cl::sycl::access::placeholder isPlaceholder>
class check_accessor_constructor_buffer<
    T, 0, kMode, cl::sycl::access::target::host_buffer, isPlaceholder> {
 public:
  static void check(cl::sycl::buffer<T, 1> &buffer, util::logger &log) {
    // construct the accessor
    cl::sycl::accessor<T, 0, kMode, cl::sycl::access::target::host_buffer,
                       isPlaceholder>
        a(buffer);

    // check the accessor
    check_accessor_members<T, 0, kMode, cl::sycl::access::target::host_buffer,
                           isPlaceholder>::check(a, buffer.get_size(),
                                                 buffer.get_count(),
                                                 "constructor(buffer, handler)",
                                                 log);
  }
};

/** Creates a placeholder buffer accessor and checks all its members
 *  for correctness.
 */
template <typename T, size_t dims, cl::sycl::access::mode kMode,
          cl::sycl::access::target kTarget>
class check_accessor_constructor_buffer<
    T, dims, kMode, kTarget, cl::sycl::access::placeholder::true_t> {

 public:
  static void check(cl::sycl::buffer<T, dims> &buffer, util::logger &log) {
    // construct the accessor
    cl::sycl::accessor<
        T, dims, kMode, kTarget, cl::sycl::access::placeholder::true_t>
            a(buffer);

    // check the accessor
    check_accessor_members<
        T, dims, kMode, kTarget, cl::sycl::access::placeholder::true_t>::check(
            a, buffer.get_size(), buffer.get_count(), getId<dims>(0),
            buffer.get_range(), "constructor(buffer)", log);
  }
};

/** Creates a 0 dimensional placeholder buffer accessor and checks
 *  all its members for correctness.
 */
template <typename T, cl::sycl::access::mode kMode,
          cl::sycl::access::target kTarget>
class check_accessor_constructor_buffer<
    T, 0, kMode, kTarget, cl::sycl::access::placeholder::true_t> {
 public:
  static void check(cl::sycl::buffer<T, 1> &buffer, util::logger &log) {
    // construct the accessor
    cl::sycl::accessor<
        T, 0, kMode, kTarget, cl::sycl::access::placeholder::true_t>
            a(buffer);

    // check the accessor
    check_accessor_members<
        T, 0, kMode, kTarget, cl::sycl::access::placeholder::true_t>::check(
            a, buffer.get_size(), buffer.get_count(),
            "constructor(buffer)", log);
  }
};

/** Creates a ranged buffer accessor and checks all its members for correctness.
 */
template <typename T, size_t dims, cl::sycl::access::mode kMode,
          cl::sycl::access::target kTarget,
          cl::sycl::access::placeholder isPlaceholder>
class check_ranged_accessor_constructor_buffer {
 public:
  static void check(cl::sycl::buffer<T, dims> &buffer, cl::sycl::handler &h,
                    cl::sycl::range<dims> range, cl::sycl::id<dims> offset,
                    util::logger &log) {
    // construct the accessor
    cl::sycl::accessor<T, dims, kMode, kTarget, isPlaceholder> a(buffer, h,
                                                                 range, offset);

    // check the accessor
    check_accessor_members<T, dims, kMode, kTarget, isPlaceholder>::check(
        a, range.size() * sizeof(T), range.size(), offset, range,
        "constructor(buffer, handler, range, offset)", log);
  }
};

/** Creates a ranged buffer host accessor and checks all its members for
 * correctness.
 */
template <typename T, size_t dims, cl::sycl::access::mode kMode,
          cl::sycl::access::placeholder isPlaceholder>
class check_ranged_accessor_constructor_buffer<
    T, dims, kMode, cl::sycl::access::target::host_buffer, isPlaceholder> {
 public:
  static void check(cl::sycl::buffer<T, dims> &buffer,
                    cl::sycl::range<dims> range, cl::sycl::id<dims> offset,
                    util::logger &log) {
    // construct the accessor
    cl::sycl::accessor<T, dims, kMode, cl::sycl::access::target::host_buffer,
                       isPlaceholder>
        a(buffer, range, offset);

    // check the accessor
    check_accessor_members<
        T, dims, kMode, cl::sycl::access::target::host_buffer,
        isPlaceholder>::check(a, range.size() * sizeof(T), range.size(), offset,
                              range,
                              "constructor(buffer, handler, range, offset)",
                              log);
  }
};

/** Creates a ranged placeholder buffer accessor and checks all
 *  its members for correctness.
 */
template <typename T, size_t dims, cl::sycl::access::mode kMode,
          cl::sycl::access::target kTarget>
class check_ranged_accessor_constructor_buffer<
    T, dims, kMode, kTarget, cl::sycl::access::placeholder::true_t> {
 public:
  static void check(cl::sycl::buffer<T, dims> &buffer,
                    cl::sycl::range<dims> range, cl::sycl::id<dims> offset,
                    util::logger &log) {
    // construct the accessor
    cl::sycl::accessor<
        T, dims, kMode, kTarget, cl::sycl::access::placeholder::true_t>
            a(buffer, range, offset);

    // check the accessor
    check_accessor_members<
        T, dims, kMode, kTarget, cl::sycl::access::placeholder::true_t>::check(
            a, range.size() * sizeof(T), range.size(), offset, range,
            "constructor(buffer, handler, range, offset)", log);
  }
};

/** enum used to denote that the buffer_accessor_dims specialization performs
 * checks either only for host_buffer or for not for host_buffer
 */
enum is_host_buffer : bool { false_t = false, true_t = true };

/** Used to test the buffer accessor combinations for global_buffer and
 * constant_buffer
 */
template <typename T, size_t dims, is_host_buffer isHostBuffer = false_t,
          cl::sycl::access::placeholder isPlaceholder =
              cl::sycl::access::placeholder::false_t>
class buffer_accessor_dims {
 public:
  static void check(util::logger &log, cl::sycl::queue &queue) {
    int size = 32;

    /** check buffer accessor constructors for n > 0 dimensions
     */

    cl::sycl::range<dims> range = getRange<dims>(size);
    std::vector<uint8_t> data(getElementsCount<dims>(range) * sizeof(T));
    std::iota(std::begin(data), std::end(data), 0);
    cl::sycl::buffer<T, dims> buffer(reinterpret_cast<T *>(data.data()), range);
    cl::sycl::id<dims> offset(range / 2);
    const auto r = range / 2;

    /** check buffer accessor constructors for global_buffer
     */
    {
      queue.submit([&](cl::sycl::handler &h) {
        /** check (buffer, handler) constructor for reading
         * global_buffer
         */
        check_accessor_constructor_buffer<
            T, dims, cl::sycl::access::mode::read,
            cl::sycl::access::target::global_buffer,
            isPlaceholder>::check(buffer, h, log);

        /** check (buffer, handler, range, offset) constructor for reading
         * global_buffer
         */
        check_ranged_accessor_constructor_buffer<
            T, dims, cl::sycl::access::mode::read,
            cl::sycl::access::target::global_buffer,
            isPlaceholder>::check(buffer, h, r, offset, log);

        /** check (buffer, handler) constructor for writing global_buffer
         */
        check_accessor_constructor_buffer<
            T, dims, cl::sycl::access::mode::write,
            cl::sycl::access::target::global_buffer,
            isPlaceholder>::check(buffer, h, log);

        /** check (buffer, handler, range, offset) constructor for writing
                 * global_buffer
                 */
        check_ranged_accessor_constructor_buffer<
            T, dims, cl::sycl::access::mode::write,
            cl::sycl::access::target::global_buffer,
            isPlaceholder>::check(buffer, h, r, offset, log);

        /** check (buffer, handler) constructor for read_write global_buffer
        */
        check_accessor_constructor_buffer<
            T, dims, cl::sycl::access::mode::read_write,
            cl::sycl::access::target::global_buffer,
            isPlaceholder>::check(buffer, h, log);

        /** check (buffer, handler, range, offset) constructor for read_write
         * global_buffer
                 */
        check_ranged_accessor_constructor_buffer<
            T, dims, cl::sycl::access::mode::read_write,
            cl::sycl::access::target::global_buffer,
            isPlaceholder>::check(buffer, h, r, offset, log);

        /** check (buffer, handler) constructor for discard_write global_buffer
        */
        check_accessor_constructor_buffer<
            T, dims, cl::sycl::access::mode::discard_write,
            cl::sycl::access::target::global_buffer,
            isPlaceholder>::check(buffer, h, log);

        /** check (buffer, handler, range, offset) constructor for discard_write
         * global_buffer
         */
        check_ranged_accessor_constructor_buffer<
            T, dims, cl::sycl::access::mode::discard_write,
            cl::sycl::access::target::global_buffer,
            isPlaceholder>::check(buffer, h, r, offset, log);

        /** check (buffer, handler) constructor for discard_read_write
        * global_buffer
        */
        check_accessor_constructor_buffer<
            T, dims, cl::sycl::access::mode::discard_read_write,
            cl::sycl::access::target::global_buffer,
            isPlaceholder>::check(buffer, h, log);

        /** check (buffer, handler, range, offset) constructor for
         * discard_read_write global_buffer
         */
        check_ranged_accessor_constructor_buffer<
            T, dims, cl::sycl::access::mode::discard_read_write,
            cl::sycl::access::target::global_buffer,
            isPlaceholder>::check(buffer, h, r, offset, log);

        /** check (buffer, handler) global_buffer accessor constructors for
        *  atomic
        */
        check_accessor_constructor_buffer<
            T, dims, cl::sycl::access::mode::atomic,
            cl::sycl::access::target::global_buffer,
            isPlaceholder>::check(buffer, h, log);

        /** check (buffer, handler, range, offset) global_buffer constructor for
         *  atomic
         */
        check_ranged_accessor_constructor_buffer<
            T, dims, cl::sycl::access::mode::atomic,
            cl::sycl::access::target::global_buffer,
            isPlaceholder>::check(buffer, h, r, offset, log);

        /** check accessor is Copy Constructible
         */
        {
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                             cl::sycl::access::target::global_buffer,
                             isPlaceholder>
              a(buffer, h, r, offset);
          auto b{a};

          check_accessor_members<T, dims, cl::sycl::access::mode::read,
                                 cl::sycl::access::target::global_buffer,
                                 isPlaceholder>::check(b, a.get_size(),
                                                       a.get_count(),
                                                       a.get_offset(),
                                                       a.get_range(),
                                                       "copy construction",
                                                       log);

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
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                             cl::sycl::access::target::global_buffer,
                             isPlaceholder>
              a(buffer, h, r, offset);
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                             cl::sycl::access::target::global_buffer,
                             isPlaceholder>
              b(buffer, h);
          b = a;

          check_accessor_members<T, dims, cl::sycl::access::mode::read,
                                 cl::sycl::access::target::global_buffer,
                                 isPlaceholder>::check(b, a.get_size(),
                                                       a.get_count(),
                                                       a.get_offset(),
                                                       a.get_range(),
                                                       "copy assignment", log);
        }

        /** check accessor is Move Constructible
         */
        {
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                             cl::sycl::access::target::global_buffer,
                             isPlaceholder>
              a(buffer, h);
          auto b{std::move(a)};

          check_accessor_members<T, dims, cl::sycl::access::mode::read,
                                 cl::sycl::access::target::global_buffer,
                                 isPlaceholder>::check(b, buffer.get_size(),
                                                       buffer.get_count(),
                                                       getId<dims>(0),
                                                       buffer.get_range(),
                                                       "move construction",
                                                       log);
        }

        /** check accessor is Move Assignable
         */
        {
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                             cl::sycl::access::target::global_buffer,
                             isPlaceholder>
              a(buffer, h);
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                             cl::sycl::access::target::global_buffer,
                             isPlaceholder>
              b(buffer, h, r, offset);
          b = std::move(a);

          check_accessor_members<T, dims, cl::sycl::access::mode::read,
                                 cl::sycl::access::target::global_buffer,
                                 isPlaceholder>::check(b, buffer.get_size(),
                                                       buffer.get_count(),
                                                       getId<dims>(0),
                                                       buffer.get_range(),
                                                       "move assignment", log);
        }

        /** dummy kernel as no kernel is required for these checks
         */
        h.single_task(
            dummy_functor<T, cl::sycl::access::target::global_buffer>{});
      });
      queue.wait_and_throw();
    }

    /** check buffer accessor constructors for constant_buffer
     */
    {
      queue.submit([&](cl::sycl::handler &h) {
        /** check (buffer, handler) constructor for reading constant_buffer
         */
        check_accessor_constructor_buffer<
            T, dims, cl::sycl::access::mode::read,
            cl::sycl::access::target::constant_buffer,
            isPlaceholder>::check(buffer, h, log);

        /** check (buffer, handler, range, offset) constructor for reading
         * constant_buffer
         */
        check_ranged_accessor_constructor_buffer<
            T, dims, cl::sycl::access::mode::read,
            cl::sycl::access::target::constant_buffer,
            isPlaceholder>::check(buffer, h, r, offset, log);

        /** check accessor is Copy Constructible
         */
        {
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                             cl::sycl::access::target::constant_buffer,
                             isPlaceholder>
              a(buffer, h);
          auto b{a};

          check_accessor_members<T, dims, cl::sycl::access::mode::read,
                                 cl::sycl::access::target::constant_buffer,
                                 isPlaceholder>::check(b, a.get_size(),
                                                       a.get_count(),
                                                       a.get_offset(),
                                                       a.get_range(),
                                                       "copy construction",
                                                       log);

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
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                             cl::sycl::access::target::constant_buffer,
                             isPlaceholder>
              a(buffer, h);
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                             cl::sycl::access::target::constant_buffer,
                             isPlaceholder>
              b(buffer, h, r, offset);
          b = a;

          check_accessor_members<T, dims, cl::sycl::access::mode::read,
                                 cl::sycl::access::target::constant_buffer,
                                 isPlaceholder>::check(b, a.get_size(),
                                                       a.get_count(),
                                                       a.get_offset(),
                                                       a.get_range(),
                                                       "copy assignment", log);
        }

        /** check accessor is Move Constructible
         */
        {
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                             cl::sycl::access::target::constant_buffer,
                             isPlaceholder>
              a(buffer, h);
          auto b{std::move(a)};

          check_accessor_members<T, dims, cl::sycl::access::mode::read,
                                 cl::sycl::access::target::constant_buffer,
                                 isPlaceholder>::check(b, buffer.get_size(),
                                                       buffer.get_count(),
                                                       getId<dims>(0),
                                                       buffer.get_range(),
                                                       "move construction",
                                                       log);
        }

        /** check accessor is Move Assignable
         */
        {
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                             cl::sycl::access::target::constant_buffer,
                             isPlaceholder>
              a(buffer, h);
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                             cl::sycl::access::target::constant_buffer,
                             isPlaceholder>
              b(buffer, h, r, offset);
          b = std::move(a);

          check_accessor_members<T, dims, cl::sycl::access::mode::read,
                                 cl::sycl::access::target::constant_buffer,
                                 isPlaceholder>::check(b, buffer.get_size(),
                                                       buffer.get_count(),
                                                       getId<dims>(0),
                                                       buffer.get_range(),
                                                       "move assignment", log);
        }

        /** dummy kernel as no kernel is required for these checks
         */
        h.single_task(
            dummy_functor<T, cl::sycl::access::target::constant_buffer>{});

      });
      queue.wait_and_throw();
    }
  }
};

/** Specialization of buffer_accessor_dims for host_buffer
 */
template <typename T, size_t dims, cl::sycl::access::placeholder isPlaceholder>
class buffer_accessor_dims<T, dims, is_host_buffer::true_t, isPlaceholder> {
 public:
  static void check(util::logger &log, cl::sycl::queue &queue) {
    int size = 32;

    /** check buffer accessor constructors for n > 0 dimensions
     */
    cl::sycl::range<dims> range = getRange<dims>(size);
    std::vector<uint8_t> data(getElementsCount<dims>(range) * sizeof(T));
    std::iota(std::begin(data), std::end(data), 0);
    cl::sycl::buffer<T, dims> buffer(reinterpret_cast<T *>(data.data()), range);
    cl::sycl::id<dims> offset(range / 2);
    const auto r = range / 2;

    /** check buffer accessor constructors for host_buffer
     */
    {
      /** check (buffer) constructor for reading host_buffer
       */
      check_accessor_constructor_buffer<T, dims, cl::sycl::access::mode::read,
                                        cl::sycl::access::target::host_buffer,
                                        isPlaceholder>::check(buffer, log);

      /** check (buffer, range, offset) constructor for reading
       * host_buffer
       */
      check_ranged_accessor_constructor_buffer<
          T, dims, cl::sycl::access::mode::read,
          cl::sycl::access::target::host_buffer, isPlaceholder>::check(buffer,
                                                                       r,
                                                                       offset,
                                                                       log);

      /** check (buffer) constructor for writing host_buffer
       */
      check_accessor_constructor_buffer<T, dims, cl::sycl::access::mode::write,
                                        cl::sycl::access::target::host_buffer,
                                        isPlaceholder>::check(buffer, log);

      /** check (buffer, range, offset) constructor for writing
       * host_buffer
       */
      check_ranged_accessor_constructor_buffer<
          T, dims, cl::sycl::access::mode::write,
          cl::sycl::access::target::host_buffer, isPlaceholder>::check(buffer,
                                                                       r,
                                                                       offset,
                                                                       log);

      /** check (buffer) constructor for read_write host_buffer
       */
      check_accessor_constructor_buffer<
          T, dims, cl::sycl::access::mode::read_write,
          cl::sycl::access::target::host_buffer, isPlaceholder>::check(buffer,
                                                                       log);

      /** check (buffer, range, offset) constructor for read_write
       * host_buffer
       */
      check_ranged_accessor_constructor_buffer<
          T, dims, cl::sycl::access::mode::read_write,
          cl::sycl::access::target::host_buffer, isPlaceholder>::check(buffer,
                                                                       r,
                                                                       offset,
                                                                       log);

      /** check (buffer) constructor for discard_write host_buffer
      */
      check_accessor_constructor_buffer<
          T, dims, cl::sycl::access::mode::discard_write,
          cl::sycl::access::target::host_buffer, isPlaceholder>::check(buffer,
                                                                       log);

      /** check (buffer, range, offset) constructor for discard_write
       * host_buffer
       */
      check_ranged_accessor_constructor_buffer<
          T, dims, cl::sycl::access::mode::discard_write,
          cl::sycl::access::target::host_buffer, isPlaceholder>::check(buffer,
                                                                       r,
                                                                       offset,
                                                                       log);

      /** check (buffer) constructor for discard_read_write host_buffer
       */
      check_accessor_constructor_buffer<
          T, dims, cl::sycl::access::mode::discard_read_write,
          cl::sycl::access::target::host_buffer, isPlaceholder>::check(buffer,
                                                                       log);

      /** check (buffer, range, offset) constructor for
       * discard_read_write host_buffer
       */
      check_ranged_accessor_constructor_buffer<
          T, dims, cl::sycl::access::mode::discard_read_write,
          cl::sycl::access::target::host_buffer, isPlaceholder>::check(buffer,
                                                                       r,
                                                                       offset,
                                                                       log);

      /** check accessor is Copy Constructible
       */
      {
        cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                           cl::sycl::access::target::host_buffer, isPlaceholder>
            a(buffer);
        auto b{a};

        check_accessor_members<T, dims, cl::sycl::access::mode::read,
                               cl::sycl::access::target::host_buffer,
                               isPlaceholder>::check(b, a.get_size(),
                                                     a.get_count(),
                                                     a.get_offset(),
                                                     a.get_range(),
                                                     "copy construction", log);

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
        cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                           cl::sycl::access::target::host_buffer, isPlaceholder>
            a(buffer, r, offset);
        cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                           cl::sycl::access::target::host_buffer, isPlaceholder>
            b(buffer);
        b = a;

        check_accessor_members<T, dims, cl::sycl::access::mode::read,
                               cl::sycl::access::target::host_buffer,
                               isPlaceholder>::check(b, a.get_size(),
                                                     a.get_count(),
                                                     a.get_offset(),
                                                     a.get_range(),
                                                     "copy assignment", log);
      }

      /** check accessor is Move Constructible
       */
      {
        cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                           cl::sycl::access::target::host_buffer, isPlaceholder>
            a(buffer);
        auto b{std::move(a)};

        check_accessor_members<T, dims, cl::sycl::access::mode::read,
                               cl::sycl::access::target::host_buffer,
                               isPlaceholder>::check(b, buffer.get_size(),
                                                     buffer.get_count(),
                                                     getId<dims>(0),
                                                     buffer.get_range(),
                                                     "move construction", log);
      }

      /** check accessor is Move Assignable
       */
      {
        cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                           cl::sycl::access::target::host_buffer, isPlaceholder>
            a(buffer);
        cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                           cl::sycl::access::target::host_buffer, isPlaceholder>
            b(buffer, r, offset);
        b = std::move(a);

        check_accessor_members<T, dims, cl::sycl::access::mode::read,
                               cl::sycl::access::target::host_buffer,
                               isPlaceholder>::check(b, buffer.get_size(),
                                                     buffer.get_count(),
                                                     getId<dims>(0),
                                                     buffer.get_range(),
                                                     "move assignment", log);
      }
    }
  }
};

/** Used to test the buffer accessor combinations for placeholder
 *  global_buffer and placeholder constant_buffer
 */
template <typename T, size_t dims>
class buffer_accessor_dims<T, dims, is_host_buffer::false_t,
                           cl::sycl::access::placeholder::true_t> {
 public:
  static void check(util::logger &log, cl::sycl::queue &queue) {
    int size = 32;

    /** check buffer accessor constructors for n > 0 dimensions
     */

    cl::sycl::range<dims> range = getRange<dims>(size);
    std::vector<uint8_t> data(getElementsCount<dims>(range) * sizeof(T));
    std::iota(std::begin(data), std::end(data), 0);
    cl::sycl::buffer<T, dims> buffer(reinterpret_cast<T *>(data.data()), range);
    cl::sycl::id<dims> offset(range / 2);
    const auto r = range / 2;

    /** check buffer accessor constructors for global_buffer
     */
    {
      /** check (buffer) constructor for reading global_buffer
       */
      check_accessor_constructor_buffer<
          T, dims, cl::sycl::access::mode::read,
          cl::sycl::access::target::global_buffer,
          cl::sycl::access::placeholder::true_t>::check(buffer, log);

      /** check (buffer, range, offset) constructor for reading global_buffer
       */
      check_ranged_accessor_constructor_buffer<
          T, dims, cl::sycl::access::mode::read,
          cl::sycl::access::target::global_buffer,
          cl::sycl::access::placeholder::true_t>::check(
              buffer, r, offset, log);

      /** check (buffer) constructor for writing global_buffer
       */
      check_accessor_constructor_buffer<
          T, dims, cl::sycl::access::mode::write,
          cl::sycl::access::target::global_buffer,
          cl::sycl::access::placeholder::true_t>::check(buffer, log);

      /** check (buffer, range, offset) constructor for writing global_buffer
       */
      check_ranged_accessor_constructor_buffer<
          T, dims, cl::sycl::access::mode::write,
          cl::sycl::access::target::global_buffer,
          cl::sycl::access::placeholder::true_t>::check(
              buffer, r, offset, log);

      /** check (buffer) constructor for read_write global_buffer
       */
      check_accessor_constructor_buffer<
          T, dims, cl::sycl::access::mode::read_write,
          cl::sycl::access::target::global_buffer,
          cl::sycl::access::placeholder::true_t>::check(buffer, log);

      /** check (buffer, handler, range, offset) constructor for read_write
       * global_buffer
       */
      check_ranged_accessor_constructor_buffer<
          T, dims, cl::sycl::access::mode::read_write,
          cl::sycl::access::target::global_buffer,
          cl::sycl::access::placeholder::true_t>::check(
              buffer, r, offset, log);

      /** check (buffer) constructor for discard_write global_buffer
       */
      check_accessor_constructor_buffer<
          T, dims, cl::sycl::access::mode::discard_write,
          cl::sycl::access::target::global_buffer,
          cl::sycl::access::placeholder::true_t>::check(buffer, log);

      /** check (buffer, range, offset) constructor for discard_write
       *  global_buffer
       */
      check_ranged_accessor_constructor_buffer<
          T, dims, cl::sycl::access::mode::discard_write,
          cl::sycl::access::target::global_buffer,
          cl::sycl::access::placeholder::true_t>::check(
              buffer, r, offset, log);

      /** check (buffer) constructor for discard_read_write global_buffer
       */
      check_accessor_constructor_buffer<
          T, dims, cl::sycl::access::mode::discard_read_write,
          cl::sycl::access::target::global_buffer,
          cl::sycl::access::placeholder::true_t>::check(buffer, log);

      /** check (buffer, range, offset) constructor for
       *  discard_read_write global_buffer
       */
      check_ranged_accessor_constructor_buffer<
          T, dims, cl::sycl::access::mode::discard_read_write,
          cl::sycl::access::target::global_buffer,
          cl::sycl::access::placeholder::true_t>::check(
              buffer, r, offset, log);

      /** check (buffer) global_buffer accessor constructors for atomic
       */
      check_accessor_constructor_buffer<
          T, dims, cl::sycl::access::mode::atomic,
          cl::sycl::access::target::global_buffer,
          cl::sycl::access::placeholder::true_t>::check(buffer, log);

      /** check (buffer, range, offset) global_buffer constructor for atomic
       */
      check_ranged_accessor_constructor_buffer<
          T, dims, cl::sycl::access::mode::atomic,
          cl::sycl::access::target::global_buffer,
          cl::sycl::access::placeholder::true_t>::check(buffer, r, offset, log);

      /** check accessor is Copy Constructible
       */
      {
        cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                           cl::sycl::access::target::global_buffer,
                           cl::sycl::access::placeholder::true_t>
            a(buffer, r, offset);
        auto b{a};

        check_accessor_members<
            T, dims, cl::sycl::access::mode::read,
            cl::sycl::access::target::global_buffer,
            cl::sycl::access::placeholder::true_t>::check(
                b, a.get_size(), a.get_count(),
                a.get_offset(), a.get_range(), "copy construction", log);

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
        cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                           cl::sycl::access::target::global_buffer,
                           cl::sycl::access::placeholder::true_t>
            a(buffer, r, offset);
        cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                           cl::sycl::access::target::global_buffer,
                           cl::sycl::access::placeholder::true_t>
            b(buffer);
        b = a;

        check_accessor_members<
            T, dims, cl::sycl::access::mode::read,
            cl::sycl::access::target::global_buffer,
            cl::sycl::access::placeholder::true_t>::check(
                b, a.get_size(), a.get_count(),
                a.get_offset(), a.get_range(), "copy assignment", log);
      }

      /** check accessor is Move Constructible
       */
      {
        cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                           cl::sycl::access::target::global_buffer,
                           cl::sycl::access::placeholder::true_t>
            a(buffer);
        auto b{std::move(a)};

        check_accessor_members<
            T, dims, cl::sycl::access::mode::read,
            cl::sycl::access::target::global_buffer,
            cl::sycl::access::placeholder::true_t>::check(
                b, buffer.get_size(), buffer.get_count(), getId<dims>(0),
                buffer.get_range(), "move construction", log);
      }

      /** check accessor is Move Assignable
       */
      {
        cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                           cl::sycl::access::target::global_buffer,
                           cl::sycl::access::placeholder::true_t>
            a(buffer);
        cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                           cl::sycl::access::target::global_buffer,
                           cl::sycl::access::placeholder::true_t>
            b(buffer, r, offset);
        b = std::move(a);

        check_accessor_members<
            T, dims, cl::sycl::access::mode::read,
            cl::sycl::access::target::global_buffer,
            cl::sycl::access::placeholder::true_t>::check(
                b, buffer.get_size(), buffer.get_count(),
                getId<dims>(0), buffer.get_range(), "move assignment", log);
      }

      queue.submit([&](cl::sycl::handler &h) {
        /** dummy kernel as no kernel is required for these checks
         */
        h.single_task(
            dummy_functor<T, cl::sycl::access::target::global_buffer>{});
      });
      queue.wait_and_throw();
    }

    /** check buffer accessor constructors for constant_buffer
     */
    {
      /** check (buffer) constructor for reading constant_buffer
       */
      check_accessor_constructor_buffer<
          T, dims, cl::sycl::access::mode::read,
          cl::sycl::access::target::constant_buffer,
          cl::sycl::access::placeholder::true_t>::check(buffer, log);

      /** check (buffer, range, offset) constructor for reading constant_buffer
       */
      check_ranged_accessor_constructor_buffer<
          T, dims, cl::sycl::access::mode::read,
          cl::sycl::access::target::constant_buffer,
          cl::sycl::access::placeholder::true_t>::check(buffer, r, offset, log);

      /** check accessor is Copy Constructible
       */
      {
        cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                           cl::sycl::access::target::constant_buffer,
                           cl::sycl::access::placeholder::true_t>
            a(buffer);
        auto b{a};

        check_accessor_members<
            T, dims, cl::sycl::access::mode::read,
            cl::sycl::access::target::constant_buffer,
            cl::sycl::access::placeholder::true_t>::check(
                b, a.get_size(), a.get_count(), a.get_offset(),
                a.get_range(), "copy construction", log);

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
        cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                           cl::sycl::access::target::constant_buffer,
                           cl::sycl::access::placeholder::true_t>
            a(buffer);
        cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                           cl::sycl::access::target::constant_buffer,
                           cl::sycl::access::placeholder::true_t>
            b(buffer, r, offset);
        b = a;

        check_accessor_members<
            T, dims, cl::sycl::access::mode::read,
            cl::sycl::access::target::constant_buffer,
            cl::sycl::access::placeholder::true_t>::check(
                b, a.get_size(), a.get_count(),
                a.get_offset(), a.get_range(), "copy assignment", log);
      }

      /** check accessor is Move Constructible
       */
      {
        cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                           cl::sycl::access::target::constant_buffer,
                           cl::sycl::access::placeholder::true_t>
            a(buffer);
        auto b{std::move(a)};

        check_accessor_members<
            T, dims, cl::sycl::access::mode::read,
            cl::sycl::access::target::constant_buffer,
            cl::sycl::access::placeholder::true_t>::check(
                b, buffer.get_size(), buffer.get_count(),
                getId<dims>(0), buffer.get_range(), "move construction", log);
      }

      /** check accessor is Move Assignable
       */
      {
        cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                           cl::sycl::access::target::constant_buffer,
                           cl::sycl::access::placeholder::true_t>
            a(buffer);
        cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                           cl::sycl::access::target::constant_buffer,
                           cl::sycl::access::placeholder::true_t>
            b(buffer, r, offset);
        b = std::move(a);

        check_accessor_members<
            T, dims, cl::sycl::access::mode::read,
            cl::sycl::access::target::constant_buffer,
            cl::sycl::access::placeholder::true_t>::check(
                b, buffer.get_size(), buffer.get_count(),
                getId<dims>(0), buffer.get_range(), "move assignment", log);
      }

      queue.submit([&](cl::sycl::handler &h) {
        /** dummy kernel as no kernel is required for these checks
         */
        h.single_task(
            dummy_functor<T, cl::sycl::access::target::constant_buffer>{});

      });
      queue.wait_and_throw();
    }
  }
};

/** Specialization of buffer_accessor_dims for the combinations of 0 dimentional
 * global_buffer and constant_buffer
 */
template <typename T, cl::sycl::access::placeholder isPlaceholder>
class buffer_accessor_dims<T, 0, is_host_buffer::false_t, isPlaceholder> {
 public:
  static void check(util::logger &log, cl::sycl::queue &queue) {
    cl::sycl::range<1> range = getRange<1>(1);
    std::vector<uint8_t> data(sizeof(T), 0);
    cl::sycl::buffer<T, 1> buffer(reinterpret_cast<T *>(data.data()), range);

    /** check buffer accessor constructors for global_buffer
     */
    {
      queue.submit([&](cl::sycl::handler &h) {
        /** check (buffer, handler) constructor for reading
         * global_buffer
         */
        check_accessor_constructor_buffer<
            T, 0, cl::sycl::access::mode::read,
            cl::sycl::access::target::global_buffer,
            isPlaceholder>::check(buffer, h, log);

        /** check (buffer, handler) constructor for writing global_buffer
         */
        check_accessor_constructor_buffer<
            T, 0, cl::sycl::access::mode::write,
            cl::sycl::access::target::global_buffer,
            isPlaceholder>::check(buffer, h, log);

        /** check (buffer, handler) constructor for read_write global_buffer
         */
        check_accessor_constructor_buffer<
            T, 0, cl::sycl::access::mode::read_write,
            cl::sycl::access::target::global_buffer,
            isPlaceholder>::check(buffer, h, log);

        /** check (buffer, handler) constructor for discard_write global_buffer
         */
        check_accessor_constructor_buffer<
            T, 0, cl::sycl::access::mode::discard_write,
            cl::sycl::access::target::global_buffer,
            isPlaceholder>::check(buffer, h, log);

        /** check (buffer, handler) constructor for discard_read_write
         * global_buffer
         */
        check_accessor_constructor_buffer<
            T, 0, cl::sycl::access::mode::discard_read_write,
            cl::sycl::access::target::global_buffer,
            isPlaceholder>::check(buffer, h, log);

        /** check (buffer, handler) global_buffer accessor constructors for
         *  atomic
         */

        check_accessor_constructor_buffer<
            T, 0, cl::sycl::access::mode::atomic,
            cl::sycl::access::target::global_buffer,
            isPlaceholder>::check(buffer, h, log);

        /** check accessor is Copy Constructible
         */
        {
          cl::sycl::accessor<T, 0, cl::sycl::access::mode::read,
                             cl::sycl::access::target::global_buffer,
                             isPlaceholder>
              a(buffer, h);
          auto b{a};

          check_accessor_members<T, 0, cl::sycl::access::mode::read,
                                 cl::sycl::access::target::global_buffer,
                                 isPlaceholder>::check(b, a.get_size(),
                                                       a.get_count(),
                                                       "copy construction",
                                                       log);

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
          cl::sycl::accessor<T, 0, cl::sycl::access::mode::read,
                             cl::sycl::access::target::global_buffer,
                             isPlaceholder>
              a(buffer, h);
          cl::sycl::accessor<T, 0, cl::sycl::access::mode::read,
                             cl::sycl::access::target::global_buffer,
                             isPlaceholder>
              b(buffer, h);
          b = a;

          check_accessor_members<T, 0, cl::sycl::access::mode::read,
                                 cl::sycl::access::target::global_buffer,
                                 isPlaceholder>::check(b, a.get_size(),
                                                       a.get_count(),
                                                       "copy assignment", log);
        }

        /** check accessor is Move Constructible
         */
        {
          cl::sycl::accessor<T, 0, cl::sycl::access::mode::read,
                             cl::sycl::access::target::global_buffer,
                             isPlaceholder>
              a(buffer, h);
          auto b{std::move(a)};

          check_accessor_members<T, 0, cl::sycl::access::mode::read,
                                 cl::sycl::access::target::global_buffer,
                                 isPlaceholder>::check(b, buffer.get_size(),
                                                       buffer.get_count(),
                                                       "move construction",
                                                       log);
        }

        /** check accessor is Move Assignable
         */
        {
          cl::sycl::accessor<T, 0, cl::sycl::access::mode::read,
                             cl::sycl::access::target::global_buffer,
                             isPlaceholder>
              a(buffer, h);
          cl::sycl::accessor<T, 0, cl::sycl::access::mode::read,
                             cl::sycl::access::target::global_buffer,
                             isPlaceholder>
              b(buffer, h);
          b = std::move(a);

          check_accessor_members<T, 0, cl::sycl::access::mode::read,
                                 cl::sycl::access::target::global_buffer,
                                 isPlaceholder>::check(b, buffer.get_size(),
                                                       buffer.get_count(),
                                                       "move assignment", log);
        }

        /** dummy kernel as no kernel is required for these checks
         */
        h.single_task(
            dummy_functor<T, cl::sycl::access::target::global_buffer>{});
      });
      queue.wait_and_throw();
    }

    /** check buffer accessor constructors for constant_buffer
     */
    {
      queue.submit([&](cl::sycl::handler &h) {
        /** check (buffer, handler) constructor for reading constant_buffer
         */
        check_accessor_constructor_buffer<
            T, 0, cl::sycl::access::mode::read,
            cl::sycl::access::target::constant_buffer,
            isPlaceholder>::check(buffer, h, log);

        /** check accessor is Copy Constructible
         */
        {
          cl::sycl::accessor<T, 0, cl::sycl::access::mode::read,
                             cl::sycl::access::target::constant_buffer,
                             isPlaceholder>
              a(buffer, h);
          auto b{a};

          check_accessor_members<T, 0, cl::sycl::access::mode::read,
                                 cl::sycl::access::target::constant_buffer,
                                 isPlaceholder>::check(b, a.get_size(),
                                                       a.get_count(),
                                                       "copy construction",
                                                       log);

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
          cl::sycl::accessor<T, 0, cl::sycl::access::mode::read,
                             cl::sycl::access::target::constant_buffer,
                             isPlaceholder>
              a(buffer, h);
          cl::sycl::accessor<T, 0, cl::sycl::access::mode::read,
                             cl::sycl::access::target::constant_buffer,
                             isPlaceholder>
              b(buffer, h);
          b = a;

          check_accessor_members<T, 0, cl::sycl::access::mode::read,
                                 cl::sycl::access::target::constant_buffer,
                                 isPlaceholder>::check(b, a.get_size(),
                                                       a.get_count(),
                                                       "copy assignment", log);
        }

        /** check accessor is Move Constructible
         */
        {
          cl::sycl::accessor<T, 0, cl::sycl::access::mode::read,
                             cl::sycl::access::target::constant_buffer,
                             isPlaceholder>
              a(buffer, h);
          auto b{std::move(a)};

          check_accessor_members<T, 0, cl::sycl::access::mode::read,
                                 cl::sycl::access::target::constant_buffer,
                                 isPlaceholder>::check(b, buffer.get_size(),
                                                       buffer.get_count(),
                                                       "move construction",
                                                       log);
        }

        /** check accessor is Move Assignable
         */
        {
          cl::sycl::accessor<T, 0, cl::sycl::access::mode::read,
                             cl::sycl::access::target::constant_buffer,
                             isPlaceholder>
              a(buffer, h);
          cl::sycl::accessor<T, 0, cl::sycl::access::mode::read,
                             cl::sycl::access::target::constant_buffer,
                             isPlaceholder>
              b(buffer, h);
          b = std::move(a);

          check_accessor_members<T, 0, cl::sycl::access::mode::read,
                                 cl::sycl::access::target::constant_buffer,
                                 isPlaceholder>::check(b, buffer.get_size(),
                                                       buffer.get_count(),
                                                       "move assignment", log);
        }

        /** dummy kernel as no kernel is required for these checks
         */
        h.single_task(
            dummy_functor<T, cl::sycl::access::target::constant_buffer>{});

      });
      queue.wait_and_throw();
    }
  }
};

/** Specialization of buffer_accessor_dims for 0 dimentional host_buffer
 */
template <typename T, cl::sycl::access::placeholder isPlaceholder>
class buffer_accessor_dims<T, 0, is_host_buffer::true_t, isPlaceholder> {
 public:
  static void check(util::logger &log, cl::sycl::queue &queue) {
    cl::sycl::range<1> range = getRange<1>(1);
    std::vector<uint8_t> data(sizeof(T), 0);
    cl::sycl::buffer<T, 1> buffer(reinterpret_cast<T *>(data.data()), range);

    /** check buffer accessor constructors for host_buffer
     */
    {
      /** check (buffer) constructor for reading host_buffer
       */
      check_accessor_constructor_buffer<T, 0, cl::sycl::access::mode::read,
                                        cl::sycl::access::target::host_buffer,
                                        isPlaceholder>::check(buffer, log);

      /** check (buffer) constructor for writing host_buffer
       */
      check_accessor_constructor_buffer<T, 0, cl::sycl::access::mode::write,
                                        cl::sycl::access::target::host_buffer,
                                        isPlaceholder>::check(buffer, log);

      /** check (buffer) constructor for read_write host_buffer
       */
      check_accessor_constructor_buffer<
          T, 0, cl::sycl::access::mode::read_write,
          cl::sycl::access::target::host_buffer, isPlaceholder>::check(buffer,
                                                                       log);

      /** check (buffer) constructor for discard_write host_buffer
       */
      check_accessor_constructor_buffer<
          T, 0, cl::sycl::access::mode::discard_write,
          cl::sycl::access::target::host_buffer, isPlaceholder>::check(buffer,
                                                                       log);

      /** check (buffer) constructor for discard_read_write host_buffer
       */
      check_accessor_constructor_buffer<
          T, 0, cl::sycl::access::mode::discard_read_write,
          cl::sycl::access::target::host_buffer, isPlaceholder>::check(buffer,
                                                                       log);

      /** check accessor is Copy Constructible
       */
      {
        cl::sycl::accessor<T, 0, cl::sycl::access::mode::read,
                           cl::sycl::access::target::host_buffer, isPlaceholder>
            a(buffer);
        auto b{a};

        check_accessor_members<T, 0, cl::sycl::access::mode::read,
                               cl::sycl::access::target::host_buffer,
                               isPlaceholder>::check(b, a.get_size(),
                                                     a.get_count(),
                                                     "copy construction", log);

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
        cl::sycl::accessor<T, 0, cl::sycl::access::mode::read,
                           cl::sycl::access::target::host_buffer, isPlaceholder>
            a(buffer);
        cl::sycl::accessor<T, 0, cl::sycl::access::mode::read,
                           cl::sycl::access::target::host_buffer, isPlaceholder>
            b(buffer);
        b = a;

        check_accessor_members<T, 0, cl::sycl::access::mode::read,
                               cl::sycl::access::target::host_buffer,
                               isPlaceholder>::check(b, a.get_size(),
                                                     a.get_count(),
                                                     "copy assignment", log);
      }

      /** check accessor is Move Constructible
       */
      {
        cl::sycl::accessor<T, 0, cl::sycl::access::mode::read,
                           cl::sycl::access::target::host_buffer, isPlaceholder>
            a(buffer);
        auto b{std::move(a)};

        check_accessor_members<T, 0, cl::sycl::access::mode::read,
                               cl::sycl::access::target::host_buffer,
                               isPlaceholder>::check(b, buffer.get_size(),
                                                     buffer.get_count(),
                                                     "move construction", log);
      }

      /** check accessor is Move Assignable
       */
      {
        cl::sycl::accessor<T, 0, cl::sycl::access::mode::read,
                           cl::sycl::access::target::host_buffer, isPlaceholder>
            a(buffer);
        cl::sycl::accessor<T, 0, cl::sycl::access::mode::read,
                           cl::sycl::access::target::host_buffer, isPlaceholder>
            b(buffer);
        b = std::move(a);

        check_accessor_members<T, 0, cl::sycl::access::mode::read,
                               cl::sycl::access::target::host_buffer,
                               isPlaceholder>::check(b, buffer.get_size(),
                                                     buffer.get_count(),
                                                     "move assignment", log);
      }
    }
  }
};

/** Specialization of buffer_accessor_dims for the combinations of 0 dimentional
 *  placeholder global_buffer and constant_buffer
 */
template <typename T>
class buffer_accessor_dims<T, 0, is_host_buffer::false_t,
                           cl::sycl::access::placeholder::true_t> {
 public:
  static void check(util::logger &log, cl::sycl::queue &queue) {
    cl::sycl::range<1> range = getRange<1>(1);
    std::vector<uint8_t> data(sizeof(T), 0);
    cl::sycl::buffer<T, 1> buffer(reinterpret_cast<T *>(data.data()), range);

    /** check buffer accessor constructors for global_buffer
     */
    {
      /** check (buffer) constructor for reading global_buffer
       */
      check_accessor_constructor_buffer<
          T, 0, cl::sycl::access::mode::read,
          cl::sycl::access::target::global_buffer,
          cl::sycl::access::placeholder::true_t>::check(buffer, log);

      /** check (buffer) constructor for writing global_buffer
       */
      check_accessor_constructor_buffer<
          T, 0, cl::sycl::access::mode::write,
          cl::sycl::access::target::global_buffer,
          cl::sycl::access::placeholder::true_t>::check(buffer, log);

      /** check (buffer) constructor for read_write global_buffer
       */
      check_accessor_constructor_buffer<
          T, 0, cl::sycl::access::mode::read_write,
          cl::sycl::access::target::global_buffer,
          cl::sycl::access::placeholder::true_t>::check(buffer, log);

      /** check (buffer) constructor for discard_write global_buffer
       */
      check_accessor_constructor_buffer<
          T, 0, cl::sycl::access::mode::discard_write,
          cl::sycl::access::target::global_buffer,
          cl::sycl::access::placeholder::true_t>::check(buffer, log);

      /** check (buffer) constructor for discard_read_write global_buffer
       */
      check_accessor_constructor_buffer<
          T, 0, cl::sycl::access::mode::discard_read_write,
          cl::sycl::access::target::global_buffer,
          cl::sycl::access::placeholder::true_t>::check(buffer, log);

      /** check (buffer) global_buffer accessor constructors for atomic
       */
      check_accessor_constructor_buffer<
          T, 0, cl::sycl::access::mode::atomic,
          cl::sycl::access::target::global_buffer,
          cl::sycl::access::placeholder::true_t>::check(buffer, log);

      /** check accessor is Copy Constructible
       */
      {
        cl::sycl::accessor<T, 0, cl::sycl::access::mode::read,
                           cl::sycl::access::target::global_buffer,
                           cl::sycl::access::placeholder::true_t>
            a(buffer);
        auto b{a};

        check_accessor_members<
            T, 0, cl::sycl::access::mode::read,
            cl::sycl::access::target::global_buffer,
            cl::sycl::access::placeholder::true_t>::check(
                b, a.get_size(), a.get_count(), "copy construction", log);

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
        cl::sycl::accessor<T, 0, cl::sycl::access::mode::read,
                           cl::sycl::access::target::global_buffer,
                           cl::sycl::access::placeholder::true_t>
            a(buffer);
        cl::sycl::accessor<T, 0, cl::sycl::access::mode::read,
                           cl::sycl::access::target::global_buffer,
                           cl::sycl::access::placeholder::true_t>
            b(buffer);
        b = a;

        check_accessor_members<
            T, 0, cl::sycl::access::mode::read,
            cl::sycl::access::target::global_buffer,
            cl::sycl::access::placeholder::true_t>::check(
                b, a.get_size(), a.get_count(), "copy assignment", log);
      }

      /** check accessor is Move Constructible
       */
      {
        cl::sycl::accessor<T, 0, cl::sycl::access::mode::read,
                           cl::sycl::access::target::global_buffer,
                           cl::sycl::access::placeholder::true_t>
            a(buffer);
        auto b{std::move(a)};

        check_accessor_members<
            T, 0, cl::sycl::access::mode::read,
            cl::sycl::access::target::global_buffer,
            cl::sycl::access::placeholder::true_t>::check(
                b, buffer.get_size(), buffer.get_count(),
                "move construction", log);
      }

      /** check accessor is Move Assignable
       */
      {
        cl::sycl::accessor<T, 0, cl::sycl::access::mode::read,
                           cl::sycl::access::target::global_buffer,
                           cl::sycl::access::placeholder::true_t>
            a(buffer);
        cl::sycl::accessor<T, 0, cl::sycl::access::mode::read,
                           cl::sycl::access::target::global_buffer,
                           cl::sycl::access::placeholder::true_t>
            b(buffer);
        b = std::move(a);

        check_accessor_members<
            T, 0, cl::sycl::access::mode::read,
            cl::sycl::access::target::global_buffer,
            cl::sycl::access::placeholder::true_t>::check(
                b, buffer.get_size(), buffer.get_count(),
                "move assignment", log);
      }

      queue.submit([&](cl::sycl::handler &h) {//xxyy - TODO: try using those placeholder accessors here
        /** dummy kernel as no kernel is required for these checks
         */
        h.single_task(
            dummy_functor<T, cl::sycl::access::target::global_buffer>{});
      });
      queue.wait_and_throw();
    }

    /** check buffer accessor constructors for constant_buffer
     */
    {
      /** check (buffer) constructor for reading constant_buffer
       */
      check_accessor_constructor_buffer<
          T, 0, cl::sycl::access::mode::read,
          cl::sycl::access::target::constant_buffer,
          cl::sycl::access::placeholder::true_t>::check(buffer, log);

      /** check accessor is Copy Constructible
       */
      {
        cl::sycl::accessor<T, 0, cl::sycl::access::mode::read,
                           cl::sycl::access::target::constant_buffer,
                           cl::sycl::access::placeholder::true_t>
            a(buffer);
        auto b{a};

        check_accessor_members<
            T, 0, cl::sycl::access::mode::read,
            cl::sycl::access::target::constant_buffer,
            cl::sycl::access::placeholder::true_t>::check(
                b, a.get_size(), a.get_count(), "copy construction", log);

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
        cl::sycl::accessor<T, 0, cl::sycl::access::mode::read,
                           cl::sycl::access::target::constant_buffer,
                           cl::sycl::access::placeholder::true_t>
            a(buffer);
        cl::sycl::accessor<T, 0, cl::sycl::access::mode::read,
                           cl::sycl::access::target::constant_buffer,
                           cl::sycl::access::placeholder::true_t>
            b(buffer);
        b = a;

        check_accessor_members<
            T, 0, cl::sycl::access::mode::read,
            cl::sycl::access::target::constant_buffer,
            cl::sycl::access::placeholder::true_t>::check(
                b, a.get_size(), a.get_count(), "copy assignment", log);
      }

      /** check accessor is Move Constructible
       */
      {
        cl::sycl::accessor<T, 0, cl::sycl::access::mode::read,
                           cl::sycl::access::target::constant_buffer,
                           cl::sycl::access::placeholder::true_t>
            a(buffer);
        auto b{std::move(a)};

        check_accessor_members<
            T, 0, cl::sycl::access::mode::read,
            cl::sycl::access::target::constant_buffer,
            cl::sycl::access::placeholder::true_t>::check(
                b, buffer.get_size(), buffer.get_count(),
                "move construction", log);
      }

      /** check accessor is Move Assignable
       */
      {
        cl::sycl::accessor<T, 0, cl::sycl::access::mode::read,
                           cl::sycl::access::target::constant_buffer,
                           cl::sycl::access::placeholder::true_t>
            a(buffer);
        cl::sycl::accessor<T, 0, cl::sycl::access::mode::read,
                           cl::sycl::access::target::constant_buffer,
                           cl::sycl::access::placeholder::true_t>
            b(buffer);
        b = std::move(a);

        check_accessor_members<
            T, 0, cl::sycl::access::mode::read,
            cl::sycl::access::target::constant_buffer,
            cl::sycl::access::placeholder::true_t>::check(
                b, buffer.get_size(), buffer.get_count(),
                "move assignment", log);
      }

      queue.submit([&](cl::sycl::handler &h) {
        /** dummy kernel as no kernel is required for these checks
         */
        h.single_task(
            dummy_functor<T, cl::sycl::access::target::constant_buffer>{});

      });
      queue.wait_and_throw();
    }
  }
};
}  // namespace accessor_utility__

#endif  // SYCL_1_2_1_TESTS_ACCESSOR_ACCESSOR_CONSTRUCTORS_BUFFER_UTILITY_H
