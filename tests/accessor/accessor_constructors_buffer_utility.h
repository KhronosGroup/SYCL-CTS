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
template <typename accTag>
class check_accessor_constructor_buffer {
  static constexpr size_t bufferDims = ((accTag::dims == 0) ? 1 : accTag::dims);
public:
  template <typename ... handlerArgsT>
  static void check(cl::sycl::buffer<typename accTag::dataT, bufferDims> &buffer,
                    sycl_cts::util::logger &log,
                    const std::string& constructorName,
                    const std::string& typeName,
                    handlerArgsT&& ... handler) {
    // construct the accessor
    typename accTag::type accessor(buffer,
                                   std::forward<handlerArgsT>(handler)...);

    // check the accessor
    check_accessor_members<accTag>::check(
        log, accessor, constructorName, typeName,
        accessor_members::size{buffer.get_size()},
        accessor_members::count{buffer.get_count()},
        accessor_members::offset<bufferDims>{getId<bufferDims>(0)},
        accessor_members::range<bufferDims>{buffer.get_range()},
        accessor_members::placeholder{accTag::placeholder});
  }
  template <typename ... handlerArgsT>
  static void check(cl::sycl::buffer<typename accTag::dataT, bufferDims> &buffer,
                    cl::sycl::range<bufferDims> range,
                    sycl_cts::util::logger &log,
                    const std::string& constructorName,
                    const std::string& typeName,
                    handlerArgsT&& ... handler) {
    // construct the accessor
    typename accTag::type accessor(buffer,
                                   std::forward<handlerArgsT>(handler)...,
                                   range);

    // check the accessor
    check_accessor_members<accTag>::check(
        log, accessor, constructorName, typeName,
        accessor_members::size{range.size() * sizeof(typename accTag::dataT)},
        accessor_members::count{range.size()},
        accessor_members::offset<bufferDims>{getId<bufferDims>(0)},
        accessor_members::range<bufferDims>{range},
        accessor_members::placeholder{accTag::placeholder});
  }
  template <typename ... handlerArgsT>
  static void check(cl::sycl::buffer<typename accTag::dataT, bufferDims> &buffer,
                    cl::sycl::range<bufferDims> range,
                    cl::sycl::id<bufferDims> offset,
                    sycl_cts::util::logger &log,
                    const std::string& constructorName,
                    const std::string& typeName,
                    handlerArgsT&& ... handler) {
    // construct the accessor
    typename accTag::type accessor(buffer,
                                   std::forward<handlerArgsT>(handler)...,
                                   range, offset);

    // check the accessor
    check_accessor_members<accTag>::check(
        log, accessor, constructorName, typeName,
        accessor_members::size{range.size() * sizeof(typename accTag::dataT)},
        accessor_members::count{range.size()},
        accessor_members::offset<bufferDims>{offset},
        accessor_members::range<bufferDims>{range},
        accessor_members::placeholder{accTag::placeholder});
  }
};


 /** check accessor is Copy Constructible
  */
template <typename accTag>
class check_accessor_copy_constructable {
  static constexpr size_t bufferDims = ((accTag::dims == 0) ? 1 : accTag::dims);
public:
  static void check(const typename accTag::type& a,
                    sycl_cts::util::logger &log,
                    const std::string& typeName) {
    auto b{a};

    check_accessor_members<accTag>::check(
        log, b, "copy construction", typeName,
        accessor_members::size::from<accTag>(a),
        accessor_members::count::from<accTag>(a),
        accessor_members::offset<bufferDims>::template from<accTag>(a),
        accessor_members::range<bufferDims>::template from<accTag>(a),
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

/** check accessor is Copy Assignable
 */
template <typename accTag>
class check_accessor_copy_assignable {
  static constexpr size_t bufferDims = ((accTag::dims == 0) ? 1 : accTag::dims);
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
        accessor_members::offset<bufferDims>::template from<accTag>(a),
        accessor_members::range<bufferDims>::template from<accTag>(a),
        accessor_members::placeholder::from<accTag>(a));
  }
};

/** check accessor is Move Constructible
 */
template <typename accTag>
class check_accessor_move_constructable {
  static constexpr size_t bufferDims = ((accTag::dims == 0) ? 1 : accTag::dims);
public:
  static void check(const typename accTag::type& a,
                    sycl_cts::util::logger &log,
                    const std::string& typeName) {
    auto b{std::move(a)};

    check_accessor_members<accTag>::check(
        log, b, "move construction", typeName,
        accessor_members::size::from<accTag>(a),
        accessor_members::count::from<accTag>(a),
        accessor_members::offset<bufferDims>::template from<accTag>(a),
        accessor_members::range<bufferDims>::template from<accTag>(a),
        accessor_members::placeholder::from<accTag>(a));
  }
};

/** check accessor is Move Assignable
 */
template <typename accTag>
class check_accessor_move_assignable {
  static constexpr size_t bufferDims = ((accTag::dims == 0) ? 1 : accTag::dims);
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
        accessor_members::offset<bufferDims>::template from<accTag>(a),
        accessor_members::range<bufferDims>::template from<accTag>(a),
        accessor_members::placeholder::from<accTag>(a));
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
          cl::sycl::access::placeholder placeholder =
              cl::sycl::access::placeholder::false_t>
class buffer_accessor_dims {
 public:
  static void check(util::logger &log, cl::sycl::queue &queue,
                    const std::string& typeName) {
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
      constexpr auto target = cl::sycl::access::target::global_buffer;

      queue.submit([&](cl::sycl::handler &h) {
        /** check constructors for reading global_buffer
         */
        {
          constexpr auto mode = cl::sycl::access::mode::read;
          using accTag = accessor_type_info<T, dims, mode, target, placeholder>;
          using verifier = check_accessor_constructor_buffer<accTag>;

          verifier::check(buffer,
                          log, "constructor(buffer, handler)",
                          typeName, h);
          verifier::check(buffer, r,
                          log, "constructor(buffer, handler, range)",
                          typeName, h);
          verifier::check(buffer, r, offset,
                          log, "constructor(buffer, handler, range, offset)",
                          typeName, h);
        }

        /** check constructors for writing global_buffer
         */
        {
          constexpr auto mode = cl::sycl::access::mode::write;
          using accTag = accessor_type_info<T, dims, mode, target, placeholder>;
          using verifier = check_accessor_constructor_buffer<accTag>;

          verifier::check(buffer,
                          log, "constructor(buffer, handler)",
                          typeName, h);
          verifier::check(buffer, r,
                          log, "constructor(buffer, handler, range)",
                          typeName, h);
          verifier::check(buffer, r, offset,
                          log, "constructor(buffer, handler, range, offset)",
                          typeName, h);
        }

        /** check constructors for read_write global_buffer
         */
        {
          constexpr auto mode = cl::sycl::access::mode::read_write;
          using accTag = accessor_type_info<T, dims, mode, target, placeholder>;
          using verifier = check_accessor_constructor_buffer<accTag>;

          verifier::check(buffer,
                          log, "constructor(buffer, handler)",
                          typeName, h);
          verifier::check(buffer, r,
                          log, "constructor(buffer, handler, range)",
                          typeName, h);
          verifier::check(buffer, r, offset,
                          log, "constructor(buffer, handler, range, offset)",
                          typeName, h);
        }

        /** check constructors for discard_write global_buffer
         */
        {
          constexpr auto mode = cl::sycl::access::mode::discard_write;
          using accTag = accessor_type_info<T, dims, mode, target, placeholder>;
          using verifier = check_accessor_constructor_buffer<accTag>;

          verifier::check(buffer,
                          log, "constructor(buffer, handler)",
                          typeName, h);
          verifier::check(buffer, r,
                          log, "constructor(buffer, handler, range)",
                          typeName, h);
          verifier::check(buffer, r, offset,
                          log, "constructor(buffer, handler, range, offset)",
                          typeName, h);
        }

        /** check constructors for discard_read_write global_buffer
         */
        {
          constexpr auto mode = cl::sycl::access::mode::discard_read_write;
          using accTag = accessor_type_info<T, dims, mode, target, placeholder>;
          using verifier = check_accessor_constructor_buffer<accTag>;

          verifier::check(buffer,
                          log, "constructor(buffer, handler)",
                          typeName, h);
          verifier::check(buffer, r,
                          log, "constructor(buffer, handler, range)",
                          typeName, h);
          verifier::check(buffer, r, offset,
                          log, "constructor(buffer, handler, range, offset)",
                          typeName, h);
        }

        /** check constructors for atomic global_buffer
         */
        {
          constexpr auto mode = cl::sycl::access::mode::atomic;
          using accTag = accessor_type_info<T, dims, mode, target, placeholder>;
          using verifier = check_accessor_constructor_buffer<accTag>;

          verifier::check(buffer,
                          log, "constructor(buffer, handler)",
                          typeName, h);
          verifier::check(buffer, r,
                          log, "constructor(buffer, handler, range)",
                          typeName, h);
          verifier::check(buffer, r, offset,
                          log, "constructor(buffer, handler, range, offset)",
                          typeName, h);
        }

        /** check common-by-reference semantics
         */
        {
          constexpr auto mode = cl::sycl::access::mode::read;
          using accTag = accessor_type_info<T, dims, mode, target, placeholder>;
          {
            using verifier = check_accessor_copy_constructable<accTag>;

            typename accTag::type srcAccessor(buffer, h, r, offset);
            verifier::check(srcAccessor, log, typeName);
          }
          {
            using verifier = check_accessor_copy_assignable<accTag>;

            typename accTag::type srcAccessor(buffer, h, r, offset);
            typename accTag::type dstAccessor(buffer, h);

            verifier::check(srcAccessor, dstAccessor, log, typeName);
          }
          {
            using verifier = check_accessor_move_constructable<accTag>;

            typename accTag::type srcAccessor(buffer, h);

            verifier::check(srcAccessor, log, typeName);
          }
          {
            using verifier = check_accessor_move_assignable<accTag>;

            typename accTag::type srcAccessor(buffer, h);
            typename accTag::type dstAccessor(buffer, h, r, offset);

            verifier::check(srcAccessor, dstAccessor, log, typeName);
          }
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
      constexpr auto target = cl::sycl::access::target::constant_buffer;

      queue.submit([&](cl::sycl::handler &h) {
        /** check constructors for reading constant_buffer
         */
        {
          constexpr auto mode = cl::sycl::access::mode::read;
          using accTag = accessor_type_info<T, dims, mode, target, placeholder>;
          using verifier = check_accessor_constructor_buffer<accTag>;

          verifier::check(buffer,
                          log, "constructor(buffer, handler)",
                          typeName, h);
          verifier::check(buffer, r,
                          log, "constructor(buffer, handler, range)",
                          typeName, h);
          verifier::check(buffer, r, offset,
                          log, "constructor(buffer, handler, range, offset)",
                          typeName, h);
        }

        /** check common-by-reference semantics
         */
        {
          constexpr auto mode = cl::sycl::access::mode::read;
          using accTag = accessor_type_info<T, dims, mode, target, placeholder>;
          {
            using verifier = check_accessor_copy_constructable<accTag>;

            typename accTag::type srcAccessor(buffer, h);
            verifier::check(srcAccessor, log, typeName);
          }
          {
            using verifier = check_accessor_copy_assignable<accTag>;

            typename accTag::type srcAccessor(buffer, h);
            typename accTag::type dstAccessor(buffer, h, r, offset);

            verifier::check(srcAccessor, dstAccessor, log, typeName);
          }
          {
            using verifier = check_accessor_move_constructable<accTag>;

            typename accTag::type srcAccessor(buffer, h);

            verifier::check(srcAccessor, log, typeName);
          }
          {
            using verifier = check_accessor_move_assignable<accTag>;

            typename accTag::type srcAccessor(buffer, h);
            typename accTag::type dstAccessor(buffer, h, r, offset);

            verifier::check(srcAccessor, dstAccessor, log, typeName);
          }
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
template <typename T, size_t dims, cl::sycl::access::placeholder placeholder>
class buffer_accessor_dims<T, dims, is_host_buffer::true_t, placeholder> {
 public:
  static void check(util::logger &log, cl::sycl::queue &queue,
                    const std::string& typeName) {
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
      constexpr auto target = cl::sycl::access::target::host_buffer;

      /** check constructors for reading host_buffer
       */
      {
        constexpr auto mode = cl::sycl::access::mode::read;
        using accTag = accessor_type_info<T, dims, mode, target, placeholder>;
        using verifier = check_accessor_constructor_buffer<accTag>;

        verifier::check(buffer,
                        log, "constructor(buffer)",
                        typeName);
        verifier::check(buffer, r,
                        log, "constructor(buffer, range)",
                        typeName);
        verifier::check(buffer, r, offset,
                        log, "constructor(buffer, range, offset)",
                        typeName);
      }

      /** check constructors for writing host_buffer
       */
      {
        constexpr auto mode = cl::sycl::access::mode::write;
        using accTag = accessor_type_info<T, dims, mode, target, placeholder>;
        using verifier = check_accessor_constructor_buffer<accTag>;

        verifier::check(buffer,
                        log, "constructor(buffer)",
                        typeName);
        verifier::check(buffer, r,
                        log, "constructor(buffer, range)",
                        typeName);
        verifier::check(buffer, r, offset,
                        log, "constructor(buffer, range, offset)",
                        typeName);
      }

      /** check constructors for read_write host_buffer
       */
      {
        constexpr auto mode = cl::sycl::access::mode::read_write;
        using accTag = accessor_type_info<T, dims, mode, target, placeholder>;
        using verifier = check_accessor_constructor_buffer<accTag>;

        verifier::check(buffer,
                        log, "constructor(buffer)",
                        typeName);
        verifier::check(buffer, r,
                        log, "constructor(buffer, range)",
                        typeName);
        verifier::check(buffer, r, offset,
                        log, "constructor(buffer, range, offset)",
                        typeName);
      }

      /** check constructors for discard_write host_buffer
       */
      {
        constexpr auto mode = cl::sycl::access::mode::discard_write;
        using accTag = accessor_type_info<T, dims, mode, target, placeholder>;
        using verifier = check_accessor_constructor_buffer<accTag>;

        verifier::check(buffer,
                        log, "constructor(buffer)",
                        typeName);
        verifier::check(buffer, r,
                        log, "constructor(buffer, range)",
                        typeName);
        verifier::check(buffer, r, offset,
                        log, "constructor(buffer, range, offset)",
                        typeName);
      }

      /** check constructors for discard_read_write host_buffer
       */
      {
        constexpr auto mode = cl::sycl::access::mode::discard_read_write;
        using accTag = accessor_type_info<T, dims, mode, target, placeholder>;
        using verifier = check_accessor_constructor_buffer<accTag>;

        verifier::check(buffer,
                        log, "constructor(buffer)",
                        typeName);
        verifier::check(buffer, r,
                        log, "constructor(buffer, range)",
                        typeName);
        verifier::check(buffer, r, offset,
                        log, "constructor(buffer, range, offset)",
                        typeName);
      }

      /** check common-by-reference semantics
       */
      {
        constexpr auto mode = cl::sycl::access::mode::read;
        using accTag = accessor_type_info<T, dims, mode, target, placeholder>;
        {
          using verifier = check_accessor_copy_constructable<accTag>;

          typename accTag::type srcAccessor(buffer);
          verifier::check(srcAccessor, log, typeName);
        }
        {
          using verifier = check_accessor_copy_assignable<accTag>;

          typename accTag::type srcAccessor(buffer, r, offset);
          typename accTag::type dstAccessor(buffer);

          verifier::check(srcAccessor, dstAccessor, log, typeName);
        }
        {
          using verifier = check_accessor_move_constructable<accTag>;

          typename accTag::type srcAccessor(buffer);

          verifier::check(srcAccessor, log, typeName);
        }
        {
          using verifier = check_accessor_move_assignable<accTag>;

          typename accTag::type srcAccessor(buffer);
          typename accTag::type dstAccessor(buffer, r, offset);

          verifier::check(srcAccessor, dstAccessor, log, typeName);
        }
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
  static constexpr auto placeholder = cl::sycl::access::placeholder::true_t;
public:
  static void check(util::logger &log, cl::sycl::queue &queue,
                    const std::string& typeName) {
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
      constexpr auto target = cl::sycl::access::target::global_buffer;

      /** check constructors for reading global_buffer
       */
      {
        constexpr auto mode = cl::sycl::access::mode::read;
        using accTag = accessor_type_info<T, dims, mode, target, placeholder>;
        using verifier = check_accessor_constructor_buffer<accTag>;

        verifier::check(buffer,
                        log, "constructor(buffer)",
                        typeName);
        verifier::check(buffer, r,
                        log, "constructor(buffer, range)",
                        typeName);
        verifier::check(buffer, r, offset,
                        log, "constructor(buffer, range, offset)",
                        typeName);
      }

      /** check constructors for writing global_buffer
       */
      {
        constexpr auto mode = cl::sycl::access::mode::write;
        using accTag = accessor_type_info<T, dims, mode, target, placeholder>;
        using verifier = check_accessor_constructor_buffer<accTag>;

        verifier::check(buffer,
                        log, "constructor(buffer)",
                        typeName);
        verifier::check(buffer, r,
                        log, "constructor(buffer, range)",
                        typeName);
        verifier::check(buffer, r, offset,
                        log, "constructor(buffer, range, offset)",
                        typeName);
      }

      /** check constructors for read_write global_buffer
       */
      {
        constexpr auto mode = cl::sycl::access::mode::read_write;
        using accTag = accessor_type_info<T, dims, mode, target, placeholder>;
        using verifier = check_accessor_constructor_buffer<accTag>;

        verifier::check(buffer,
                        log, "constructor(buffer)",
                        typeName);
        verifier::check(buffer, r,
                        log, "constructor(buffer, range)",
                        typeName);
        verifier::check(buffer, r, offset,
                        log, "constructor(buffer, range, offset)",
                        typeName);
      }

      /** check constructors for discard_write global_buffer
       */
      {
        constexpr auto mode = cl::sycl::access::mode::discard_write;
        using accTag = accessor_type_info<T, dims, mode, target, placeholder>;
        using verifier = check_accessor_constructor_buffer<accTag>;

        verifier::check(buffer,
                        log, "constructor(buffer)",
                        typeName);
        verifier::check(buffer, r,
                        log, "constructor(buffer, range)",
                        typeName);
        verifier::check(buffer, r, offset,
                        log, "constructor(buffer, range, offset)",
                        typeName);
      }

      /** check constructors for discard_read_write global_buffer
       */
      {
        constexpr auto mode = cl::sycl::access::mode::discard_read_write;
        using accTag = accessor_type_info<T, dims, mode, target, placeholder>;
        using verifier = check_accessor_constructor_buffer<accTag>;

        verifier::check(buffer,
                        log, "constructor(buffer)",
                        typeName);
        verifier::check(buffer, r,
                        log, "constructor(buffer, range)",
                        typeName);
        verifier::check(buffer, r, offset,
                        log, "constructor(buffer, range, offset)",
                        typeName);
      }

      /** check constructors for atomic global_buffer
       */
      {
        constexpr auto mode = cl::sycl::access::mode::atomic;
        using accTag = accessor_type_info<T, dims, mode, target, placeholder>;
        using verifier = check_accessor_constructor_buffer<accTag>;

        verifier::check(buffer,
                        log, "constructor(buffer)",
                        typeName);
        verifier::check(buffer, r,
                        log, "constructor(buffer, range)",
                        typeName);
        verifier::check(buffer, r, offset,
                        log, "constructor(buffer, range, offset)",
                        typeName);
      }


      /** check common-by-reference semantics
       */
      {
        constexpr auto mode = cl::sycl::access::mode::read;
        using accTag = accessor_type_info<T, dims, mode, target, placeholder>;
        {
          using verifier = check_accessor_copy_constructable<accTag>;

          typename accTag::type srcAccessor(buffer, r, offset);
          verifier::check(srcAccessor, log, typeName);
        }
        {
          using verifier = check_accessor_copy_assignable<accTag>;

          typename accTag::type srcAccessor(buffer, r, offset);
          typename accTag::type dstAccessor(buffer);

          verifier::check(srcAccessor, dstAccessor, log, typeName);
        }
        {
          using verifier = check_accessor_move_constructable<accTag>;

          typename accTag::type srcAccessor(buffer);

          verifier::check(srcAccessor, log, typeName);
        }
        {
          using verifier = check_accessor_move_assignable<accTag>;

          typename accTag::type srcAccessor(buffer);
          typename accTag::type dstAccessor(buffer, r, offset);

          verifier::check(srcAccessor, dstAccessor, log, typeName);
        }
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
      constexpr auto target = cl::sycl::access::target::constant_buffer;

      /** check constructors for reading constant_buffer
       */
      {
        constexpr auto mode = cl::sycl::access::mode::read;
        using accTag = accessor_type_info<T, dims, mode, target, placeholder>;
        using verifier = check_accessor_constructor_buffer<accTag>;

        verifier::check(buffer,
                        log, "constructor(buffer)",
                        typeName);
        verifier::check(buffer, r,
                        log, "constructor(buffer, range)",
                        typeName);
        verifier::check(buffer, r, offset,
                        log, "constructor(buffer, range, offset)",
                        typeName);
      }

      /** check common-by-reference semantics
       */
      {
        constexpr auto mode = cl::sycl::access::mode::read;
        using accTag = accessor_type_info<T, dims, mode, target, placeholder>;
        {
          using verifier = check_accessor_copy_constructable<accTag>;

          typename accTag::type srcAccessor(buffer);
          verifier::check(srcAccessor, log, typeName);
        }
        {
          using verifier = check_accessor_copy_assignable<accTag>;

          typename accTag::type srcAccessor(buffer);
          typename accTag::type dstAccessor(buffer, r, offset);

          verifier::check(srcAccessor, dstAccessor, log, typeName);
        }
        {
          using verifier = check_accessor_move_constructable<accTag>;

          typename accTag::type srcAccessor(buffer);

          verifier::check(srcAccessor, log, typeName);
        }
        {
          using verifier = check_accessor_move_assignable<accTag>;

          typename accTag::type srcAccessor(buffer);
          typename accTag::type dstAccessor(buffer, r, offset);

          verifier::check(srcAccessor, dstAccessor, log, typeName);
        }
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
template <typename T, cl::sycl::access::placeholder placeholder>
class buffer_accessor_dims<T, 0, is_host_buffer::false_t, placeholder> {
  static constexpr size_t dims = 0;
public:
  static void check(util::logger &log, cl::sycl::queue &queue,
                    const std::string& typeName) {
    cl::sycl::range<1> range = getRange<1>(1);
    std::vector<uint8_t> data(sizeof(T), 0);
    cl::sycl::buffer<T, 1> buffer(reinterpret_cast<T *>(data.data()), range);

    /** check buffer accessor constructors for global_buffer
     */
    {
      constexpr auto target = cl::sycl::access::target::global_buffer;

      queue.submit([&](cl::sycl::handler &h) {
        /** check constructors for reading global_buffer
         */
        {
          constexpr auto mode = cl::sycl::access::mode::read;
          using accTag = accessor_type_info<T, dims, mode, target, placeholder>;
          using verifier = check_accessor_constructor_buffer<accTag>;

          verifier::check(buffer,
                          log, "constructor(buffer, handler)",
                          typeName, h);
        }

        /** check constructors for writing global_buffer
         */
        {
          constexpr auto mode = cl::sycl::access::mode::write;
          using accTag = accessor_type_info<T, dims, mode, target, placeholder>;
          using verifier = check_accessor_constructor_buffer<accTag>;

          verifier::check(buffer,
                          log, "constructor(buffer, handler)",
                          typeName, h);
        }

        /** check constructors for read_write global_buffer
         */
        {
          constexpr auto mode = cl::sycl::access::mode::read_write;
          using accTag = accessor_type_info<T, dims, mode, target, placeholder>;
          using verifier = check_accessor_constructor_buffer<accTag>;

          verifier::check(buffer,
                          log, "constructor(buffer, handler)",
                          typeName, h);
        }

        /** check constructors for discard_write global_buffer
         */
        {
          constexpr auto mode = cl::sycl::access::mode::discard_write;
          using accTag = accessor_type_info<T, dims, mode, target, placeholder>;
          using verifier = check_accessor_constructor_buffer<accTag>;

          verifier::check(buffer,
                          log, "constructor(buffer, handler)",
                          typeName, h);
        }

        /** check constructors for discard_read_write global_buffer
         */
        {
          constexpr auto mode = cl::sycl::access::mode::discard_read_write;
          using accTag = accessor_type_info<T, dims, mode, target, placeholder>;
          using verifier = check_accessor_constructor_buffer<accTag>;

          verifier::check(buffer,
                          log, "constructor(buffer, handler)",
                          typeName, h);
        }

        /** check constructors for atomic global_buffer
         */
        {
          constexpr auto mode = cl::sycl::access::mode::atomic;
          using accTag = accessor_type_info<T, dims, mode, target, placeholder>;
          using verifier = check_accessor_constructor_buffer<accTag>;

          verifier::check(buffer,
                          log, "constructor(buffer, handler)",
                          typeName, h);
        }

        /** check common-by-reference semantics
         */
        {
          constexpr auto mode = cl::sycl::access::mode::read;
          using accTag = accessor_type_info<T, dims, mode, target, placeholder>;
          {
            using verifier = check_accessor_copy_constructable<accTag>;

            typename accTag::type srcAccessor(buffer, h);
            verifier::check(srcAccessor, log, typeName);
          }
          {
            using verifier = check_accessor_copy_assignable<accTag>;

            typename accTag::type srcAccessor(buffer, h);
            typename accTag::type dstAccessor(buffer, h);

            verifier::check(srcAccessor, dstAccessor, log, typeName);
          }
          {
            using verifier = check_accessor_move_constructable<accTag>;

            typename accTag::type srcAccessor(buffer, h);

            verifier::check(srcAccessor, log, typeName);
          }
          {
            using verifier = check_accessor_move_assignable<accTag>;

            typename accTag::type srcAccessor(buffer, h);
            typename accTag::type dstAccessor(buffer, h);

            verifier::check(srcAccessor, dstAccessor, log, typeName);
          }
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
      constexpr auto target = cl::sycl::access::target::constant_buffer;

      queue.submit([&](cl::sycl::handler &h) {
        /** check constructors for reading constant_buffer
         */
        {
          constexpr auto mode = cl::sycl::access::mode::read;
          using accTag = accessor_type_info<T, dims, mode, target, placeholder>;
          using verifier = check_accessor_constructor_buffer<accTag>;

          verifier::check(buffer,
                          log, "constructor(buffer, handler)",
                          typeName, h);
        }

        /** check common-by-reference semantics
         */
        {
          constexpr auto mode = cl::sycl::access::mode::read;
          using accTag = accessor_type_info<T, dims, mode, target, placeholder>;
          {
            using verifier = check_accessor_copy_constructable<accTag>;

            typename accTag::type srcAccessor(buffer, h);
            verifier::check(srcAccessor, log, typeName);
          }
          {
            using verifier = check_accessor_copy_assignable<accTag>;

            typename accTag::type srcAccessor(buffer, h);
            typename accTag::type dstAccessor(buffer, h);

            verifier::check(srcAccessor, dstAccessor, log, typeName);
          }
          {
            using verifier = check_accessor_move_constructable<accTag>;

            typename accTag::type srcAccessor(buffer, h);

            verifier::check(srcAccessor, log, typeName);
          }
          {
            using verifier = check_accessor_move_assignable<accTag>;

            typename accTag::type srcAccessor(buffer, h);
            typename accTag::type dstAccessor(buffer, h);

            verifier::check(srcAccessor, dstAccessor, log, typeName);
          }
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
template <typename T, cl::sycl::access::placeholder placeholder>
class buffer_accessor_dims<T, 0, is_host_buffer::true_t, placeholder> {
  static constexpr size_t dims = 0;
public:
  static void check(util::logger &log, cl::sycl::queue &queue,
                    const std::string& typeName) {
    cl::sycl::range<1> range = getRange<1>(1);
    std::vector<uint8_t> data(sizeof(T), 0);
    cl::sycl::buffer<T, 1> buffer(reinterpret_cast<T *>(data.data()), range);

    /** check buffer accessor constructors for host_buffer
     */
    {
      constexpr auto target = cl::sycl::access::target::host_buffer;

      /** check constructors for reading host_buffer
       */
      {
        constexpr auto mode = cl::sycl::access::mode::read;
        using accTag = accessor_type_info<T, dims, mode, target, placeholder>;
        using verifier = check_accessor_constructor_buffer<accTag>;

        verifier::check(buffer,
                        log, "constructor(buffer)",
                        typeName);
      }

      /** check constructors for writing host_buffer
       */
      {
        constexpr auto mode = cl::sycl::access::mode::write;
        using accTag = accessor_type_info<T, dims, mode, target, placeholder>;
        using verifier = check_accessor_constructor_buffer<accTag>;

        verifier::check(buffer,
                        log, "constructor(buffer)",
                        typeName);
      }

      /** check constructors for read_write host_buffer
       */
      {
        constexpr auto mode = cl::sycl::access::mode::read_write;
        using accTag = accessor_type_info<T, dims, mode, target, placeholder>;
        using verifier = check_accessor_constructor_buffer<accTag>;

        verifier::check(buffer,
                        log, "constructor(buffer)",
                        typeName);
      }

      /** check constructors for discard_write host_buffer
       */
      {
        constexpr auto mode = cl::sycl::access::mode::discard_write;
        using accTag = accessor_type_info<T, dims, mode, target, placeholder>;
        using verifier = check_accessor_constructor_buffer<accTag>;

        verifier::check(buffer,
                        log, "constructor(buffer)",
                        typeName);
      }

      /** check constructors for discard_read_write host_buffer
       */
      {
        constexpr auto mode = cl::sycl::access::mode::discard_read_write;
        using accTag = accessor_type_info<T, dims, mode, target, placeholder>;
        using verifier = check_accessor_constructor_buffer<accTag>;

        verifier::check(buffer,
                        log, "constructor(buffer)",
                        typeName);
      }

      /** check common-by-reference semantics
       */
      {
        constexpr auto mode = cl::sycl::access::mode::read;
        using accTag = accessor_type_info<T, dims, mode, target, placeholder>;
        {
          using verifier = check_accessor_copy_constructable<accTag>;

          typename accTag::type srcAccessor(buffer);
          verifier::check(srcAccessor, log, typeName);
        }
        {
          using verifier = check_accessor_copy_assignable<accTag>;

          typename accTag::type srcAccessor(buffer);
          typename accTag::type dstAccessor(buffer);

          verifier::check(srcAccessor, dstAccessor, log, typeName);
        }
        {
          using verifier = check_accessor_move_constructable<accTag>;

          typename accTag::type srcAccessor(buffer);

          verifier::check(srcAccessor, log, typeName);
        }
        {
          using verifier = check_accessor_move_assignable<accTag>;

          typename accTag::type srcAccessor(buffer);
          typename accTag::type dstAccessor(buffer);

          verifier::check(srcAccessor, dstAccessor, log, typeName);
        }
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
  static constexpr size_t dims = 0;
  static constexpr auto placeholder = cl::sycl::access::placeholder::true_t;
public:
  static void check(util::logger &log, cl::sycl::queue &queue,
                    const std::string& typeName) {
    cl::sycl::range<1> range = getRange<1>(1);
    std::vector<uint8_t> data(sizeof(T), 0);
    cl::sycl::buffer<T, 1> buffer(reinterpret_cast<T *>(data.data()), range);

    /** check buffer accessor constructors for global_buffer
     */
    {
      constexpr auto target = cl::sycl::access::target::global_buffer;

      /** check constructors for reading global_buffer
       */
      {
        constexpr auto mode = cl::sycl::access::mode::read;
        using accTag = accessor_type_info<T, dims, mode, target, placeholder>;
        using verifier = check_accessor_constructor_buffer<accTag>;

        verifier::check(buffer,
                        log, "constructor(buffer)",
                        typeName);
      }

      /** check constructors for writing global_buffer
       */
      {
        constexpr auto mode = cl::sycl::access::mode::write;
        using accTag = accessor_type_info<T, dims, mode, target, placeholder>;
        using verifier = check_accessor_constructor_buffer<accTag>;

        verifier::check(buffer,
                        log, "constructor(buffer)",
                        typeName);
      }

      /** check constructors for read_write global_buffer
       */
      {
        constexpr auto mode = cl::sycl::access::mode::read_write;
        using accTag = accessor_type_info<T, dims, mode, target, placeholder>;
        using verifier = check_accessor_constructor_buffer<accTag>;

        verifier::check(buffer,
                        log, "constructor(buffer)",
                        typeName);
      }

      /** check constructors for discard_write global_buffer
       */
      {
        constexpr auto mode = cl::sycl::access::mode::discard_write;
        using accTag = accessor_type_info<T, dims, mode, target, placeholder>;
        using verifier = check_accessor_constructor_buffer<accTag>;

        verifier::check(buffer,
                        log, "constructor(buffer)",
                        typeName);
      }

      /** check constructors for discard_read_write global_buffer
       */
      {
        constexpr auto mode = cl::sycl::access::mode::discard_read_write;
        using accTag = accessor_type_info<T, dims, mode, target, placeholder>;
        using verifier = check_accessor_constructor_buffer<accTag>;

        verifier::check(buffer,
                        log, "constructor(buffer)",
                        typeName);
      }

      /** check constructors for atomic global_buffer
       */
      {
        constexpr auto mode = cl::sycl::access::mode::atomic;
        using accTag = accessor_type_info<T, dims, mode, target, placeholder>;
        using verifier = check_accessor_constructor_buffer<accTag>;

        verifier::check(buffer,
                        log, "constructor(buffer)",
                        typeName);
      }

      /** check common-by-reference semantics
       */
      {
        constexpr auto mode = cl::sycl::access::mode::read;
        using accTag = accessor_type_info<T, dims, mode, target, placeholder>;
        {
          using verifier = check_accessor_copy_constructable<accTag>;

          typename accTag::type srcAccessor(buffer);
          verifier::check(srcAccessor, log, typeName);
        }
        {
          using verifier = check_accessor_copy_assignable<accTag>;

          typename accTag::type srcAccessor(buffer);
          typename accTag::type dstAccessor(buffer);

          verifier::check(srcAccessor, dstAccessor, log, typeName);
        }
        {
          using verifier = check_accessor_move_constructable<accTag>;

          typename accTag::type srcAccessor(buffer);

          verifier::check(srcAccessor, log, typeName);
        }
        {
          using verifier = check_accessor_move_assignable<accTag>;

          typename accTag::type srcAccessor(buffer);
          typename accTag::type dstAccessor(buffer);

          verifier::check(srcAccessor, dstAccessor, log, typeName);
        }
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
      constexpr auto target = cl::sycl::access::target::constant_buffer;

      /** check constructors for reading constant_buffer
       */
      {
        constexpr auto mode = cl::sycl::access::mode::read;
        using accTag = accessor_type_info<T, dims, mode, target, placeholder>;
        using verifier = check_accessor_constructor_buffer<accTag>;

        verifier::check(buffer,
                        log, "constructor(buffer)",
                        typeName);
      }

      /** check common-by-reference semantics
       */
      {
        constexpr auto mode = cl::sycl::access::mode::read;
        using accTag = accessor_type_info<T, dims, mode, target, placeholder>;
        {
          using verifier = check_accessor_copy_constructable<accTag>;

          typename accTag::type srcAccessor(buffer);
          verifier::check(srcAccessor, log, typeName);
        }
        {
          using verifier = check_accessor_copy_assignable<accTag>;

          typename accTag::type srcAccessor(buffer);
          typename accTag::type dstAccessor(buffer);

          verifier::check(srcAccessor, dstAccessor, log, typeName);
        }
        {
          using verifier = check_accessor_move_constructable<accTag>;

          typename accTag::type srcAccessor(buffer);

          verifier::check(srcAccessor, log, typeName);
        }
        {
          using verifier = check_accessor_move_assignable<accTag>;

          typename accTag::type srcAccessor(buffer);
          typename accTag::type dstAccessor(buffer);

          verifier::check(srcAccessor, dstAccessor, log, typeName);
        }
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
