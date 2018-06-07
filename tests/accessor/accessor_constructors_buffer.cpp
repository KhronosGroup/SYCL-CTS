/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2018 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#define TEST_NAME accessor_constructors_buffer

#include "../common/common.h"
#include "accessor_constructors_utility.h"

namespace TEST_NAMESPACE {
/** unique dummy_functor per file
 *  this is a hack until the CMake script is fixed; kill both the alias and the
 *  dummy class once it is fixed
 */
class dummy_accessor_constructors_buffer {};
using dummy_functor = ::dummy_functor<dummy_accessor_constructors_buffer>;

template <typename T, size_t dims>
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
        /** check (buffer, handler) constructor for reading global_buffer
         */
        {
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                             cl::sycl::access::target::global_buffer,
                             cl::sycl::access::placeholder::false_t>
              a(buffer, h);

          if (a.get_size() != getElementsCount<dims>(range) * sizeof(T)) {
            FAIL(log,
                 "global_buffer accessor for read is not constructed correctly "
                 "(get_size)");
          }

          if (a.get_count() != getElementsCount<dims>(range)) {
            FAIL(log,
                 "global_buffer accessor for read is not constructed correctly "
                 "(get_count)");
          }

          if (a.get_offset() != getId<dims>(0)) {
            FAIL(log,
                 "global_buffer accessor for read is not constructed correctly "
                 "(get_offset)");
          }

          if (a.get_range() != range) {
            FAIL(log,
                 "global_buffer accessor for read is not constructed correctly "
                 "(get_range)");
          }

          if (a.is_placeholder() != false) {
            FAIL(log,
                 "global_buffer accessor for read is not constructed correctly "
                 "(is_placeholder)");
          }
        }

        /** check (buffer, handler, range, offset) constructor for reading
         * global_buffer
         */
        {
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                             cl::sycl::access::target::global_buffer,
                             cl::sycl::access::placeholder::false_t>
              a(buffer, h, r, offset);

          if (a.get_size() != getElementsCount<dims>(range) * sizeof(T)) {
            FAIL(log,
                 "global_buffer ranged accessor for read is not constructed "
                 "correctly (get_size)");
          }

          if (a.get_count() != getElementsCount<dims>(range)) {
            FAIL(log,
                 "global_buffer ranged accessor for read is not constructed "
                 "correctly (get_count)");
          }

          if (a.get_offset() != offset) {
            FAIL(log,
                 "global_buffer ranged accessor for read is not constructed "
                 "correctly (get_offset)");
          }

          if (a.get_range() != range) {
            FAIL(log,
                 "global_buffer ranged accessor for read is not constructed "
                 "correctly (get_range)");
          }

          if (a.is_placeholder() != false) {
            FAIL(log,
                 "global_buffer ranged accessor for read is not constructed "
                 "correctly (is_placeholder)");
          }
        }

        /** check (buffer, handler) constructor for writing global_buffer
         */
        {
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::write,
                             cl::sycl::access::target::global_buffer,
                             cl::sycl::access::placeholder::false_t>
              a(buffer, h);

          if (a.get_size() != getElementsCount<dims>(range) * sizeof(T)) {
            FAIL(
                log,
                "global_buffer accessor for write is not constructed correctly "
                "(get_size)");
          }

          if (a.get_count() != getElementsCount<dims>(range)) {
            FAIL(
                log,
                "global_buffer accessor for write is not constructed correctly "
                "(get_count)");
          }

          if (a.get_offset() != getId<dims>(0)) {
            FAIL(
                log,
                "global_buffer accessor for write is not constructed correctly "
                "(get_offset)");
          }

          if (a.get_range() != range) {
            FAIL(
                log,
                "global_buffer accessor for write is not constructed correctly "
                "(get_range)");
          }

          if (a.is_placeholder() != false) {
            FAIL(
                log,
                "global_buffer accessor for write is not constructed correctly "
                "(is_placeholder)");
          }
        }
        /** check (buffer, handler, range, offset) constructor for writing
         * global_buffer
         */
        {
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::write,
                             cl::sycl::access::target::global_buffer,
                             cl::sycl::access::placeholder::false_t>
              a(buffer, h, r, offset);

          if (a.get_size() != getElementsCount<dims>(range) * sizeof(T)) {
            FAIL(log,
                 "global_buffer ranged accessor for write is not constructed "
                 "correctly (get_size)");
          }

          if (a.get_count() != getElementsCount<dims>(range)) {
            FAIL(log,
                 "global_buffer ranged accessor for write is not constructed "
                 "correctly (get_count)");
          }

          if (a.get_offset() != offset) {
            FAIL(log,
                 "global_buffer ranged accessor for write is not constructed "
                 "correctly (get_offset)");
          }

          if (a.get_range() != range) {
            FAIL(log,
                 "global_buffer ranged accessor for write is not constructed "
                 "correctly (get_range)");
          }

          if (a.is_placeholder() != false) {
            FAIL(log,
                 "global_buffer ranged accessor for write is not constructed "
                 "correctly (is_placeholder)");
          }
        }

        /** check (buffer, handler) constructor for read_write global_buffer
         */
        {
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::read_write,
                             cl::sycl::access::target::global_buffer,
                             cl::sycl::access::placeholder::false_t>
              a(buffer, h);

          if (a.get_size() != getElementsCount<dims>(range) * sizeof(T)) {
            FAIL(log,
                 "global_buffer accessor for read_write is not constructed "
                 "correctly (get_size)");
          }

          if (a.get_count() != getElementsCount<dims>(range)) {
            FAIL(log,
                 "global_buffer accessor for read_write is not constructed "
                 "correctly (get_count)");
          }

          if (a.get_offset() != getId<dims>(0)) {
            FAIL(log,
                 "global_buffer accessor for read_write is not constructed "
                 "correctly (get_offset)");
          }

          if (a.get_range() != range) {
            FAIL(log,
                 "global_buffer accessor for read_write is not constructed "
                 "correctly (get_range)");
          }

          if (a.is_placeholder() != false) {
            FAIL(log,
                 "global_buffer accessor for read_write is not constructed "
                 "correctly (is_placeholder)");
          }
        }
        /** check (buffer, handler, range, offset) constructor for read_write
         * global_buffer
         */
        {
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::read_write,
                             cl::sycl::access::target::global_buffer,
                             cl::sycl::access::placeholder::false_t>
              a(buffer, h, r, offset);

          if (a.get_size() != getElementsCount<dims>(range) * sizeof(T)) {
            FAIL(log,
                 "global_buffer ranged accessor for read_write is not "
                 "constructed correctly (get_size)");
          }

          if (a.get_count() != getElementsCount<dims>(range)) {
            FAIL(log,
                 "global_buffer ranged accessor for read_write is not "
                 "constructed correctly (get_count)");
          }

          if (a.get_offset() != offset) {
            FAIL(log,
                 "global_buffer ranged accessor for read_write is not "
                 "constructed correctly (get_offset)");
          }

          if (a.get_range() != range) {
            FAIL(log,
                 "global_buffer ranged accessor for read_write is not "
                 "constructed correctly (get_range)");
          }

          if (a.is_placeholder() != false) {
            FAIL(log,
                 "global_buffer ranged accessor for read_write is not "
                 "constructed correctly (is_placeholder)");
          }
        }

        /** check (buffer, handler) constructor for discard_write global_buffer
         */
        {
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::discard_write,
                             cl::sycl::access::target::global_buffer,
                             cl::sycl::access::placeholder::false_t>
              a(buffer, h);

          if (a.get_size() != getElementsCount<dims>(range) * sizeof(T)) {
            FAIL(log,
                 "global_buffer accessor for discard_write is not constructed "
                 "correctly (get_size)");
          }

          if (a.get_count() != getElementsCount<dims>(range)) {
            FAIL(log,
                 "global_buffer accessor for discard_write is not constructed "
                 "correctly (get_count)");
          }

          if (a.get_offset() != getId<dims>(0)) {
            FAIL(log,
                 "global_buffer accessor for discard_write is not constructed "
                 "correctly (get_offset)");
          }

          if (a.get_range() != range) {
            FAIL(log,
                 "global_buffer accessor for discard_write is not constructed "
                 "correctly (get_range)");
          }

          if (a.is_placeholder() != false) {
            FAIL(log,
                 "global_buffer accessor for discard_write is not constructed "
                 "correctly (is_placeholder)");
          }
        }
        /** check (buffer, handler, range, offset) constructor for discard_write
         * global_buffer
         */
        {
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::discard_write,
                             cl::sycl::access::target::global_buffer,
                             cl::sycl::access::placeholder::false_t>
              a(buffer, h, r, offset);

          if (a.get_size() != getElementsCount<dims>(range) * sizeof(T)) {
            FAIL(log,
                 "global_buffer ranged accessor for discard_write is not "
                 "constructed correctly (get_size)");
          }

          if (a.get_count() != getElementsCount<dims>(range)) {
            FAIL(log,
                 "global_buffer ranged accessor for discard_write is not "
                 "constructed correctly (get_count)");
          }

          if (a.get_offset() != offset) {
            FAIL(log,
                 "global_buffer ranged accessor for discard_write is not "
                 "constructed correctly (get_offset)");
          }

          if (a.get_range() != range) {
            FAIL(log,
                 "global_buffer ranged accessor for discard_write is not "
                 "constructed correctly (get_range)");
          }

          if (a.is_placeholder() != false) {
            FAIL(log,
                 "global_buffer ranged accessor for discard_write is not "
                 "constructed correctly (is_placeholder)");
          }
        }

        /** check (buffer, handler) constructor for discard_read_write
         * global_buffer
         */
        {
          cl::sycl::accessor<T, dims,
                             cl::sycl::access::mode::discard_read_write,
                             cl::sycl::access::target::global_buffer,
                             cl::sycl::access::placeholder::false_t>
              a(buffer, h);

          if (a.get_size() != getElementsCount<dims>(range) * sizeof(T)) {
            FAIL(log,
                 "global_buffer accessor for discard_read_write is not "
                 "constructed correctly (get_size)");
          }

          if (a.get_count() != getElementsCount<dims>(range)) {
            FAIL(log,
                 "global_buffer accessor for discard_read_write is not "
                 "constructed correctly (get_count)");
          }

          if (a.get_offset() != getId<dims>(0)) {
            FAIL(log,
                 "global_buffer accessor for discard_read_write is not "
                 "constructed correctly (get_offset)");
          }

          if (a.get_range() != range) {
            FAIL(log,
                 "global_buffer accessor for discard_read_write is not "
                 "constructed correctly (get_range)");
          }

          if (a.is_placeholder() != false) {
            FAIL(log,
                 "global_buffer accessor for discard_read_write is not "
                 "constructed correctly (is_placeholder)");
          }
        }
        /** check (buffer, handler, range, offset) constructor for
         * discard_read_write global_buffer
         */
        {
          cl::sycl::accessor<T, dims,
                             cl::sycl::access::mode::discard_read_write,
                             cl::sycl::access::target::global_buffer,
                             cl::sycl::access::placeholder::false_t>
              a(buffer, h, r, offset);

          if (a.get_size() != getElementsCount<dims>(range) * sizeof(T)) {
            FAIL(log,
                 "global_buffer ranged accessor for discard_read_write is not "
                 "constructed correctly (get_size)");
          }

          if (a.get_count() != getElementsCount<dims>(range)) {
            FAIL(log,
                 "global_buffer ranged accessor for discard_read_write is not "
                 "constructed correctly (get_count)");
          }

          if (a.get_offset() != offset) {
            FAIL(log,
                 "global_buffer ranged accessor for discard_read_write is not "
                 "constructed correctly (get_offset)");
          }

          if (a.get_range() != range) {
            FAIL(log,
                 "global_buffer ranged accessor for discard_read_write is not "
                 "constructed correctly (get_range)");
          }

          if (a.is_placeholder() != false) {
            FAIL(log,
                 "global_buffer ranged accessor for discard_read_write is not "
                 "constructed correctly (is_placeholder)");
          }
        }

        /** check (buffer, handler) global_buffer accessor constructors for
         *  atomic
         */

        {
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::atomic,
                             cl::sycl::access::target::global_buffer,
                             cl::sycl::access::placeholder::false_t>
              a(buffer, h);

          if (a.get_size() != getElementsCount<dims>(range) * sizeof(T)) {
            FAIL(log,
                 "global_buffer accessor for atomic is not constructed "
                 "correctly (get_size)");
          }

          if (a.get_count() != getElementsCount<dims>(range)) {
            FAIL(log,
                 "global_buffer accessor for atomic is not constructed "
                 "correctly (get_count)");
          }

          if (a.get_offset() != getId<dims>(0)) {
            FAIL(log,
                 "global_buffer accessor for atomic is not constructed "
                 "correctly (get_offset)");
          }

          if (a.get_range() != range) {
            FAIL(log,
                 "global_buffer accessor for atomic is not constructed "
                 "correctly (get_range)");
          }

          if (a.is_placeholder() != false) {
            FAIL(log,
                 "global_buffer accessor for atomic is not constructed "
                 "correctly (is_placeholder)");
          }
        }
        /** check (buffer, handler, range, offset) global_buffer constructor for
         *  atomic
         */
        {
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::atomic,
                             cl::sycl::access::target::global_buffer,
                             cl::sycl::access::placeholder::false_t>
              a(buffer, h, r, offset);

          if (a.get_size() != getElementsCount<dims>(range) * sizeof(T)) {
            FAIL(log,
                 "global_buffer ranged accessor for atomic is not constructed "
                 "correctly (get_size)");
          }

          if (a.get_count() != getElementsCount<dims>(range)) {
            FAIL(log,
                 "global_buffer ranged accessor for atomic is not constructed "
                 "correctly (get_count)");
          }

          if (a.get_offset() != offset) {
            FAIL(log,
                 "global_buffer ranged accessor for atomic is not constructed "
                 "correctly (get_offset)");
          }

          if (a.get_range() != range) {
            FAIL(log,
                 "global_buffer ranged accessor for atomic is not constructed "
                 "correctly (get_range)");
          }

          if (a.is_placeholder() != false) {
            FAIL(log,
                 "global_buffer ranged accessor for atomic is not constructed "
                 "correctly (is_placeholder)");
          }
        }

        /** check accessor is Copy Constructible
         */
        {
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                             cl::sycl::access::target::global_buffer,
                             cl::sycl::access::placeholder::false_t>
              a(buffer, h, r, offset);
          auto b{a};

          if (a.get_size() != b.get_size()) {
            FAIL(log,
                 "global_buffer accessor is not copy constructible (get_size)");
          }

          if (a.get_count() != b.get_count()) {
            FAIL(
                log,
                "global_buffer accessor is not copy constructible (get_count)");
          }

          if (a.get_offset() != b.get_offset()) {
            FAIL(log,
                 "global_buffer accessor is not copy constructible "
                 "(get_offset)");
          }

          if (a.get_range() != b.get_range()) {
            FAIL(
                log,
                "global_buffer accessor is not copy constructible (get_range)");
          }

          if (a.is_placeholder() != b.is_placeholder()) {
            FAIL(log,
                 "global_buffer accessor is not copy constructible "
                 "(is_placeholder)");
          }

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
                             cl::sycl::access::placeholder::false_t>
              a(buffer, h, r, offset);
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                             cl::sycl::access::target::global_buffer,
                             cl::sycl::access::placeholder::false_t>
              b(buffer, h);
          b = a;

          if (a.get_size() != b.get_size()) {
            FAIL(log,
                 "global_buffer accessor is not copy assignable (get_size)");
          }

          if (a.get_count() != b.get_count()) {
            FAIL(log,
                 "global_buffer accessor is not copy assignable (get_count)");
          }

          if (a.get_offset() != b.get_offset()) {
            FAIL(log,
                 "global_buffer accessor is not copy assignable (get_offset)");
          }

          if (a.get_range() != b.get_range()) {
            FAIL(log,
                 "global_buffer accessor is not copy assignable (get_range)");
          }

          if (a.is_placeholder() != b.is_placeholder()) {
            FAIL(log,
                 "global_buffer accessor is not copy assignable "
                 "(is_placeholder)");
          }
        }

        /** check accessor is Move Constructible
         */
        {
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                             cl::sycl::access::target::global_buffer,
                             cl::sycl::access::placeholder::false_t>
              a(buffer, h, r, offset);
          auto b{std::move(a)};

          if (b.get_size() != getElementsCount<dims>(range) * sizeof(T)) {
            FAIL(log,
                 "global_buffer accessor is not move constructible (get_size)");
          }

          if (b.get_count() != getElementsCount<dims>(range)) {
            FAIL(
                log,
                "global_buffer accessor is not move constructible (get_count)");
          }

          if (b.get_offset() != offset) {
            FAIL(log,
                 "global_buffer accessor is not move constructible "
                 "(get_offset)");
          }

          if (b.get_range() != range) {
            FAIL(log,
                 "global_buffer accessor is not move constructible "
                 "(get_range)");
          }

          if (b.is_placeholder() != false) {
            FAIL(log,
                 "global_buffer accessor is not move constructible "
                 "(is_placeholder)");
          }
        }

        /** check accessor is Move Assignable
         */
        {
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                             cl::sycl::access::target::global_buffer,
                             cl::sycl::access::placeholder::false_t>
              a(buffer, h, r, offset);
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                             cl::sycl::access::target::global_buffer,
                             cl::sycl::access::placeholder::false_t>
              b(buffer, h);
          b = std::move(a);

          if (b.get_size() != getElementsCount<dims>(range) * sizeof(T)) {
            FAIL(log,
                 "global_buffer accessor is not move assignable (get_size)");
          }

          if (b.get_count() != getElementsCount<dims>(range)) {
            FAIL(log,
                 "global_buffer accessor is not move assignable (get_count)");
          }

          if (b.get_offset() != offset) {
            FAIL(log,
                 "global_buffer accessor is not move assignable (get_offset)");
          }

          if (b.get_range() != range) {
            FAIL(log,
                 "global_buffer accessor is not move assignable (get_range)");
          }

          if (b.is_placeholder() != false) {
            FAIL(log,
                 "global_buffer accessor is not move assignable "
                 "(is_placeholder)");
          }
        }

        /** dummy kernel as no kernel is required for these checks
         */
        h.single_task(dummy_functor{});
      });
      queue.wait_and_throw();
    }

    /** check buffer accessor constructors for constant_buffer
     */
    {
      queue.submit([&](cl::sycl::handler &h) {
        /** check (buffer, handler) constructor for reading constant_buffer
         */
        {
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                             cl::sycl::access::target::constant_buffer,
                             cl::sycl::access::placeholder::false_t>
              a(buffer, h);

          if (a.get_size() != getElementsCount<dims>(range) * sizeof(T)) {
            FAIL(log,
                 "constant_buffer accessor for read is not constructed "
                 "correctly (get_size)");
          }

          if (a.get_count() != getElementsCount<dims>(range)) {
            FAIL(log,
                 "constant_buffer accessor for read is not constructed "
                 "correctly (get_count)");
          }

          if (a.get_offset() != getId<dims>(0)) {
            FAIL(log,
                 "constant_buffer accessor for read is not constructed "
                 "correctly (get_offset)");
          }

          if (a.get_range() != range) {
            FAIL(log,
                 "constant_buffer accessor for read is not constructed "
                 "correctly (get_range)");
          }

          if (a.is_placeholder() != false) {
            FAIL(log,
                 "constant_buffer accessor for read is not constructed "
                 "correctly (is_placeholder)");
          }
        }
        /** check (buffer, handler, range, offset) constructor for reading
         * constant_buffer
         */
        {
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                             cl::sycl::access::target::constant_buffer,
                             cl::sycl::access::placeholder::false_t>
              a(buffer, h, r, offset);

          if (a.get_size() != getElementsCount<dims>(range) * sizeof(T)) {
            FAIL(log,
                 "constant_buffer ranged accessor for read is not constructed "
                 "correctly (get_size)");
          }

          if (a.get_count() != getElementsCount<dims>(range)) {
            FAIL(log,
                 "constant_buffer ranged accessor for read is not constructed "
                 "correctly (get_count)");
          }

          if (a.get_offset() != offset) {
            FAIL(log,
                 "constant_buffer ranged accessor for read is not constructed "
                 "correctly (get_offset)");
          }

          if (a.get_range() != range) {
            FAIL(log,
                 "constant_buffer ranged accessor for read is not constructed "
                 "correctly (get_range)");
          }

          if (a.is_placeholder() != false) {
            FAIL(log,
                 "constant_buffer ranged accessor for read is not constructed "
                 "correctly (is_placeholder)");
          }
        }

        /** check accessor is Copy Constructible
         */
        {
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                             cl::sycl::access::target::constant_buffer,
                             cl::sycl::access::placeholder::false_t>
              a(buffer, h, r, offset);
          auto b{a};

          if (a.get_size() != b.get_size()) {
            FAIL(log,
                 "constant_buffer accessor is not copy constructible "
                 "(get_size)");
          }

          if (a.get_count() != b.get_count()) {
            FAIL(log,
                 "constant_buffer accessor is not copy constructible "
                 "(get_count)");
          }

          if (a.get_offset() != b.get_offset()) {
            FAIL(log,
                 "constant_buffer accessor is not copy constructible "
                 "(get_offset)");
          }

          if (a.get_range() != b.get_range()) {
            FAIL(log,
                 "constant_buffer accessor is not copy constructible "
                 "(get_range)");
          }

          if (a.is_placeholder() != b.is_placeholder()) {
            FAIL(log,
                 "constant_buffer accessor is not copy constructible "
                 "(is_placeholder)");
          }

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
                             cl::sycl::access::placeholder::false_t>
              a(buffer, h, r, offset);
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                             cl::sycl::access::target::constant_buffer,
                             cl::sycl::access::placeholder::false_t>
              b(buffer, h);
          b = a;

          if (a.get_size() != b.get_size()) {
            FAIL(log,
                 "constant_buffer accessor is not copy assignable (get_size)");
          }

          if (a.get_count() != b.get_count()) {
            FAIL(log,
                 "constant_buffer accessor is not copy assignable (get_count)");
          }

          if (a.get_offset() != b.get_offset()) {
            FAIL(
                log,
                "constant_buffer accessor is not copy assignable (get_offset)");
          }

          if (a.get_range() != b.get_range()) {
            FAIL(log,
                 "constant_buffer accessor is not copy assignable (get_range)");
          }

          if (a.is_placeholder() != b.is_placeholder()) {
            FAIL(log,
                 "constant_buffer accessor is not copy assignable "
                 "(is_placeholder)");
          }
        }

        /** check accessor is Move Constructible
         */
        {
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                             cl::sycl::access::target::constant_buffer,
                             cl::sycl::access::placeholder::false_t>
              a(buffer, h, r, offset);
          auto b{std::move(a)};

          if (b.get_size() != getElementsCount<dims>(range) * sizeof(T)) {
            FAIL(log,
                 "constant_buffer accessor is not move constructible "
                 "(get_size)");
          }

          if (b.get_count() != getElementsCount<dims>(range)) {
            FAIL(log,
                 "constant_buffer accessor is not move constructible "
                 "(get_count)");
          }

          if (b.get_offset() != offset) {
            FAIL(log,
                 "constant_buffer accessor is not move constructible "
                 "(get_offset)");
          }

          if (b.get_range() != range) {
            FAIL(log,
                 "constant_buffer accessor is not move constructible "
                 "(get_range)");
          }

          if (b.is_placeholder() != false) {
            FAIL(log,
                 "constant_buffer accessor is not move constructible "
                 "(is_placeholder)");
          }
        }

        /** check accessor is Move Assignable
         */
        {
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                             cl::sycl::access::target::constant_buffer,
                             cl::sycl::access::placeholder::false_t>
              a(buffer, h, r, offset);
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                             cl::sycl::access::target::constant_buffer,
                             cl::sycl::access::placeholder::false_t>
              b(buffer, h);
          b = std::move(a);

          if (b.get_size() != getElementsCount<dims>(range) * sizeof(T)) {
            FAIL(log,
                 "constant_buffer accessor is not move assignable (get_size)");
          }

          if (b.get_count() != getElementsCount<dims>(range)) {
            FAIL(log,
                 "constant_buffer accessor is not move assignable (get_count)");
          }

          if (b.get_offset() != offset) {
            FAIL(
                log,
                "constant_buffer accessor is not move assignable (get_offset)");
          }

          if (b.get_range() != range) {
            FAIL(log,
                 "constant_buffer accessor is not move assignable (get_range)");
          }

          if (b.is_placeholder() != false) {
            FAIL(log,
                 "constant_buffer accessor is not move assignable "
                 "(is_placeholder)");
          }
        }

        /** dummy kernel as no kernel is required for these checks
         */
        h.single_task(dummy_functor{});

      });
      queue.wait_and_throw();
    }

    /** check buffer accessor constructors for host_buffer
     */
    {
      /** check (buffer) constructor for reading host_buffer
       */
      {
        cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                           cl::sycl::access::target::host_buffer,
                           cl::sycl::access::placeholder::false_t>
            a(buffer);

        if (a.get_size() != getElementsCount<dims>(range) * sizeof(T)) {
          FAIL(log,
               "host_buffer accessor for read is not constructed correctly "
               "(get_size)");
        }

        if (a.get_count() != getElementsCount<dims>(range)) {
          FAIL(log,
               "host_buffer accessor for read is not constructed correctly "
               "(get_count)");
        }

        if (a.get_offset() != getId<dims>(0)) {
          FAIL(log,
               "host_buffer accessor for read is not constructed correctly "
               "(get_offset)");
        }

        if (a.get_range() != range) {
          FAIL(log,
               "host_buffer accessor for read is not constructed correctly "
               "(get_range)");
        }

        if (a.is_placeholder() != false) {
          FAIL(log,
               "host_buffer accessor for read is not constructed correctly "
               "(is_placeholder)");
        }
      }
      /** check (buffer, range, offset) constructor for reading
       * host_buffer
       */
      {
        cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                           cl::sycl::access::target::host_buffer,
                           cl::sycl::access::placeholder::false_t>
            a(buffer, r, offset);

        if (a.get_size() != getElementsCount<dims>(range) * sizeof(T)) {
          FAIL(log,
               "host_buffer ranged accessor for read is not constructed "
               "correctly (get_size)");
        }

        if (a.get_count() != getElementsCount<dims>(range)) {
          FAIL(log,
               "host_buffer ranged accessor for read is not constructed "
               "correctly (get_count)");
        }

        if (a.get_offset() != offset) {
          FAIL(log,
               "host_buffer ranged accessor for read is not constructed "
               "correctly (get_offset)");
        }

        if (a.get_range() != range) {
          FAIL(log,
               "host_buffer ranged accessor for read is not constructed "
               "correctly (get_range)");
        }

        if (a.is_placeholder() != false) {
          FAIL(log,
               "host_buffer ranged accessor for read is not constructed "
               "correctly (is_placeholder)");
        }
      }

      /** check (buffer) constructor for writing host_buffer
       */
      {
        cl::sycl::accessor<T, dims, cl::sycl::access::mode::write,
                           cl::sycl::access::target::host_buffer,
                           cl::sycl::access::placeholder::false_t>
            a(buffer);

        if (a.get_size() != getElementsCount<dims>(range) * sizeof(T)) {
          FAIL(log,
               "host_buffer accessor for write is not constructed correctly "
               "(get_size)");
        }

        if (a.get_count() != getElementsCount<dims>(range)) {
          FAIL(log,
               "host_buffer accessor for write is not constructed correctly "
               "(get_count)");
        }

        if (a.get_offset() != getId<dims>(0)) {
          FAIL(log,
               "host_buffer accessor for write is not constructed correctly "
               "(get_offset)");
        }

        if (a.get_range() != range) {
          FAIL(log,
               "host_buffer accessor for write is not constructed correctly "
               "(get_range)");
        }

        if (a.is_placeholder() != false) {
          FAIL(log,
               "host_buffer accessor for write is not constructed correctly "
               "(is_placeholder)");
        }
      }
      /** check (buffer, range, offset) constructor for writing
       * host_buffer
       */
      {
        cl::sycl::accessor<T, dims, cl::sycl::access::mode::write,
                           cl::sycl::access::target::host_buffer,
                           cl::sycl::access::placeholder::false_t>
            a(buffer, r, offset);

        if (a.get_size() != getElementsCount<dims>(range) * sizeof(T)) {
          FAIL(log,
               "host_buffer ranged accessor for write is not constructed "
               "correctly (get_size)");
        }

        if (a.get_count() != getElementsCount<dims>(range)) {
          FAIL(log,
               "host_buffer ranged accessor for write is not constructed "
               "correctly (get_count)");
        }

        if (a.get_offset() != offset) {
          FAIL(log,
               "host_buffer ranged accessor for write is not constructed "
               "correctly (get_offset)");
        }

        if (a.get_range() != range) {
          FAIL(log,
               "host_buffer ranged accessor for write is not constructed "
               "correctly (get_range)");
        }

        if (a.is_placeholder() != false) {
          FAIL(log,
               "host_buffer ranged accessor for write is not constructed "
               "correctly (is_placeholder)");
        }
      }

      /** check (buffer) constructor for read_write host_buffer
       */
      {
        cl::sycl::accessor<T, dims, cl::sycl::access::mode::read_write,
                           cl::sycl::access::target::host_buffer,
                           cl::sycl::access::placeholder::false_t>
            a(buffer);

        if (a.get_size() != getElementsCount<dims>(range) * sizeof(T)) {
          FAIL(log,
               "host_buffer accessor for read_write is not constructed "
               "correctly (get_size)");
        }

        if (a.get_count() != getElementsCount<dims>(range)) {
          FAIL(log,
               "host_buffer accessor for read_write is not constructed "
               "correctly (get_count)");
        }

        if (a.get_offset() != getId<dims>(0)) {
          FAIL(log,
               "host_buffer accessor for read_write is not constructed "
               "correctly (get_offset)");
        }

        if (a.get_range() != range) {
          FAIL(log,
               "host_buffer accessor for read_write is not constructed "
               "correctly (get_range)");
        }

        if (a.is_placeholder() != false) {
          FAIL(log,
               "host_buffer accessor for read_write is not constructed "
               "correctly (is_placeholder)");
        }
      }
      /** check (buffer, range, offset) constructor for read_write
       * host_buffer
       */
      {
        cl::sycl::accessor<T, dims, cl::sycl::access::mode::read_write,
                           cl::sycl::access::target::host_buffer,
                           cl::sycl::access::placeholder::false_t>
            a(buffer, r, offset);

        if (a.get_size() != getElementsCount<dims>(range) * sizeof(T)) {
          FAIL(log,
               "host_buffer ranged accessor for read_write is not constructed "
               "correctly (get_size)");
        }

        if (a.get_count() != getElementsCount<dims>(range)) {
          FAIL(log,
               "host_buffer ranged accessor for read_write is not constructed "
               "correctly (get_count)");
        }

        if (a.get_offset() != offset) {
          FAIL(log,
               "host_buffer ranged accessor for read_write is not constructed "
               "correctly (get_offset)");
        }

        if (a.get_range() != range) {
          FAIL(log,
               "host_buffer ranged accessor for read_write is not constructed "
               "correctly (get_range)");
        }

        if (a.is_placeholder() != false) {
          FAIL(log,
               "host_buffer ranged accessor for read_write is not constructed "
               "correctly (is_placeholder)");
        }
      }

      /** check (buffer) constructor for discard_write host_buffer
       */
      {
        cl::sycl::accessor<T, dims, cl::sycl::access::mode::discard_write,
                           cl::sycl::access::target::host_buffer,
                           cl::sycl::access::placeholder::false_t>
            a(buffer);

        if (a.get_size() != getElementsCount<dims>(range) * sizeof(T)) {
          FAIL(log,
               "host_buffer accessor for discard_write is not constructed "
               "correctly (get_size)");
        }

        if (a.get_count() != getElementsCount<dims>(range)) {
          FAIL(log,
               "host_buffer accessor for discard_write is not constructed "
               "correctly (get_count)");
        }

        if (a.get_offset() != getId<dims>(0)) {
          FAIL(log,
               "host_buffer accessor for discard_write is not constructed "
               "correctly (get_offset)");
        }

        if (a.get_range() != range) {
          FAIL(log,
               "host_buffer accessor for discard_write is not constructed "
               "correctly (get_range)");
        }

        if (a.is_placeholder() != false) {
          FAIL(log,
               "host_buffer accessor fordiscard_write is not constructed "
               "correctly (is_placeholder)");
        }
      }
      /** check (buffer, range, offset) constructor for discard_write
       * host_buffer
       */
      {
        cl::sycl::accessor<T, dims, cl::sycl::access::mode::discard_write,
                           cl::sycl::access::target::host_buffer,
                           cl::sycl::access::placeholder::false_t>
            a(buffer, r, offset);

        if (a.get_size() != getElementsCount<dims>(range) * sizeof(T)) {
          FAIL(log,
               "host_buffer ranged accessor for discard_write is not "
               "constructed correctly (get_size)");
        }

        if (a.get_count() != getElementsCount<dims>(range)) {
          FAIL(log,
               "host_buffer ranged accessor for discard_write is not "
               "constructed correctly (get_count)");
        }

        if (a.get_offset() != offset) {
          FAIL(log,
               "host_buffer ranged accessor for discard_write is not "
               "constructed correctly (get_offset)");
        }

        if (a.get_range() != range) {
          FAIL(log,
               "host_buffer ranged accessor for discard_write is not "
               "constructed correctly (get_range)");
        }

        if (a.is_placeholder() != false) {
          FAIL(log,
               "host_buffer ranged accessor for discard_write is not "
               "constructed correctly (is_placeholder)");
        }
      }

      /** check (buffer) constructor for discard_read_write host_buffer
       */
      {
        cl::sycl::accessor<T, dims, cl::sycl::access::mode::discard_read_write,
                           cl::sycl::access::target::host_buffer,
                           cl::sycl::access::placeholder::false_t>
            a(buffer);

        if (a.get_size() != getElementsCount<dims>(range) * sizeof(T)) {
          FAIL(log,
               "host_buffer accessor for discard_read_write is not constructed "
               "correctly (get_size)");
        }

        if (a.get_count() != getElementsCount<dims>(range)) {
          FAIL(log,
               "host_buffer accessor for discard_read_write is not constructed "
               "correctly (get_count)");
        }

        if (a.get_offset() != getId<dims>(0)) {
          FAIL(log,
               "host_buffer accessor for discard_read_write is not constructed "
               "correctly (get_offset)");
        }

        if (a.get_range() != range) {
          FAIL(log,
               "host_buffer accessor for discard_read_write is not constructed "
               "correctly (get_range)");
        }

        if (a.is_placeholder() != false) {
          FAIL(log,
               "host_buffer accessor for discard_read_write is not constructed "
               "correctly (is_placeholder)");
        }
      }
      /** check (buffer, range, offset) constructor for
       * discard_read_write host_buffer
       */
      {
        cl::sycl::accessor<T, dims, cl::sycl::access::mode::discard_read_write,
                           cl::sycl::access::target::host_buffer,
                           cl::sycl::access::placeholder::false_t>
            a(buffer, r, offset);

        if (a.get_size() != getElementsCount<dims>(range) * sizeof(T)) {
          FAIL(log,
               "host_buffer ranged accessor for discard_read_write is not "
               "constructed correctly (get_size)");
        }

        if (a.get_count() != getElementsCount<dims>(range)) {
          FAIL(log,
               "host_buffer ranged accessor for discard_read_write is not "
               "constructed correctly (get_count)");
        }

        if (a.get_offset() != offset) {
          FAIL(log,
               "host_buffer ranged accessor for discard_read_write is not "
               "constructed correctly (get_offset)");
        }

        if (a.get_range() != range) {
          FAIL(log,
               "host_buffer ranged accessor for discard_read_write is not "
               "constructed correctly (get_range)");
        }

        if (a.is_placeholder() != false) {
          FAIL(log,
               "host_buffer ranged accessor for discard_read_write is not "
               "constructed correctly (is_placeholder)");
        }
      }

      /** check accessor is Copy Constructible
       */
      {
        cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                           cl::sycl::access::target::host_buffer,
                           cl::sycl::access::placeholder::false_t>
            a(buffer, r, offset);
        auto b{a};

        if (a.get_size() != b.get_size()) {
          FAIL(log,
               "host_buffer accessor is not copy constructible (get_size)");
        }

        if (a.get_count() != b.get_count()) {
          FAIL(log,
               "host_buffer accessor is not copy constructible (get_count)");
        }

        if (a.get_offset() != b.get_offset()) {
          FAIL(log,
               "host_buffer accessor is not copy constructible (get_offset)");
        }

        if (a.get_range() != b.get_range()) {
          FAIL(log,
               "host_buffer accessor is not copy constructible (get_range)");
        }

        if (a.is_placeholder() != b.is_placeholder()) {
          FAIL(log,
               "host_buffer accessor is not copy constructible "
               "(is_placeholder)");
        }

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
                           cl::sycl::access::target::host_buffer,
                           cl::sycl::access::placeholder::false_t>
            a(buffer, r, offset);
        cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                           cl::sycl::access::target::host_buffer,
                           cl::sycl::access::placeholder::false_t>
            b(buffer);
        b = a;

        if (a.get_size() != b.get_size()) {
          FAIL(log, "host_buffer accessor is not copy assignable (get_size)");
        }

        if (a.get_count() != b.get_count()) {
          FAIL(log, "host_buffer accessor is not copy assignable (get_count)");
        }

        if (a.get_offset() != b.get_offset()) {
          FAIL(log, "host_buffer accessor is not copy assignable (get_offset)");
        }

        if (a.get_range() != b.get_range()) {
          FAIL(log, "host_buffer accessor is not copy assignable (get_range)");
        }

        if (a.is_placeholder() != b.is_placeholder()) {
          FAIL(log,
               "host_buffer accessor is not copy assignable (is_placeholder)");
        }
      }

      /** check accessor is Move Constructible
       */
      {
        cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                           cl::sycl::access::target::host_buffer,
                           cl::sycl::access::placeholder::false_t>
            a(buffer, r, offset);
        auto b{std::move(a)};

        if (b.get_size() != getElementsCount<dims>(range) * sizeof(T)) {
          FAIL(log,
               "host_buffer accessor is not move constructible (get_size)");
        }

        if (b.get_count() != getElementsCount<dims>(range)) {
          FAIL(log,
               "host_buffer accessor is not move constructible (get_count)");
        }

        if (b.get_offset() != offset) {
          FAIL(log,
               "host_buffer accessor is not move constructible (get_offset)");
        }

        if (b.get_range() != range) {
          FAIL(log,
               "host_buffer accessor is not move constructible (get_range)");
        }

        if (b.is_placeholder() != false) {
          FAIL(log,
               "host_buffer accessor is not move constructible "
               "(is_placeholder)");
        }
      }

      /** check accessor is Move Assignable
       */
      {
        cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                           cl::sycl::access::target::host_buffer,
                           cl::sycl::access::placeholder::false_t>
            a(buffer, r, offset);
        cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                           cl::sycl::access::target::host_buffer,
                           cl::sycl::access::placeholder::false_t>
            b(buffer);
        b = std::move(a);

        if (b.get_size() != getElementsCount<dims>(range) * sizeof(T)) {
          FAIL(log, "host_buffer accessor is not move assignable (get_size)");
        }

        if (b.get_count() != getElementsCount<dims>(range)) {
          FAIL(log, "host_buffer accessor is not move assignable (get_count)");
        }

        if (b.get_offset() != offset) {
          FAIL(log, "host_buffer accessor is not move assignable (get_offset)");
        }

        if (b.get_range() != range) {
          FAIL(log, "host_buffer accessor is not move assignable (get_range)");
        }

        if (b.is_placeholder() != false) {
          FAIL(log,
               "host_buffer accessor is not move assignable (is_placeholder)");
        }
      }
    }
  }
};

template <typename T>
class buffer_accessor_dims<T, 0> {
 public:
  static void check(util::logger &log, cl::sycl::queue &queue) {
    cl::sycl::range<1> range = getRange<1>(1);
    std::vector<uint8_t> data(sizeof(T), 0);
    cl::sycl::buffer<T, 1> buffer(reinterpret_cast<T *>(data.data()), range);

    /** check buffer accessor constructors for global_buffer
     */
    {
      queue.submit([&](cl::sycl::handler &h) {
        /** check (buffer) constructor for read global_buffer
         */
        {
          cl::sycl::accessor<T, 0, cl::sycl::access::mode::read,
                             cl::sycl::access::target::global_buffer,
                             cl::sycl::access::placeholder::false_t>
              a(buffer, h);
          if (a.get_size() != sizeof(T)) {
            FAIL(log,
                 "global_buffer accessor for read is not constructed correctly "
                 "(get_size)");
          }

          if (a.get_count() != 1) {
            FAIL(log,
                 "global_buffer accessor for read is not constructed correctly "
                 "(get_count)");
          }

          if (a.is_placeholder() != false) {
            FAIL(log,
                 "global_buffer accessor for read is not constructed correctly "
                 "(is_placeholder)");
          }
        }

        /** check (buffer) constructor for write global_buffer
         */
        {
          cl::sycl::accessor<T, 0, cl::sycl::access::mode::write,
                             cl::sycl::access::target::global_buffer,
                             cl::sycl::access::placeholder::false_t>
              a(buffer, h);
          if (a.get_size() != sizeof(T)) {
            FAIL(log,
                 "global_buffer accessor for write is not constructed "
                 "correctly (get_size)");
          }

          if (a.get_count() != 1) {
            FAIL(log,
                 "global_buffer accessor for write is not constructed "
                 "correctly (get_count)");
          }

          if (a.is_placeholder() != false) {
            FAIL(log,
                 "global_buffer accessor for write is not constructed "
                 "correctly (is_placeholder)");
          }
        }

        /** check (buffer) constructor for read_write global_buffer
         */
        {
          cl::sycl::accessor<T, 0, cl::sycl::access::mode::read_write,
                             cl::sycl::access::target::global_buffer,
                             cl::sycl::access::placeholder::false_t>
              a(buffer, h);
          if (a.get_size() != sizeof(T)) {
            FAIL(log,
                 "global_buffer accessor for read_write is not constructed "
                 "correctly (get_size)");
          }

          if (a.get_count() != 1) {
            FAIL(log,
                 "global_buffer accessor for read_write is not constructed "
                 "correctly (get_count)");
          }

          if (a.is_placeholder() != false) {
            FAIL(log,
                 "global_buffer accessor for read_write is not constructed "
                 "correctly (is_placeholder)");
          }
        }

        /** check (buffer) constructor for discard_write global_buffer
         */
        {
          cl::sycl::accessor<T, 0, cl::sycl::access::mode::discard_write,
                             cl::sycl::access::target::global_buffer,
                             cl::sycl::access::placeholder::false_t>
              a(buffer, h);
          if (a.get_size() != sizeof(T)) {
            FAIL(log,
                 "global_buffer accessor for discard_write is not constructed "
                 "correctly (get_size)");
          }

          if (a.get_count() != 1) {
            FAIL(log,
                 "global_buffer accessor for discard_write is not constructed "
                 "correctly (get_count)");
          }

          if (a.is_placeholder() != false) {
            FAIL(log,
                 "global_buffer accessor for discard_write is not constructed "
                 "correctly (is_placeholder)");
          }
        }

        /** check (buffer) constructor for discard_read_write global_buffer
         */
        {
          cl::sycl::accessor<T, 0, cl::sycl::access::mode::discard_read_write,
                             cl::sycl::access::target::global_buffer,
                             cl::sycl::access::placeholder::false_t>
              a(buffer, h);
          if (a.get_size() != sizeof(T)) {
            FAIL(log,
                 "global_buffer accessor for discard_read_write is not "
                 "constructed correctly (get_size)");
          }

          if (a.get_count() != 1) {
            FAIL(log,
                 "global_buffer accessor for discard_read_write is not "
                 "constructed correctly (get_count)");
          }

          if (a.is_placeholder() != false) {
            FAIL(log,
                 "global_buffer accessor for discard_read_write is not "
                 "constructed correctly (is_placeholder)");
          }
        }

        /** check (buffer) constructor for atomic global_buffer
         */
        {
          cl::sycl::accessor<T, 0, cl::sycl::access::mode::atomic,
                             cl::sycl::access::target::global_buffer,
                             cl::sycl::access::placeholder::false_t>
              a(buffer, h);
          if (a.get_size() != sizeof(T)) {
            FAIL(log,
                 "global_buffer accessor for atomic is not constructed "
                 "correctly (get_size)");
          }

          if (a.get_count() != 1) {
            FAIL(log,
                 "global_buffer accessor for atomic is not constructed "
                 "correctly (get_count)");
          }

          if (a.is_placeholder() != false) {
            FAIL(log,
                 "global_buffer accessor for atomic is not constructed "
                 "correctly (is_placeholder)");
          }
        }

        /** check accessor is Copy Constructible
         */
        {
          cl::sycl::accessor<T, 0, cl::sycl::access::mode::read,
                             cl::sycl::access::target::global_buffer,
                             cl::sycl::access::placeholder::false_t>
              a(buffer, h);
          auto b{a};

          if (a.get_size() != b.get_size()) {
            FAIL(log,
                 "global_buffer accessor is not copy constructible (get_size)");
          }

          if (a.get_count() != b.get_count()) {
            FAIL(
                log,
                "global_buffer accessor is not copy constructible (get_count)");
          }

          if (a.is_placeholder() != b.is_placeholder()) {
            FAIL(log,
                 "global_buffer accessor is not copy constructible "
                 "(is_placeholder)");
          }

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
                             cl::sycl::access::placeholder::false_t>
              a(buffer, h);

          cl::sycl::accessor<T, 0, cl::sycl::access::mode::read,
                             cl::sycl::access::target::global_buffer,
                             cl::sycl::access::placeholder::false_t>
              b(buffer, h);
          b = a;

          if (a.get_size() != b.get_size()) {
            FAIL(log,
                 "global_buffer accessor is not copy assignable (get_size)");
          }

          if (a.get_count() != b.get_count()) {
            FAIL(log,
                 "global_buffer accessor is not copy assignable (get_count)");
          }

          if (a.is_placeholder() != false) {
            FAIL(log,
                 "global_buffer accessor is not copy assignable "
                 "(is_placeholder)");
          }
        }

        /** check accessor is Move Constructible
         */
        {
          cl::sycl::accessor<T, 0, cl::sycl::access::mode::read,
                             cl::sycl::access::target::global_buffer,
                             cl::sycl::access::placeholder::false_t>
              a(buffer, h);
          auto b{std::move(a)};

          if (b.get_size() != sizeof(T)) {
            FAIL(log,
                 "global_buffer accessor is not move constructible (get_size)");
          }

          if (b.get_count() != 1) {
            FAIL(
                log,
                "global_buffer accessor is not move constructible (get_count)");
          }

          if (b.is_placeholder() != false) {
            FAIL(log,
                 "global_buffer accessor is not move constructible "
                 "(is_placeholder)");
          }
        }

        /** check accessor is Move Assignable
         */
        {
          cl::sycl::accessor<T, 0, cl::sycl::access::mode::read,
                             cl::sycl::access::target::global_buffer,
                             cl::sycl::access::placeholder::false_t>
              a(buffer, h);

          cl::sycl::accessor<T, 0, cl::sycl::access::mode::read,
                             cl::sycl::access::target::global_buffer,
                             cl::sycl::access::placeholder::false_t>
              b(buffer, h);
          b = std::move(a);

          if (b.get_size() != sizeof(T)) {
            FAIL(log,
                 "global_buffer accessor is not move constructible (get_size)");
          }

          if (b.get_count() != 1) {
            FAIL(
                log,
                "global_buffer accessor is not move constructible (get_count)");
          }

          if (b.is_placeholder() != false) {
            FAIL(log,
                 "global_buffer accessor is not move constructible "
                 "(is_placeholder)");
          }
        }

        /** dummy kernel as no kernel is required for these checks
         */
        h.single_task(dummy_functor{});

      });
      queue.wait_and_throw();
    }

    /** check buffer accessor constructors for constant_buffer
     */
    {
      queue.submit([&](cl::sycl::handler &h) {
        /** check (buffer) constructor for read constant_buffer
         */
        {
          cl::sycl::accessor<T, 0, cl::sycl::access::mode::read,
                             cl::sycl::access::target::constant_buffer,
                             cl::sycl::access::placeholder::false_t>
              a(buffer, h);
          if (a.get_size() != sizeof(T)) {
            FAIL(log,
                 "constant_buffer accessor for read is not constructed "
                 "correctly (get_size)");
          }

          if (a.get_count() != 1) {
            FAIL(log,
                 "constant_buffer accessor for read is not constructed "
                 "correctly (get_count)");
          }

          if (a.is_placeholder() != false) {
            FAIL(log,
                 "constant_buffer accessor for read is not constructed "
                 "correctly (is_placeholder)");
          }
        }

        /** check accessor is Copy Constructible
         */
        {
          cl::sycl::accessor<T, 0, cl::sycl::access::mode::read,
                             cl::sycl::access::target::constant_buffer,
                             cl::sycl::access::placeholder::false_t>
              a(buffer, h);
          auto b{a};

          if (a.get_size() != b.get_size()) {
            FAIL(log,
                 "constant_buffer accessor is not copy constructible "
                 "(get_size)");
          }

          if (a.get_count() != b.get_count()) {
            FAIL(log,
                 "constant_buffer accessor is not copy constructible "
                 "(get_count)");
          }

          if (a.is_placeholder() != b.is_placeholder()) {
            FAIL(log,
                 "constant_buffer accessor is not copy constructible "
                 "(is_placeholder)");
          }

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
                             cl::sycl::access::placeholder::false_t>
              a(buffer, h);

          cl::sycl::accessor<T, 0, cl::sycl::access::mode::read,
                             cl::sycl::access::target::constant_buffer,
                             cl::sycl::access::placeholder::false_t>
              b(buffer, h);
          b = a;

          if (a.get_size() != b.get_size()) {
            FAIL(log,
                 "constant_buffer accessor is not copy assignable (get_size)");
          }

          if (a.get_count() != b.get_count()) {
            FAIL(log,
                 "constant_buffer accessor is not copy assignable (get_count)");
          }

          if (a.is_placeholder() != false) {
            FAIL(log,
                 "constant_buffer accessor is not copy assignable "
                 "(is_placeholder)");
          }
        }

        /** check accessor is Move Constructible
         */
        {
          cl::sycl::accessor<T, 0, cl::sycl::access::mode::read,
                             cl::sycl::access::target::constant_buffer,
                             cl::sycl::access::placeholder::false_t>
              a(buffer, h);
          auto b{std::move(a)};

          if (b.get_size() != sizeof(T)) {
            FAIL(log,
                 "constant_buffer accessor is not move constructible "
                 "(get_size)");
          }

          if (b.get_count() != 1) {
            FAIL(log,
                 "constant_buffer accessor is not move constructible "
                 "(get_count)");
          }

          if (b.is_placeholder() != false) {
            FAIL(log,
                 "constant_buffer accessor is not move constructible "
                 "(is_placeholder)");
          }
        }

        /** check accessor is Move Assignable
         */
        {
          cl::sycl::accessor<T, 0, cl::sycl::access::mode::read,
                             cl::sycl::access::target::constant_buffer,
                             cl::sycl::access::placeholder::false_t>
              a(buffer, h);

          cl::sycl::accessor<T, 0, cl::sycl::access::mode::read,
                             cl::sycl::access::target::constant_buffer,
                             cl::sycl::access::placeholder::false_t>
              b(buffer, h);
          b = std::move(a);

          if (b.get_size() != sizeof(T)) {
            FAIL(log,
                 "constant_buffer accessor is not move constructible "
                 "(get_size)");
          }

          if (b.get_count() != 1) {
            FAIL(log,
                 "constant_buffer accessor is not move constructible "
                 "(get_count)");
          }

          if (b.is_placeholder() != false) {
            FAIL(log,
                 "constant_buffer accessor is not move constructible "
                 "(is_placeholder)");
          }
        }

        /** dummy kernel as no kernel is required for these checks
         */
        h.single_task(dummy_functor{});

      });
      queue.wait_and_throw();
    }

    /** check buffer accessor constructors for host_buffer
     */
    {
      /** check (buffer) constructor for read host_buffer
       */
      {
        cl::sycl::accessor<T, 0, cl::sycl::access::mode::read,
                           cl::sycl::access::target::host_buffer,
                           cl::sycl::access::placeholder::false_t>
            a(buffer);
        if (a.get_size() != sizeof(T)) {
          FAIL(log,
               "host_buffer accessor for read is not constructed correctly "
               "(get_size)");
        }

        if (a.get_count() != 1) {
          FAIL(log,
               "host_buffer accessor for read is not constructed correctly "
               "(get_count)");
        }

        if (a.is_placeholder() != false) {
          FAIL(log,
               "host_buffer accessor for read is not constructed correctly "
               "(is_placeholder)");
        }
      }

      /** check (buffer) constructor for write host_buffer
       */
      {
        cl::sycl::accessor<T, 0, cl::sycl::access::mode::write,
                           cl::sycl::access::target::host_buffer,
                           cl::sycl::access::placeholder::false_t>
            a(buffer);
        if (a.get_size() != sizeof(T)) {
          FAIL(log,
               "host_buffer accessor for write is not constructed correctly "
               "(get_size)");
        }

        if (a.get_count() != 1) {
          FAIL(log,
               "host_buffer accessor for write is not constructed correctly "
               "(get_count)");
        }

        if (a.is_placeholder() != false) {
          FAIL(log,
               "host_buffer accessor for write is not constructed correctly "
               "(is_placeholder)");
        }
      }

      /** check (buffer) constructor for read_write host_buffer
       */
      {
        cl::sycl::accessor<T, 0, cl::sycl::access::mode::read_write,
                           cl::sycl::access::target::host_buffer,
                           cl::sycl::access::placeholder::false_t>
            a(buffer);
        if (a.get_size() != sizeof(T)) {
          FAIL(log,
               "host_buffer accessor for read_write is not constructed "
               "correctly (get_size)");
        }

        if (a.get_count() != 1) {
          FAIL(log,
               "host_buffer accessor for read_write is not constructed "
               "correctly (get_count)");
        }

        if (a.is_placeholder() != false) {
          FAIL(log,
               "host_buffer accessor for read_write is not constructed "
               "correctly (is_placeholder)");
        }
      }

      /** check (buffer) constructor for discard_write host_buffer
       */
      {
        cl::sycl::accessor<T, 0, cl::sycl::access::mode::discard_write,
                           cl::sycl::access::target::host_buffer,
                           cl::sycl::access::placeholder::false_t>
            a(buffer);
        if (a.get_size() != sizeof(T)) {
          FAIL(log,
               "host_buffer accessor for discard_write is not constructed "
               "correctly (get_size)");
        }

        if (a.get_count() != 1) {
          FAIL(log,
               "host_buffer accessor for discard_write is not constructed "
               "correctly (get_count)");
        }

        if (a.is_placeholder() != false) {
          FAIL(log,
               "host_buffer accessor for discard_write is not constructed "
               "correctly (is_placeholder)");
        }
      }

      /** check (buffer) constructor for discard_read_write host_buffer
       */
      {
        cl::sycl::accessor<T, 0, cl::sycl::access::mode::discard_read_write,
                           cl::sycl::access::target::host_buffer,
                           cl::sycl::access::placeholder::false_t>
            a(buffer);
        if (a.get_size() != sizeof(T)) {
          FAIL(log,
               "host_buffer accessor for discard_read_write is not "
               "constructed correctly (get_size)");
        }

        if (a.get_count() != 1) {
          FAIL(log,
               "host_buffer accessor for discard_read_write is not "
               "constructed correctly (get_count)");
        }

        if (a.is_placeholder() != false) {
          FAIL(log,
               "host_buffer accessor for discard_read_write is not "
               "constructed correctly (is_placeholder)");
        }
      }

      /** check accessor is Copy Constructible
       */
      {
        cl::sycl::accessor<T, 0, cl::sycl::access::mode::read,
                           cl::sycl::access::target::host_buffer,
                           cl::sycl::access::placeholder::false_t>
            a(buffer);
        auto b{a};

        if (a.get_size() != b.get_size()) {
          FAIL(log,
               "global_buffer accessor is not copy constructible (get_size)");
        }

        if (a.get_count() != b.get_count()) {
          FAIL(log,
               "global_buffer accessor is not copy constructible (get_count)");
        }

        if (a.is_placeholder() != b.is_placeholder()) {
          FAIL(log,
               "host_buffer accessor is not copy constructible "
               "(is_placeholder)");
        }

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
                           cl::sycl::access::target::host_buffer,
                           cl::sycl::access::placeholder::false_t>
            a(buffer);

        cl::sycl::accessor<T, 0, cl::sycl::access::mode::read,
                           cl::sycl::access::target::host_buffer,
                           cl::sycl::access::placeholder::false_t>
            b(buffer);
        b = a;

        if (a.get_size() != b.get_size()) {
          FAIL(log, "global_buffer accessor is not copy assignable (get_size)");
        }

        if (a.get_count() != b.get_count()) {
          FAIL(log,
               "global_buffer accessor is not copy assignable (get_count)");
        }

        if (a.is_placeholder() != false) {
          FAIL(log,
               "host_buffer accessor is not copy assignable (is_placeholder)");
        }
      }

      /** check accessor is Move Constructible
       */
      {
        cl::sycl::accessor<T, 0, cl::sycl::access::mode::read,
                           cl::sycl::access::target::host_buffer,
                           cl::sycl::access::placeholder::false_t>
            a(buffer);
        auto b{std::move(a)};

        if (b.get_size() != sizeof(T)) {
          FAIL(log,
               "global_buffer accessor is not move constructible (get_size)");
        }

        if (b.get_count() != 1) {
          FAIL(log,
               "global_buffer accessor is not move constructible (get_count)");
        }

        if (b.is_placeholder() != false) {
          FAIL(log,
               "host_buffer accessor is not move constructible "
               "(is_placeholder)");
        }
      }

      /** check accessor is Move Assignable
       */
      {
        cl::sycl::accessor<T, 0, cl::sycl::access::mode::read,
                           cl::sycl::access::target::host_buffer,
                           cl::sycl::access::placeholder::false_t>
            a(buffer);

        cl::sycl::accessor<T, 0, cl::sycl::access::mode::read,
                           cl::sycl::access::target::host_buffer,
                           cl::sycl::access::placeholder::false_t>
            b(buffer);
        b = std::move(a);

        if (b.get_size() != sizeof(T)) {
          FAIL(log,
               "global_buffer accessor is not move constructible (get_size)");
        }

        if (b.get_count() != 1) {
          FAIL(log,
               "global_buffer accessor is not move constructible (get_count)");
        }

        if (b.is_placeholder() != false) {
          FAIL(log,
               "host_buffer accessor is not move constructible "
               "(is_placeholder)");
        }
      }
    }
  }
};

struct user_struct {
  float a;
  int b;
  char c;
};

/** tests the constructors for cl::sycl::accessor
 */
class TEST_NAME : public util::test_base {
 public:
  /** return information about this test
   */
  void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  template <typename T>
  void check_all_dims(util::logger &log, cl::sycl::queue &queue) {
    buffer_accessor_dims<T, 0>::check(log, queue);
    buffer_accessor_dims<T, 1>::check(log, queue);
    buffer_accessor_dims<T, 2>::check(log, queue);
    buffer_accessor_dims<T, 3>::check(log, queue);
  }

  /** execute this test
   */
  void run(util::logger &log) override {
    try {
      auto queue = util::get_cts_object::queue();

      /** check accessor constructors for int
       */
      check_all_dims<int>(log, queue);

      /** check accessor constructors for float
       */
      check_all_dims<float>(log, queue);

      /** check accessor constructors for double
       */
      check_all_dims<double>(log, queue);

      /** check accessor constructors for char
       */
      check_all_dims<char>(log, queue);

      /** check accessor constructors for vec
       */
      check_all_dims<cl::sycl::int2>(log, queue);

      /** check accessor constructors for vec
       */
      check_all_dims<cl::sycl::int3>(log, queue);

      /** check accessor constructors for vec
       */
      check_all_dims<cl::sycl::float4>(log, queue);

      /** check accessor constructors for user_struct
       */
      check_all_dims<user_struct>(log, queue);

      queue.wait_and_throw();
    } catch (const cl::sycl::exception &e) {
      log_exception(log, e);
      cl::sycl::string_class errorMsg =
          "a SYCL exception was caught: " + cl::sycl::string_class(e.what());
      FAIL(log, errorMsg.c_str());
    }
  }
};

// register this test with the test_collection
util::test_proxy<TEST_NAME> proxy;

}  // namespace TEST_NAMESPACE
