/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#define TEST_NAME accessor_constructors_half

#include "../common/common.h"
#include "../accessor/accessor_constructors_utility.h"

namespace TEST_NAMESPACE {
/** unique dummy_functor per file
 *  this is a hack until the CMake script is fixed; kill both the alias and the
 *  dummy class once it is fixed
 */
class dummy_accessor_constructors_half {};
using dummy_functor = ::dummy_functor<dummy_accessor_constructors_half>;

template <typename T, size_t dims>
class buffer_accessor_half_dims {
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

          if (a.get_offset() != getRange<dims>(0)) {
            FAIL(log,
                 "global_buffer accessor for read is not constructed correctly "
                 "(get_offset)");
          }

          if (a.get_range() != range) {
            FAIL(log,
                 "global_buffer accessor for read is not constructed correctly "
                 "(get_range)");
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

          if (a.get_offset() != cl::sycl::range<dims>(range / 2)) {
            FAIL(log,
                 "global_buffer ranged accessor for read is not constructed "
                 "correctly (get_offset)");
          }

          if (a.get_range() != range) {
            FAIL(log,
                 "global_buffer ranged accessor for read is not constructed "
                 "correctly (get_range)");
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

          if (a.get_offset() != getRange<dims>(0)) {
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

          if (a.get_offset() != cl::sycl::range<dims>(range / 2)) {
            FAIL(log,
                 "global_buffer ranged accessor for write is not constructed "
                 "correctly (get_offset)");
          }

          if (a.get_range() != range) {
            FAIL(log,
                 "global_buffer ranged accessor for write is not constructed "
                 "correctly (get_range)");
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

          if (a.get_offset() != getRange<dims>(0)) {
            FAIL(log,
                 "global_buffer accessor for read_write is not constructed "
                 "correctly (get_offset)");
          }

          if (a.get_range() != range) {
            FAIL(log,
                 "global_buffer accessor for read_write is not constructed "
                 "correctly (get_range)");
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

          if (a.get_offset() != cl::sycl::range<dims>(range / 2)) {
            FAIL(log,
                 "global_buffer ranged accessor for read_write is not "
                 "constructed correctly (get_offset)");
          }

          if (a.get_range() != range) {
            FAIL(log,
                 "global_buffer ranged accessor for read_write is not "
                 "constructed correctly (get_range)");
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

          if (a.get_offset() != getRange<dims>(0)) {
            FAIL(log,
                 "global_buffer accessor for discard_write is not constructed "
                 "correctly (get_offset)");
          }

          if (a.get_range() != range) {
            FAIL(log,
                 "global_buffer accessor for discard_write is not constructed "
                 "correctly (get_range)");
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

          if (a.get_offset() != cl::sycl::range<dims>(range / 2)) {
            FAIL(log,
                 "global_buffer ranged accessor for discard_write is not "
                 "constructed correctly (get_offset)");
          }

          if (a.get_range() != range) {
            FAIL(log,
                 "global_buffer ranged accessor for discard_write is not "
                 "constructed correctly (get_range)");
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

          if (a.get_offset() != getRange<dims>(0)) {
            FAIL(log,
                 "global_buffer accessor for discard_read_write is not "
                 "constructed correctly (get_offset)");
          }

          if (a.get_range() != range) {
            FAIL(log,
                 "global_buffer accessor for discard_read_write is not "
                 "constructed correctly (get_range)");
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

          if (a.get_offset() != cl::sycl::range<dims>(range / 2)) {
            FAIL(log,
                 "global_buffer ranged accessor for discard_read_write is not "
                 "constructed correctly (get_offset)");
          }

          if (a.get_range() != range) {
            FAIL(log,
                 "global_buffer ranged accessor for discard_read_write is not "
                 "constructed correctly (get_range)");
          }
        }

        /** check buffer accessor constructors for atomic, only available in
         * global_buffer target
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

          if (a.get_offset() != getRange<dims>(0)) {
            FAIL(log,
                 "global_buffer accessor for atomic is not constructed "
                 "correctly (get_offset)");
          }

          if (a.get_range() != range) {
            FAIL(log,
                 "global_buffer accessor for atomic is not constructed "
                 "correctly (get_range)");
          }
        }
        /** check (buffer, handler, range, offset) constructor for atomic
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

          if (a.get_offset() != cl::sycl::range<dims>(range / 2)) {
            FAIL(log,
                 "global_buffer ranged accessor for atomic is not constructed "
                 "correctly (get_offset)");
          }

          if (a.get_range() != range) {
            FAIL(log,
                 "global_buffer ranged accessor for atomic is not constructed "
                 "correctly (get_range)");
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

          if (b.get_offset() != cl::sycl::range<dims>(range / 2)) {
            FAIL(log,
                 "global_buffer accessor is not move constructible "
                 "(get_offset)");
          }

          if (b.get_range() != range) {
            FAIL(log,
                 "global_buffer accessor is not move constructible "
                 "(get_range)");
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

          if (b.get_offset() != cl::sycl::range<dims>(range / 2)) {
            FAIL(log,
                 "global_buffer accessor is not move assignable (get_offset)");
          }

          if (b.get_range() != range) {
            FAIL(log,
                 "global_buffer accessor is not move assignable (get_range)");
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

          if (a.get_offset() != getRange<dims>(0)) {
            FAIL(log,
                 "constant_buffer accessor for read is not constructed "
                 "correctly (get_offset)");
          }

          if (a.get_range() != range) {
            FAIL(log,
                 "constant_buffer accessor for read is not constructed "
                 "correctly (get_range)");
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

          if (a.get_offset() != cl::sycl::range<dims>(range / 2)) {
            FAIL(log,
                 "constant_buffer ranged accessor for read is not constructed "
                 "correctly (get_offset)");
          }

          if (a.get_range() != range) {
            FAIL(log,
                 "constant_buffer ranged accessor for read is not constructed "
                 "correctly (get_range)");
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

          if (b.get_offset() != cl::sycl::range<dims>(range / 2)) {
            FAIL(log,
                 "constant_buffer accessor is not move constructible "
                 "(get_offset)");
          }

          if (b.get_range() != range) {
            FAIL(log,
                 "constant_buffer accessor is not move constructible "
                 "(get_range)");
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

          if (b.get_offset() != cl::sycl::range<dims>(range / 2)) {
            FAIL(
                log,
                "constant_buffer accessor is not move assignable (get_offset)");
          }

          if (b.get_range() != range) {
            FAIL(log,
                 "constant_buffer accessor is not move assignable (get_range)");
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

        if (a.get_offset() != getRange<dims>(0)) {
          FAIL(log,
               "host_buffer accessor for read is not constructed correctly "
               "(get_offset)");
        }

        if (a.get_range() != range) {
          FAIL(log,
               "host_buffer accessor for read is not constructed correctly "
               "(get_range)");
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

        if (a.get_offset() != cl::sycl::range<dims>(range / 2)) {
          FAIL(log,
               "host_buffer ranged accessor for read is not constructed "
               "correctly (get_offset)");
        }

        if (a.get_range() != range) {
          FAIL(log,
               "host_buffer ranged accessor for read is not constructed "
               "correctly (get_range)");
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

        if (a.get_offset() != getRange<dims>(0)) {
          FAIL(log,
               "host_buffer accessor for write is not constructed correctly "
               "(get_offset)");
        }

        if (a.get_range() != range) {
          FAIL(log,
               "host_buffer accessor for write is not constructed correctly "
               "(get_range)");
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

        if (a.get_offset() != cl::sycl::range<dims>(range / 2)) {
          FAIL(log,
               "host_buffer ranged accessor for write is not constructed "
               "correctly (get_offset)");
        }

        if (a.get_range() != range) {
          FAIL(log,
               "host_buffer ranged accessor for write is not constructed "
               "correctly (get_range)");
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

        if (a.get_offset() != getRange<dims>(0)) {
          FAIL(log,
               "host_buffer accessor for read_write is not constructed "
               "correctly (get_offset)");
        }

        if (a.get_range() != range) {
          FAIL(log,
               "host_buffer accessor for read_write is not constructed "
               "correctly (get_range)");
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

        if (a.get_offset() != cl::sycl::range<dims>(range / 2)) {
          FAIL(log,
               "host_buffer ranged accessor for read_write is not constructed "
               "correctly (get_offset)");
        }

        if (a.get_range() != range) {
          FAIL(log,
               "host_buffer ranged accessor for read_write is not constructed "
               "correctly (get_range)");
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

        if (a.get_offset() != getRange<dims>(0)) {
          FAIL(log,
               "host_buffer accessor for discard_write is not constructed "
               "correctly (get_offset)");
        }

        if (a.get_range() != range) {
          FAIL(log,
               "host_buffer accessor for discard_write is not constructed "
               "correctly (get_range)");
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
               "constructed "
               "correctly (get_size)");
        }

        if (a.get_count() != getElementsCount<dims>(range)) {
          FAIL(log,
               "host_buffer ranged accessor for discard_write is not "
               "constructed "
               "correctly (get_count)");
        }

        if (a.get_offset() != cl::sycl::range<dims>(range / 2)) {
          FAIL(log,
               "host_buffer ranged accessor for discard_write is not "
               "constructed "
               "correctly (get_offset)");
        }

        if (a.get_range() != range) {
          FAIL(log,
               "host_buffer ranged accessor for discard_write is not "
               "constructed "
               "correctly (get_range)");
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

        if (a.get_offset() != getRange<dims>(0)) {
          FAIL(log,
               "host_buffer accessor for discard_read_write is not constructed "
               "correctly (get_offset)");
        }

        if (a.get_range() != range) {
          FAIL(log,
               "host_buffer accessor for discard_read_write is not constructed "
               "correctly (get_range)");
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

        if (a.get_offset() != cl::sycl::range<dims>(range / 2)) {
          FAIL(log,
               "host_buffer ranged accessor for discard_read_write is not "
               "constructed correctly (get_offset)");
        }

        if (a.get_range() != range) {
          FAIL(log,
               "host_buffer ranged accessor for discard_read_write is not "
               "constructed correctly (get_range)");
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

        if (b.get_offset() != cl::sycl::range<dims>(range / 2)) {
          FAIL(log,
               "host_buffer accessor is not move constructible (get_offset)");
        }

        if (b.get_range() != range) {
          FAIL(log,
               "host_buffer accessor is not move constructible (get_range)");
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

        if (b.get_offset() != cl::sycl::range<dims>(range / 2)) {
          FAIL(log, "host_buffer accessor is not move assignable (get_offset)");
        }

        if (b.get_range() != range) {
          FAIL(log, "host_buffer accessor is not move assignable (get_range)");
        }
      }
    }
  }
};

template <typename T>
class buffer_accessor_half_dims<T, 0> {
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
                 "global_buffer accessor for read is not constructed "
                 "correctly (get_size)");
          }

          if (a.get_count() != 1) {
            FAIL(log,
                 "global_buffer accessor for read is not constructed "
                 "correctly (get_count)");
          }

          if (a.get_offset() != getRange<1>(0)) {
            FAIL(log,
                 "global_buffer accessor for read is not constructed "
                 "correctly (get_offset)");
          }

          if (a.get_range() != getRange<1>(1)) {
            FAIL(log,
                 "global_buffer accessor for read is not constructed "
                 "correctly (get_range)");
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

          if (a.get_offset() != getRange<1>(0)) {
            FAIL(log,
                 "global_buffer accessor for write is not constructed "
                 "correctly (get_offset)");
          }

          if (a.get_range() != getRange<1>(1)) {
            FAIL(log,
                 "global_buffer accessor for write is not constructed "
                 "correctly (get_range)");
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

          if (a.get_offset() != cl::sycl::range<1>(0)) {
            FAIL(log,
                 "global_buffer accessor for read_write is not constructed "
                 "correctly (get_offset)");
          }

          if (a.get_range() != cl::sycl::range<1>(1)) {
            FAIL(log,
                 "global_buffer accessor for read_write is not constructed "
                 "correctly (get_range)");
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

          if (a.get_offset() != getRange<1>(0)) {
            FAIL(log,
                 "global_buffer accessor for discard_write is not constructed "
                 "correctly (get_offset)");
          }

          if (a.get_range() != getRange<1>(1)) {
            FAIL(log,
                 "global_buffer accessor for discard_write is not constructed "
                 "correctly (get_range)");
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

          if (a.get_offset() != getRange<1>(0)) {
            FAIL(log,
                 "global_buffer accessor for discard_read_write is not "
                 "constructed correctly (get_offset)");
          }

          if (a.get_range() != getRange<1>(1)) {
            FAIL(log,
                 "global_buffer accessor for discard_read_write is not "
                 "constructed correctly (get_range)");
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

          if (a.get_offset() != getRange<1>(0)) {
            FAIL(log,
                 "global_buffer accessor for atomic is not constructed "
                 "correctly (get_offset)");
          }

          if (a.get_range() != getRange<1>(1)) {
            FAIL(log,
                 "global_buffer accessor for atomic is not constructed "
                 "correctly (get_range)");
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

          if (a.get_offset() != b.get_offset()) {
            FAIL(log,
                 "global_buffer accessor is not copy assignable (get_offset)");
          }

          if (a.get_range() != b.get_range()) {
            FAIL(log,
                 "global_buffer accessor is not copy assignable (get_range)");
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

          if (b.get_offset() != cl::sycl::range<1>(0)) {
            FAIL(log,
                 "global_buffer accessor is not move constructible "
                 "(get_offset)");
          }

          if (b.get_range() != cl::sycl::range<1>(1)) {
            FAIL(log,
                 "global_buffer accessor is not move constructible "
                 "(get_range)");
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

          if (b.get_offset() != cl::sycl::range<1>(0)) {
            FAIL(log,
                 "global_buffer accessor is not move constructible "
                 "(get_offset)");
          }

          if (b.get_range() != cl::sycl::range<1>(1)) {
            FAIL(log,
                 "global_buffer accessor is not move constructible "
                 "(get_range)");
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

          if (a.get_offset() != getRange<1>(0)) {
            FAIL(log,
                 "constant_buffer accessor for read is not constructed "
                 "correctly (get_offset)");
          }

          if (a.get_range() != getRange<1>(1)) {
            FAIL(log,
                 "constant_buffer accessor for read is not constructed "
                 "correctly (get_range)");
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

          if (a.get_offset() != b.get_offset()) {
            FAIL(
                log,
                "constant_buffer accessor is not copy assignable (get_offset)");
          }

          if (a.get_range() != b.get_range()) {
            FAIL(log,
                 "constant_buffer accessor is not copy assignable (get_range)");
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

          if (b.get_offset() != cl::sycl::range<1>(0)) {
            FAIL(log,
                 "constant_buffer accessor is not move constructible "
                 "(get_offset)");
          }

          if (b.get_range() != cl::sycl::range<1>(1)) {
            FAIL(log,
                 "constant_buffer accessor is not move constructible "
                 "(get_range)");
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

          if (b.get_offset() != cl::sycl::range<1>(0)) {
            FAIL(log,
                 "constant_buffer accessor is not move constructible "
                 "(get_offset)");
          }

          if (b.get_range() != cl::sycl::range<1>(1)) {
            FAIL(log,
                 "constant_buffer accessor is not move constructible "
                 "(get_range)");
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
               "host_buffer accessor for read is not constructed "
               "correctly (get_size)");
        }

        if (a.get_count() != 1) {
          FAIL(log,
               "host_buffer accessor for read is not constructed "
               "correctly (get_count)");
        }

        if (a.get_offset() != getRange<1>(0)) {
          FAIL(log,
               "host_buffer accessor for read is not constructed "
               "correctly (get_offset)");
        }

        if (a.get_range() != getRange<1>(1)) {
          FAIL(log,
               "host_buffer accessor for read is not constructed "
               "correctly (get_range)");
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
               "host_buffer accessor for write is not constructed "
               "correctly (get_size)");
        }

        if (a.get_count() != 1) {
          FAIL(log,
               "host_buffer accessor for write is not constructed "
               "correctly (get_count)");
        }

        if (a.get_offset() != getRange<1>(0)) {
          FAIL(log,
               "host_buffer accessor for write is not constructed "
               "correctly (get_offset)");
        }

        if (a.get_range() != getRange<1>(1)) {
          FAIL(log,
               "host_buffer accessor for write is not constructed "
               "correctly (get_range)");
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

        if (a.get_offset() != cl::sycl::range<1>(0)) {
          FAIL(log,
               "host_buffer accessor for read_write is not constructed "
               "correctly (get_offset)");
        }

        if (a.get_range() != cl::sycl::range<1>(1)) {
          FAIL(log,
               "host_buffer accessor for read_write is not constructed "
               "correctly (get_range)");
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

        if (a.get_offset() != getRange<1>(0)) {
          FAIL(log,
               "host_buffer accessor for discard_write is not constructed "
               "correctly (get_offset)");
        }

        if (a.get_range() != getRange<1>(1)) {
          FAIL(log,
               "host_buffer accessor for discard_write is not constructed "
               "correctly (get_range)");
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

        if (a.get_offset() != getRange<1>(0)) {
          FAIL(log,
               "host_buffer accessor for discard_read_write is not "
               "constructed correctly (get_offset)");
        }

        if (a.get_range() != getRange<1>(1)) {
          FAIL(log,
               "host_buffer accessor for discard_read_write is not "
               "constructed correctly (get_range)");
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

        if (a.get_offset() != b.get_offset()) {
          FAIL(log,
               "global_buffer accessor is not copy constructible "
               "(get_offset)");
        }

        if (a.get_range() != b.get_range()) {
          FAIL(log,
               "global_buffer accessor is not copy constructible (get_range)");
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

        if (a.get_offset() != b.get_offset()) {
          FAIL(log,
               "global_buffer accessor is not copy assignable (get_offset)");
        }

        if (a.get_range() != b.get_range()) {
          FAIL(log,
               "global_buffer accessor is not copy assignable (get_range)");
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

        if (b.get_offset() != cl::sycl::range<1>(0)) {
          FAIL(log,
               "global_buffer accessor is not move constructible "
               "(get_offset)");
        }

        if (b.get_range() != cl::sycl::range<1>(1)) {
          FAIL(log,
               "global_buffer accessor is not move constructible "
               "(get_range)");
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

        if (b.get_offset() != cl::sycl::range<1>(0)) {
          FAIL(log,
               "global_buffer accessor is not move constructible "
               "(get_offset)");
        }

        if (b.get_range() != cl::sycl::range<1>(1)) {
          FAIL(log,
               "global_buffer accessor is not move constructible "
               "(get_range)");
        }
      }
    }
  }
};

template <typename T, size_t dims>
class placeholder_accessor_half_dims {
 public:
  static void check(util::logger &log, cl::sycl::queue &queue) {
    int size = 32;

    /** check buffer accessor constructors for n > 0 dimension
     */

    cl::sycl::range<dims> range = getRange<dims>(size);
    std::vector<uint8_t> data(getElementsCount<dims>(range) * sizeof(T), 0);
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
                             cl::sycl::access::placeholder::true_t>
              a(buffer, h);

          if (a.get_size() != getElementsCount<dims>(range) * sizeof(T)) {
            FAIL(log,
                 "global_buffer placeholder accessor for read is not "
                 "constructed correctly "
                 "(get_size)");
          }

          if (a.get_count() != getElementsCount<dims>(range)) {
            FAIL(log,
                 "global_buffer placeholder accessor for read is not "
                 "constructed correctly "
                 "(get_count)");
          }

          if (a.get_offset() != getRange<dims>(0)) {
            FAIL(log,
                 "global_buffer placeholder accessor for read is not "
                 "constructed correctly "
                 "(get_offset)");
          }

          if (a.get_range() != range) {
            FAIL(log,
                 "global_buffer placeholder accessor for read is not "
                 "constructed correctly "
                 "(get_range)");
          }
        }

        /** check (buffer, handler, range, offset) constructor for reading
         * global_buffer
         */
        {
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                             cl::sycl::access::target::global_buffer,
                             cl::sycl::access::placeholder::true_t>
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

          if (a.get_offset() != cl::sycl::range<dims>(range / 2)) {
            FAIL(log,
                 "global_buffer ranged accessor for read is not constructed "
                 "correctly (get_offset)");
          }

          if (a.get_range() != range) {
            FAIL(log,
                 "global_buffer ranged accessor for read is not constructed "
                 "correctly (get_range)");
          }
        }

        /** check (buffer, handler) constructor for writing global_buffer
         */
        {
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::write,
                             cl::sycl::access::target::global_buffer,
                             cl::sycl::access::placeholder::true_t>
              a(buffer, h);

          if (a.get_size() != getElementsCount<dims>(range) * sizeof(T)) {
            FAIL(log,
                 "global_buffer placeholder accessor for write is not "
                 "constructed correctly "
                 "(get_size)");
          }

          if (a.get_count() != getElementsCount<dims>(range)) {
            FAIL(log,
                 "global_buffer placeholder accessor for write is not "
                 "constructed correctly "
                 "(get_count)");
          }

          if (a.get_offset() != getRange<dims>(0)) {
            FAIL(log,
                 "global_buffer placeholder accessor for write is not "
                 "constructed correctly "
                 "(get_offset)");
          }

          if (a.get_range() != range) {
            FAIL(log,
                 "global_buffer placeholder accessor for write is not "
                 "constructed correctly "
                 "(get_range)");
          }
        }
        /** check (buffer, handler, range, offset) constructor for writing
         * global_buffer
         */
        {
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::write,
                             cl::sycl::access::target::global_buffer,
                             cl::sycl::access::placeholder::true_t>
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

          if (a.get_offset() != cl::sycl::range<dims>(range / 2)) {
            FAIL(log,
                 "global_buffer ranged accessor for write is not constructed "
                 "correctly (get_offset)");
          }

          if (a.get_range() != range) {
            FAIL(log,
                 "global_buffer ranged accessor for write is not constructed "
                 "correctly (get_range)");
          }
        }

        /** check (buffer, handler) constructor for read_write global_buffer
         */
        {
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::read_write,
                             cl::sycl::access::target::global_buffer,
                             cl::sycl::access::placeholder::true_t>
              a(buffer, h);

          if (a.get_size() != getElementsCount<dims>(range) * sizeof(T)) {
            FAIL(log,
                 "global_buffer placeholder accessor for read_write is not "
                 "constructed correctly (get_size)");
          }

          if (a.get_count() != getElementsCount<dims>(range)) {
            FAIL(log,
                 "global_buffer placeholder accessor for read_write is not "
                 "constructed correctly (get_count)");
          }

          if (a.get_offset() != getRange<dims>(0)) {
            FAIL(log,
                 "global_buffer placeholder accessor for read_write is not "
                 "constructed correctly (get_offset)");
          }

          if (a.get_range() != range) {
            FAIL(log,
                 "global_buffer placeholder accessor for read_write is not "
                 "constructed correctly (get_range)");
          }
        }
        /** check (buffer, handler, range, offset) constructor for read_write
         * global_buffer
         */
        {
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::read_write,
                             cl::sycl::access::target::global_buffer,
                             cl::sycl::access::placeholder::true_t>
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

          if (a.get_offset() != cl::sycl::range<dims>(range / 2)) {
            FAIL(log,
                 "global_buffer ranged accessor for read_write is not "
                 "constructed correctly (get_offset)");
          }

          if (a.get_range() != range) {
            FAIL(log,
                 "global_buffer ranged accessor for read_write is not "
                 "constructed correctly (get_range)");
          }
        }

        /** check (buffer, handler) constructor for discard_write global_buffer
         */
        {
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::discard_write,
                             cl::sycl::access::target::global_buffer,
                             cl::sycl::access::placeholder::true_t>
              a(buffer, h);

          if (a.get_size() != getElementsCount<dims>(range) * sizeof(T)) {
            FAIL(log,
                 "global_buffer placeholder accessor for discard_write is not "
                 "constructed correctly (get_size)");
          }

          if (a.get_count() != getElementsCount<dims>(range)) {
            FAIL(log,
                 "global_buffer placeholder accessor for discard_write is not "
                 "constructed correctly (get_count)");
          }

          if (a.get_offset() != getRange<dims>(0)) {
            FAIL(log,
                 "global_buffer placeholder accessor for discard_write is not "
                 "constructed correctly (get_offset)");
          }

          if (a.get_range() != range) {
            FAIL(log,
                 "global_buffer placeholder accessor for discard_write is not "
                 "constructed correctly (get_range)");
          }
        }
        /** check (buffer, handler, range, offset) constructor for discard_write
         * global_buffer
         */
        {
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::discard_write,
                             cl::sycl::access::target::global_buffer,
                             cl::sycl::access::placeholder::true_t>
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

          if (a.get_offset() != cl::sycl::range<dims>(range / 2)) {
            FAIL(log,
                 "global_buffer ranged accessor for discard_write is not "
                 "constructed correctly (get_offset)");
          }

          if (a.get_range() != range) {
            FAIL(log,
                 "global_buffer ranged accessor for discard_write is not "
                 "constructed correctly (get_range)");
          }
        }

        /** check (buffer, handler) constructor for discard_read_write
         * global_buffer
         */
        {
          cl::sycl::accessor<T, dims,
                             cl::sycl::access::mode::discard_read_write,
                             cl::sycl::access::target::global_buffer,
                             cl::sycl::access::placeholder::true_t>
              a(buffer, h);

          if (a.get_size() != getElementsCount<dims>(range) * sizeof(T)) {
            FAIL(log,
                 "global_buffer placeholder accessor for discard_read_write is "
                 "not constructed correctly (get_size)");
          }

          if (a.get_count() != getElementsCount<dims>(range)) {
            FAIL(log,
                 "global_buffer placeholder accessor for discard_read_write is "
                 "not constructed correctly (get_count)");
          }

          if (a.get_offset() != getRange<dims>(0)) {
            FAIL(log,
                 "global_buffer placeholder accessor for discard_read_write is "
                 "not constructed correctly (get_offset)");
          }

          if (a.get_range() != range) {
            FAIL(log,
                 "global_buffer placeholder accessor for discard_read_write is "
                 "not constructed correctly (get_range)");
          }
        }
        /** check (buffer, handler, range, offset) constructor for
         * discard_read_write global_buffer
         */
        {
          cl::sycl::accessor<T, dims,
                             cl::sycl::access::mode::discard_read_write,
                             cl::sycl::access::target::global_buffer,
                             cl::sycl::access::placeholder::true_t>
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

          if (a.get_offset() != cl::sycl::range<dims>(range / 2)) {
            FAIL(log,
                 "global_buffer ranged accessor for discard_read_write is not "
                 "constructed correctly (get_offset)");
          }

          if (a.get_range() != range) {
            FAIL(log,
                 "global_buffer ranged accessor for discard_read_write is not "
                 "constructed correctly (get_range)");
          }
        }

        /** check buffer accessor constructors for atomic, only available in
         * global_buffer target
         */

        {
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::atomic,
                             cl::sycl::access::target::global_buffer,
                             cl::sycl::access::placeholder::true_t>
              a(buffer, h);

          if (a.get_size() != getElementsCount<dims>(range) * sizeof(T)) {
            FAIL(log,
                 "global_buffer placeholder accessor for atomic is not "
                 "constructed correctly (get_size)");
          }

          if (a.get_count() != getElementsCount<dims>(range)) {
            FAIL(log,
                 "global_buffer placeholder accessor for atomic is not "
                 "constructed correctly (get_count)");
          }

          if (a.get_offset() != getRange<dims>(0)) {
            FAIL(log,
                 "global_buffer placeholder accessor for atomic is not "
                 "constructed correctly (get_offset)");
          }

          if (a.get_range() != range) {
            FAIL(log,
                 "global_buffer placeholder accessor for atomic is not "
                 "constructed correctly (get_range)");
          }
        }
        /** check (buffer, handler, range, offset) constructor for atomic
         */
        {
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::atomic,
                             cl::sycl::access::target::global_buffer,
                             cl::sycl::access::placeholder::true_t>
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

          if (a.get_offset() != cl::sycl::range<dims>(range / 2)) {
            FAIL(log,
                 "global_buffer ranged accessor for atomic is not constructed "
                 "correctly (get_offset)");
          }

          if (a.get_range() != range) {
            FAIL(log,
                 "global_buffer ranged accessor for atomic is not constructed "
                 "correctly (get_range)");
          }
        }

        /** check accessor is Copy Constructible
         */
        {
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                             cl::sycl::access::target::global_buffer,
                             cl::sycl::access::placeholder::true_t>
              a(buffer, h, r, offset);
          auto b{a};

          if (a.get_size() != b.get_size()) {
            FAIL(log,
                 "global_buffer placeholder accessor is not copy constructible "
                 "(get_size)");
          }

          if (a.get_count() != b.get_count()) {
            FAIL(log,
                 "global_buffer placeholder accessor is not copy constructible "
                 "(get_count)");
          }

          if (a.get_offset() != b.get_offset()) {
            FAIL(log,
                 "global_buffer placeholder accessor is not copy constructible "
                 "(get_offset)");
          }

          if (a.get_range() != b.get_range()) {
            FAIL(log,
                 "global_buffer placeholder accessor is not copy constructible "
                 "(get_range)");
          }
        }

        /** check accessor is Copy Assignable
         */
        {
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                             cl::sycl::access::target::global_buffer,
                             cl::sycl::access::placeholder::true_t>
              a(buffer, h, r, offset);
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                             cl::sycl::access::target::global_buffer,
                             cl::sycl::access::placeholder::true_t>
              b(buffer, h);
          b = a;

          if (a.get_size() != b.get_size()) {
            FAIL(log,
                 "global_buffer placeholder accessor is not copy assignable "
                 "(get_size)");
          }

          if (a.get_count() != b.get_count()) {
            FAIL(log,
                 "global_buffer placeholder accessor is not copy assignable "
                 "(get_count)");
          }

          if (a.get_offset() != b.get_offset()) {
            FAIL(log,
                 "global_buffer placeholder accessor is not copy assignable "
                 "(get_offset)");
          }

          if (a.get_range() != b.get_range()) {
            FAIL(log,
                 "global_buffer placeholder accessor is not copy assignable "
                 "(get_range)");
          }
        }

        /** check accessor is Move Constructible
         */
        {
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                             cl::sycl::access::target::global_buffer,
                             cl::sycl::access::placeholder::true_t>
              a(buffer, h, r, offset);
          auto b{std::move(a)};

          if (b.get_size() != getElementsCount<dims>(range) * sizeof(T)) {
            FAIL(log,
                 "global_buffer placeholder accessor is not move constructible "
                 "(get_size)");
          }

          if (b.get_count() != getElementsCount<dims>(range)) {
            FAIL(log,
                 "global_buffer placeholder accessor is not move constructible "
                 "(get_count)");
          }

          if (b.get_offset() != cl::sycl::range<dims>(range / 2)) {
            FAIL(log,
                 "global_buffer placeholder accessor is not move constructible "
                 "(get_offset)");
          }

          if (b.get_range() != range) {
            FAIL(log,
                 "global_buffer placeholder accessor is not move constructible "
                 "(get_range)");
          }
        }

        /** check accessor is Move Assignable
         */
        {
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                             cl::sycl::access::target::global_buffer,
                             cl::sycl::access::placeholder::true_t>
              a(buffer, h, r, offset);
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                             cl::sycl::access::target::global_buffer,
                             cl::sycl::access::placeholder::true_t>
              b(buffer, h);
          b = std::move(a);

          if (b.get_size() != getElementsCount<dims>(range) * sizeof(T)) {
            FAIL(log,
                 "global_buffer placeholder accessor is not move assignable "
                 "(get_size)");
          }

          if (b.get_count() != getElementsCount<dims>(range)) {
            FAIL(log,
                 "global_buffer placeholder accessor is not move assignable "
                 "(get_count)");
          }

          if (b.get_offset() != cl::sycl::range<dims>(range / 2)) {
            FAIL(log,
                 "global_buffer placeholder accessor is not move assignable "
                 "(get_offset)");
          }

          if (b.get_range() != range) {
            FAIL(log,
                 "global_buffer placeholder accessor is not move assignable "
                 "(get_range)");
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
                             cl::sycl::access::placeholder::true_t>
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

          if (a.get_offset() != getRange<dims>(0)) {
            FAIL(log,
                 "constant_buffer accessor for read is not constructed "
                 "correctly (get_offset)");
          }

          if (a.get_range() != range) {
            FAIL(log,
                 "constant_buffer accessor for read is not constructed "
                 "correctly (get_range)");
          }
        }
        /** check (buffer, handler, range, offset) constructor for reading
         * constant_buffer
         */
        {
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                             cl::sycl::access::target::constant_buffer,
                             cl::sycl::access::placeholder::true_t>
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

          if (a.get_offset() != cl::sycl::range<dims>(range / 2)) {
            FAIL(log,
                 "constant_buffer ranged accessor for read is not constructed "
                 "correctly (get_offset)");
          }

          if (a.get_range() != range) {
            FAIL(log,
                 "constant_buffer ranged accessor for read is not constructed "
                 "correctly (get_range)");
          }
        }

        /** check accessor is Copy Constructible
         */
        {
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                             cl::sycl::access::target::constant_buffer,
                             cl::sycl::access::placeholder::true_t>
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
        }

        /** check accessor is Copy Assignable
         */
        {
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                             cl::sycl::access::target::constant_buffer,
                             cl::sycl::access::placeholder::true_t>
              a(buffer, h, r, offset);
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                             cl::sycl::access::target::constant_buffer,
                             cl::sycl::access::placeholder::true_t>
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
        }

        /** check accessor is Move Constructible
         */
        {
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                             cl::sycl::access::target::constant_buffer,
                             cl::sycl::access::placeholder::true_t>
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

          if (b.get_offset() != cl::sycl::range<dims>(range / 2)) {
            FAIL(log,
                 "constant_buffer accessor is not move constructible "
                 "(get_offset)");
          }

          if (b.get_range() != range) {
            FAIL(log,
                 "constant_buffer accessor is not move constructible "
                 "(get_range)");
          }
        }

        /** check accessor is Move Assignable
         */
        {
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                             cl::sycl::access::target::constant_buffer,
                             cl::sycl::access::placeholder::true_t>
              a(buffer, h, r, offset);
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                             cl::sycl::access::target::constant_buffer,
                             cl::sycl::access::placeholder::true_t>
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

          if (b.get_offset() != cl::sycl::range<dims>(range / 2)) {
            FAIL(
                log,
                "constant_buffer accessor is not move assignable (get_offset)");
          }

          if (b.get_range() != range) {
            FAIL(log,
                 "constant_buffer accessor is not move assignable (get_range)");
          }
        }

        /** dummy kernel as no kernel is required for these checks
         */
        h.single_task(dummy_functor{});

      });
      queue.wait_and_throw();
    }
  }
};

template <typename T>
class placeholder_accessor_half_dims<T, 0> {
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
                             cl::sycl::access::placeholder::true_t>
              a(buffer, h);
          if (a.get_size() != sizeof(T)) {
            FAIL(log,
                 "global_buffer placeholder accessor for read is not "
                 "constructed correctly (get_size)");
          }

          if (a.get_count() != 1) {
            FAIL(log,
                 "global_buffer placeholder accessor for read is not "
                 "constructed correctly (get_count)");
          }

          if (a.get_offset() != getRange<1>(0)) {
            FAIL(log,
                 "global_buffer placeholder accessor for read is not "
                 "constructed correctly (get_offset)");
          }

          if (a.get_range() != getRange<1>(1)) {
            FAIL(log,
                 "global_buffer placeholder accessor for read is not "
                 "constructed correctly (get_range)");
          }
        }

        /** check (buffer) constructor for write global_buffer
         */
        {
          cl::sycl::accessor<T, 0, cl::sycl::access::mode::write,
                             cl::sycl::access::target::global_buffer,
                             cl::sycl::access::placeholder::true_t>
              a(buffer, h);
          if (a.get_size() != sizeof(T)) {
            FAIL(log,
                 "global_buffer placeholder accessor for write is not "
                 "constructed correctly (get_size)");
          }

          if (a.get_count() != 1) {
            FAIL(log,
                 "global_buffer placeholder accessor for write is not "
                 "constructed correctly (get_count)");
          }

          if (a.get_offset() != getRange<1>(0)) {
            FAIL(log,
                 "global_buffer placeholder accessor for write is not "
                 "constructed correctly (get_offset)");
          }

          if (a.get_range() != getRange<1>(1)) {
            FAIL(log,
                 "global_buffer placeholder accessor for write is not "
                 "constructed correctly (get_range)");
          }
        }

        /** check (buffer) constructor for read_write global_buffer
         */
        {
          cl::sycl::accessor<T, 0, cl::sycl::access::mode::read_write,
                             cl::sycl::access::target::global_buffer,
                             cl::sycl::access::placeholder::true_t>
              a(buffer, h);
          if (a.get_size() != sizeof(T)) {
            FAIL(log,
                 "global_buffer placeholder accessor for read_write is not "
                 "constructed correctly (get_size)");
          }

          if (a.get_count() != 1) {
            FAIL(log,
                 "global_buffer placeholder accessor for read_write is not "
                 "constructed correctly (get_count)");
          }

          if (a.get_offset() != cl::sycl::range<1>(0)) {
            FAIL(log,
                 "global_buffer placeholder accessor for read_write is not "
                 "constructed correctly (get_offset)");
          }

          if (a.get_range() != cl::sycl::range<1>(1)) {
            FAIL(log,
                 "global_buffer placeholder accessor for read_write is not "
                 "constructed correctly (get_range)");
          }
        }

        /** check (buffer) constructor for discard_write global_buffer
         */
        {
          cl::sycl::accessor<T, 0, cl::sycl::access::mode::discard_write,
                             cl::sycl::access::target::global_buffer,
                             cl::sycl::access::placeholder::true_t>
              a(buffer, h);
          if (a.get_size() != sizeof(T)) {
            FAIL(log,
                 "global_buffer placeholder accessor for discard_write is not "
                 "constructed correctly (get_size)");
          }

          if (a.get_count() != 1) {
            FAIL(log,
                 "global_buffer placeholder accessor for discard_write is not "
                 "constructed correctly (get_count)");
          }

          if (a.get_offset() != getRange<1>(0)) {
            FAIL(log,
                 "global_buffer placeholder accessor for discard_write is not "
                 "constructed correctly (get_offset)");
          }

          if (a.get_range() != getRange<1>(1)) {
            FAIL(log,
                 "global_buffer placeholder accessor for discard_write is not "
                 "constructed correctly (get_range)");
          }
        }

        /** check (buffer) constructor for discard_read_write global_buffer
         */
        {
          cl::sycl::accessor<T, 0, cl::sycl::access::mode::discard_read_write,
                             cl::sycl::access::target::global_buffer,
                             cl::sycl::access::placeholder::true_t>
              a(buffer, h);
          if (a.get_size() != sizeof(T)) {
            FAIL(log,
                 "global_buffer placeholder accessor for discard_read_write is "
                 "not constructed correctly (get_size)");
          }

          if (a.get_count() != 1) {
            FAIL(log,
                 "global_buffer placeholder accessor for discard_read_write is "
                 "not constructed correctly (get_count)");
          }

          if (a.get_offset() != getRange<1>(0)) {
            FAIL(log,
                 "global_buffer placeholder accessor for discard_read_write is "
                 "not constructed correctly (get_offset)");
          }

          if (a.get_range() != getRange<1>(1)) {
            FAIL(log,
                 "global_buffer placeholder accessor for discard_read_write is "
                 "not constructed correctly (get_range)");
          }
        }

        /** check (buffer) constructor for atomic global_buffer
         */
        {
          cl::sycl::accessor<T, 0, cl::sycl::access::mode::atomic,
                             cl::sycl::access::target::global_buffer,
                             cl::sycl::access::placeholder::true_t>
              a(buffer, h);
          if (a.get_size() != sizeof(T)) {
            FAIL(log,
                 "global_buffer placeholder accessor for atomic is not "
                 "constructed correctly (get_size)");
          }

          if (a.get_count() != 1) {
            FAIL(log,
                 "global_buffer placeholder accessor for atomic is not "
                 "constructed correctly (get_count)");
          }

          if (a.get_offset() != getRange<1>(0)) {
            FAIL(log,
                 "global_buffer placeholder accessor for atomic is not "
                 "constructed correctly (get_offset)");
          }

          if (a.get_range() != getRange<1>(1)) {
            FAIL(log,
                 "global_buffer placeholder accessor for atomic is not "
                 "constructed correctly (get_range)");
          }
        }

        /** check accessor is Copy Constructible
         */
        {
          cl::sycl::accessor<T, 0, cl::sycl::access::mode::read,
                             cl::sycl::access::target::global_buffer,
                             cl::sycl::access::placeholder::true_t>
              a(buffer, h);
          auto b{a};

          if (a.get_size() != b.get_size()) {
            FAIL(log,
                 "global_buffer placeholder accessor is not copy constructible "
                 "(get_size)");
          }

          if (a.get_count() != b.get_count()) {
            FAIL(log,
                 "global_buffer placeholder accessor is not copy constructible "
                 "(get_count)");
          }

          if (a.get_offset() != b.get_offset()) {
            FAIL(log,
                 "global_buffer placeholder accessor is not copy constructible "
                 "(get_offset)");
          }

          if (a.get_range() != b.get_range()) {
            FAIL(log,
                 "global_buffer placeholder accessor is not copy constructible "
                 "(get_range)");
          }
        }

        /** check accessor is Copy Assignable
         */
        {
          cl::sycl::accessor<T, 0, cl::sycl::access::mode::read,
                             cl::sycl::access::target::global_buffer,
                             cl::sycl::access::placeholder::true_t>
              a(buffer, h);

          cl::sycl::accessor<T, 0, cl::sycl::access::mode::read,
                             cl::sycl::access::target::global_buffer,
                             cl::sycl::access::placeholder::true_t>
              b(buffer, h);
          b = a;

          if (a.get_size() != b.get_size()) {
            FAIL(log,
                 "global_buffer placeholder accessor is not copy assignable "
                 "(get_size)");
          }

          if (a.get_count() != b.get_count()) {
            FAIL(log,
                 "global_buffer placeholder accessor is not copy assignable "
                 "(get_count)");
          }

          if (a.get_offset() != b.get_offset()) {
            FAIL(log,
                 "global_buffer placeholder accessor is not copy assignable "
                 "(get_offset)");
          }

          if (a.get_range() != b.get_range()) {
            FAIL(log,
                 "global_buffer placeholder accessor is not copy assignable "
                 "(get_range)");
          }
        }

        /** check accessor is Move Constructible
         */
        {
          cl::sycl::accessor<T, 0, cl::sycl::access::mode::read,
                             cl::sycl::access::target::global_buffer,
                             cl::sycl::access::placeholder::true_t>
              a(buffer, h);
          auto b{std::move(a)};

          if (b.get_size() != sizeof(T)) {
            FAIL(log,
                 "global_buffer placeholder accessor is not move constructible "
                 "(get_size)");
          }

          if (b.get_count() != 1) {
            FAIL(log,
                 "global_buffer placeholder accessor is not move constructible "
                 "(get_count)");
          }

          if (b.get_offset() != cl::sycl::range<1>(0)) {
            FAIL(log,
                 "global_buffer placeholder accessor is not move constructible "
                 "(get_offset)");
          }

          if (b.get_range() != cl::sycl::range<1>(1)) {
            FAIL(log,
                 "global_buffer placeholder accessor is not move constructible "
                 "(get_range)");
          }
        }

        /** check accessor is Move Assignable
         */
        {
          cl::sycl::accessor<T, 0, cl::sycl::access::mode::read,
                             cl::sycl::access::target::global_buffer,
                             cl::sycl::access::placeholder::true_t>
              a(buffer, h);

          cl::sycl::accessor<T, 0, cl::sycl::access::mode::read,
                             cl::sycl::access::target::global_buffer,
                             cl::sycl::access::placeholder::true_t>
              b(buffer, h);
          b = std::move(a);

          if (b.get_size() != sizeof(T)) {
            FAIL(log,
                 "global_buffer placeholder accessor is not move constructible "
                 "(get_size)");
          }

          if (b.get_count() != 1) {
            FAIL(log,
                 "global_buffer placeholder accessor is not move constructible "
                 "(get_count)");
          }

          if (b.get_offset() != cl::sycl::range<1>(0)) {
            FAIL(log,
                 "global_buffer placeholder accessor is not move constructible "
                 "(get_offset)");
          }

          if (b.get_range() != cl::sycl::range<1>(1)) {
            FAIL(log,
                 "global_buffer placeholder accessor is not move constructible "
                 "(get_range)");
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
                             cl::sycl::access::placeholder::true_t>
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

          if (a.get_offset() != getRange<1>(0)) {
            FAIL(log,
                 "constant_buffer accessor for read is not constructed "
                 "correctly (get_offset)");
          }

          if (a.get_range() != getRange<1>(1)) {
            FAIL(log,
                 "constant_buffer accessor for read is not constructed "
                 "correctly (get_range)");
          }
        }

        /** check accessor is Copy Constructible
         */
        {
          cl::sycl::accessor<T, 0, cl::sycl::access::mode::read,
                             cl::sycl::access::target::constant_buffer,
                             cl::sycl::access::placeholder::true_t>
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
        }

        /** check accessor is Copy Assignable
         */
        {
          cl::sycl::accessor<T, 0, cl::sycl::access::mode::read,
                             cl::sycl::access::target::constant_buffer,
                             cl::sycl::access::placeholder::true_t>
              a(buffer, h);

          cl::sycl::accessor<T, 0, cl::sycl::access::mode::read,
                             cl::sycl::access::target::constant_buffer,
                             cl::sycl::access::placeholder::true_t>
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
        }

        /** check accessor is Move Constructible
         */
        {
          cl::sycl::accessor<T, 0, cl::sycl::access::mode::read,
                             cl::sycl::access::target::constant_buffer,
                             cl::sycl::access::placeholder::true_t>
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

          if (b.get_offset() != cl::sycl::range<1>(0)) {
            FAIL(log,
                 "constant_buffer accessor is not move constructible "
                 "(get_offset)");
          }

          if (b.get_range() != cl::sycl::range<1>(1)) {
            FAIL(log,
                 "constant_buffer accessor is not move constructible "
                 "(get_range)");
          }
        }

        /** check accessor is Move Assignable
         */
        {
          cl::sycl::accessor<T, 0, cl::sycl::access::mode::read,
                             cl::sycl::access::target::constant_buffer,
                             cl::sycl::access::placeholder::true_t>
              a(buffer, h);

          cl::sycl::accessor<T, 0, cl::sycl::access::mode::read,
                             cl::sycl::access::target::constant_buffer,
                             cl::sycl::access::placeholder::true_t>
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

          if (b.get_offset() != cl::sycl::range<1>(0)) {
            FAIL(log,
                 "constant_buffer accessor is not move constructible "
                 "(get_offset)");
          }

          if (b.get_range() != cl::sycl::range<1>(1)) {
            FAIL(log,
                 "constant_buffer accessor is not move constructible "
                 "(get_range)");
          }
        }

        /** dummy kernel as no kernel is required for these checks
         */
        h.single_task(dummy_functor{});

      });
      queue.wait_and_throw();
    }
  }
};

template <typename T, size_t dims>
class local_accessor_half_dims {
 public:
  static void check(util::logger &log, cl::sycl::queue &queue) {
    int size = 32;

    /** check buffer accessor constructors for n > 0 dimension
     */

    cl::sycl::range<dims> range = getRange<dims>(size);
    std::vector<uint8_t> data(getElementsCount<dims>(range) * sizeof(T), 0);
    cl::sycl::buffer<T, dims> buffer(reinterpret_cast<T *>(data.data()), range);
    /** check buffer accessor constructors for local
     */
    {
      queue.submit([&](cl::sycl::handler &h) {
        /** check (handler, range) constructor for read_write local
         */
        {
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::read_write,
                             cl::sycl::access::target::local,
                             cl::sycl::access::placeholder::false_t>
              a(range, h);

          if (a.get_size() != getElementsCount<dims>(range) * sizeof(T)) {
            FAIL(log,
                 "local accessor for read_write is not constructed "
                 "correctly (get_size)");
          }

          if (a.get_count() != getElementsCount<dims>(range)) {
            FAIL(log,
                 "local accessor for read_write is not constructed "
                 "correctly (get_count)");
          }
        }
        /** check buffer accessor constructors for atomic, only available in
         * local target
         */

        {
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::atomic,
                             cl::sycl::access::target::local,
                             cl::sycl::access::placeholder::false_t>
              a(range, h);

          if (a.get_size() != getElementsCount<dims>(range) * sizeof(T)) {
            FAIL(log,
                 "local accessor for atomic is not constructed "
                 "correctly (get_size)");
          }

          if (a.get_count() != getElementsCount<dims>(range)) {
            FAIL(log,
                 "local accessor for atomic is not constructed "
                 "correctly (get_count)");
          }
        }
        /** check accessor is Copy Constructible
         */
        {
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::read_write,
                             cl::sycl::access::target::local,
                             cl::sycl::access::placeholder::false_t>
              a(range, h);
          auto b{a};

          if (a.get_size() != b.get_size()) {
            FAIL(log, "local accessor is not copy constructible (get_size)");
          }

          if (a.get_count() != b.get_count()) {
            FAIL(log, "local accessor is not copy constructible (get_count)");
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

          if (a.get_size() != b.get_size()) {
            FAIL(log, "local accessor is not copy assignable (get_size)");
          }

          if (a.get_count() != b.get_count()) {
            FAIL(log, "local accessor is not copy assignable (get_count)");
          }
        }

        /** check accessor is Move Constructible
         */
        {
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::read_write,
                             cl::sycl::access::target::local,
                             cl::sycl::access::placeholder::false_t>
              a(range, h);
          auto b{std::move(a)};

          if (b.get_size() != getElementsCount<dims>(range) * sizeof(T)) {
            FAIL(log, "local accessor is not move constructible (get_size)");
          }

          if (b.get_count() != getElementsCount<dims>(range)) {
            FAIL(log, "local accessor is not move constructible (get_count)");
          }
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

          if (b.get_size() != getElementsCount<dims>(range) * sizeof(T)) {
            FAIL(log, "local accessor is not move assignable (get_size)");
          }

          if (b.get_count() != getElementsCount<dims>(range)) {
            FAIL(log, "local accessor is not move assignable (get_count)");
          }
        }

        /** dummy kernel as no kernel is required for these checks
         */
        h.single_task(dummy_functor{});
      });
      queue.wait_and_throw();
    }
  }
};

template <typename T>
class local_accessor_half_dims<T, 0> {
 public:
  static void check(util::logger &log, cl::sycl::queue &queue) {
    /** check buffer accessor constructors for local
     */
    {
      queue.submit([&](cl::sycl::handler &h) {
        /** check (buffer) constructor for read_write local
         */
        {
          cl::sycl::accessor<T, 0, cl::sycl::access::mode::read_write,
                             cl::sycl::access::target::local,
                             cl::sycl::access::placeholder::false_t>
              a(h);
          if (a.get_size() != sizeof(T)) {
            FAIL(log,
                 "local accessor for read_write is not constructed "
                 "correctly (get_size)");
          }

          if (a.get_count() != 1) {
            FAIL(log,
                 "local accessor for read_write is not constructed "
                 "correctly (get_count)");
          }
        }

        /** check (buffer) constructor for atomic local
         */
        {
          cl::sycl::accessor<T, 0, cl::sycl::access::mode::atomic,
                             cl::sycl::access::target::local,
                             cl::sycl::access::placeholder::false_t>
              a(h);
          if (a.get_size() != sizeof(T)) {
            FAIL(log,
                 "local accessor for atomic is not constructed "
                 "correctly (get_size)");
          }

          if (a.get_count() != 1) {
            FAIL(log,
                 "local accessor for atomic is not constructed "
                 "correctly (get_count)");
          }
        }

        /** check accessor is Copy Constructible
         */
        {
          cl::sycl::accessor<T, 0, cl::sycl::access::mode::read_write,
                             cl::sycl::access::target::local,
                             cl::sycl::access::placeholder::false_t>
              a(h);
          auto b{a};

          if (a.get_size() != b.get_size()) {
            FAIL(log, "local accessor is not copy constructible (get_size)");
          }

          if (a.get_count() != b.get_count()) {
            FAIL(log, "local accessor is not copy constructible (get_count)");
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

          if (a.get_size() != b.get_size()) {
            FAIL(log, "local accessor is not copy assignable (get_size)");
          }

          if (a.get_count() != b.get_count()) {
            FAIL(log, "local accessor is not copy assignable (get_count)");
          }
        }

        /** check accessor is Move Constructible
         */
        {
          cl::sycl::accessor<T, 0, cl::sycl::access::mode::read_write,
                             cl::sycl::access::target::local,
                             cl::sycl::access::placeholder::false_t>
              a(h);
          auto b{std::move(a)};

          if (b.get_size() != sizeof(T)) {
            FAIL(log, "local accessor is not move constructible (get_size)");
          }

          if (b.get_count() != 1) {
            FAIL(log, "local accessor is not move constructible (get_count)");
          }
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

          if (b.get_size() != sizeof(T)) {
            FAIL(log, "local accessor is not move constructible (get_size)");
          }

          if (b.get_count() != 1) {
            FAIL(log, "local accessor is not move constructible (get_count)");
          }

          /** dummy kernel as no kernel is required for these checks
           */
          h.single_task(dummy_functor{});
        }
      });
      queue.wait_and_throw();
    }
  }
};

template <typename T, size_t dims>
class image_accessor_half_dims {
 public:
  static void check(util::logger &log, cl::sycl::queue &queue) {
    int size = 32;
    cl::sycl::range<dims> range = getRange<dims>(size);
    std::vector<cl_float> data(range.size() * 4, 0.0f);
    cl::sycl::image<dims> image(data.data(),
                                cl::sycl::image_channel_order::rgba,
                                cl::sycl::image_channel_type::fp32, range);
    int elementSize = sizeof(cl_float) * 4;  // each element contains 4 channels

    /** check image accessor constructors for image
     */
    {
      queue.submit([&](cl::sycl::handler &h) {
        /** check (image, handler) constructor for reading image
         */
        {
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                             cl::sycl::access::target::image,
                             cl::sycl::access::placeholder::false_t>
              a(image, h);

          if (a.get_size() < getElementsCount<dims>(range) * elementSize) {
            FAIL(log,
                 "image accessor for read is not constructed correctly "
                 "(get_size)");
          }

          if (a.get_count() != getElementsCount<dims>(range)) {
            FAIL(log,
                 "image accessor for read is not constructed correctly "
                 "(get_count)");
          }
        }

        /** check (image, handler) constructor for writing image
         */
        {
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::write,
                             cl::sycl::access::target::image,
                             cl::sycl::access::placeholder::false_t>
              a(image, h);

          if (a.get_size() < getElementsCount<dims>(range) * elementSize) {
            FAIL(log,
                 "image accessor for write is not constructed correctly "
                 "(get_size)");
          }

          if (a.get_count() != getElementsCount<dims>(range)) {
            FAIL(log,
                 "image accessor for write is not constructed correctly "
                 "(get_count)");
          }
        }

        /** check (image, handler) constructor for discard_write image
         */
        {
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::discard_write,
                             cl::sycl::access::target::image,
                             cl::sycl::access::placeholder::false_t>
              a(image, h);

          if (a.get_size() < getElementsCount<dims>(range) * elementSize) {
            FAIL(log,
                 "image accessor for discard_write is not constructed "
                 "correctly (get_size)");
          }

          if (a.get_count() != getElementsCount<dims>(range)) {
            FAIL(log,
                 "image accessor for discard_write is not constructed "
                 "correctly (get_count)");
          }
        }

        /** check accessor is Copy Constructible
         */
        {
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                             cl::sycl::access::target::image,
                             cl::sycl::access::placeholder::false_t>
              a(image, h);
          auto b{a};

          if (a.get_size() < b.get_size()) {
            FAIL(log, "image accessor is not copy constructible (get_size)");
          }

          if (a.get_count() != b.get_count()) {
            FAIL(log, "image accessor is not copy constructible (get_count)");
          }
        }

        /** check accessor is Copy Assignable
         */
        {
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                             cl::sycl::access::target::image,
                             cl::sycl::access::placeholder::false_t>
              a(image, h);
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                             cl::sycl::access::target::image,
                             cl::sycl::access::placeholder::false_t>
              b(image, h);
          b = a;

          if (a.get_size() < b.get_size()) {
            FAIL(log, "image accessor is not copy assignable (get_size)");
          }

          if (a.get_count() != b.get_count()) {
            FAIL(log, "image accessor is not copy assignable (get_count)");
          }
        }

        /** check accessor is Move Constructible
         */
        {
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                             cl::sycl::access::target::image,
                             cl::sycl::access::placeholder::false_t>
              a(image, h);
          auto b{std::move(a)};

          if (b.get_size() < getElementsCount<dims>(range) * elementSize) {
            FAIL(log, "image accessor is not move constructible (get_size)");
          }

          if (b.get_count() != getElementsCount<dims>(range)) {
            FAIL(log, "image accessor is not move constructible (get_count)");
          }
        }

        /** check accessor is Move Assignable
         */
        {
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                             cl::sycl::access::target::image,
                             cl::sycl::access::placeholder::false_t>
              a(image, h);
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                             cl::sycl::access::target::image,
                             cl::sycl::access::placeholder::false_t>
              b(image, h);
          b = std::move(a);

          if (b.get_size() < getElementsCount<dims>(range) * elementSize) {
            FAIL(log, "image accessor is not move assignable (get_size)");
          }

          if (b.get_count() != getElementsCount<dims>(range)) {
            FAIL(log, "image accessor is not move assignable (get_count)");
          }
        }

        /** dummy kernel as no kernel is required for these checks
         */
        h.single_task(dummy_functor{});
      });
      queue.wait_and_throw();
    }

    /** check host_image accessor constructors for host_image
     */
    {
      /** check (image, handler) constructor for reading host_image
       */
      {
        cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                           cl::sycl::access::target::host_image,
                           cl::sycl::access::placeholder::false_t>
            a(image);

        if (a.get_size() < getElementsCount<dims>(range) * elementSize) {
          FAIL(log,
               "host image accessor for read is not constructed correctly "
               "(get_size)");
        }

        if (a.get_count() != getElementsCount<dims>(range)) {
          FAIL(log,
               "host image accessor for read is not constructed correctly "
               "(get_count)");
        }
      }

      /** check (image, handler) constructor for writing host_image
       */
      {
        cl::sycl::accessor<T, dims, cl::sycl::access::mode::write,
                           cl::sycl::access::target::host_image,
                           cl::sycl::access::placeholder::false_t>
            a(image);

        if (a.get_size() < getElementsCount<dims>(range) * elementSize) {
          FAIL(log,
               "host image accessor for write is not constructed correctly "
               "(get_size)");
        }

        if (a.get_count() != getElementsCount<dims>(range)) {
          FAIL(log,
               "host image accessor for write is not constructed correctly "
               "(get_count)");
        }
      }

      /** check (image, handler) constructor for discard_write host_image
       */
      {
        cl::sycl::accessor<T, dims, cl::sycl::access::mode::discard_write,
                           cl::sycl::access::target::host_image,
                           cl::sycl::access::placeholder::false_t>
            a(image);

        if (a.get_size() < getElementsCount<dims>(range) * elementSize) {
          FAIL(log,
               "host image accessor for discard_write is not constructed "
               "correctly (get_size)");
        }

        if (a.get_count() != getElementsCount<dims>(range)) {
          FAIL(log,
               "host image accessor for discard_write is not constructed "
               "correctly (get_count)");
        }
      }

      /** check accessor is Copy Constructible
       */
      {
        cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                           cl::sycl::access::target::host_image,
                           cl::sycl::access::placeholder::false_t>
            a(image);
        auto b{a};

        if (a.get_size() < b.get_size()) {
          FAIL(log, "host image accessor is not copy constructible (get_size)");
        }

        if (a.get_count() != b.get_count()) {
          FAIL(log,
               "host image accessor is not copy constructible (get_count)");
        }
      }

      /** check accessor is Copy Assignable
       */
      {
        cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                           cl::sycl::access::target::host_image,
                           cl::sycl::access::placeholder::false_t>
            a(image);
        cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                           cl::sycl::access::target::host_image,
                           cl::sycl::access::placeholder::false_t>
            b(image);
        b = a;

        if (a.get_size() < b.get_size()) {
          FAIL(log, "host image accessor is not copy assignable (get_size)");
        }

        if (a.get_count() != b.get_count()) {
          FAIL(log, "host image accessor is not copy assignable (get_count)");
        }
      }

      /** check accessor is Move Constructible
       */
      {
        cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                           cl::sycl::access::target::host_image,
                           cl::sycl::access::placeholder::false_t>
            a(image);
        auto b{std::move(a)};

        if (b.get_size() < getElementsCount<dims>(range) * elementSize) {
          FAIL(log, "host image accessor is not move constructible (get_size)");
        }

        if (b.get_count() != getElementsCount<dims>(range)) {
          FAIL(log,
               "host image accessor is not move constructible (get_count)");
        }
      }

      /** check accessor is Move Assignable
       */
      {
        cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                           cl::sycl::access::target::host_image,
                           cl::sycl::access::placeholder::false_t>
            a(image);
        cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                           cl::sycl::access::target::host_image,
                           cl::sycl::access::placeholder::false_t>
            b(image);
        b = std::move(a);

        if (b.get_size() < getElementsCount<dims>(range) * elementSize) {
          FAIL(log, "host image accessor is not move assignable (get_size)");
        }

        if (b.get_count() != getElementsCount<dims>(range)) {
          FAIL(log, "host image accessor is not move assignable (get_count)");
        }
      }
    }
  }
};

template <typename T, size_t dims>
class image_array_accessor_half_dims {
 public:
  static void check(util::logger &log, cl::sycl::queue &queue) {
    int size = 32;
    cl::sycl::range<dims + 1> range = getRange<dims + 1>(size);
    std::vector<cl_float> data(range.size() * 4, 0.0f);
    cl::sycl::image<dims + 1> image(data.data(),
                                    cl::sycl::image_channel_order::rgba,
                                    cl::sycl::image_channel_type::fp32, range);
    int elementSize = sizeof(cl_float) * 4;  // each element contains 4 channels

    /** check image array accessor constructors for image
     */
    {
      queue.submit([&](cl::sycl::handler &h) {
        /** check (image, handler) constructor for reading image
         */
        {
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                             cl::sycl::access::target::image_array,
                             cl::sycl::access::placeholder::false_t>
              a(image, h);

          if (a.get_size() < getElementsCount<dims + 1>(range) * elementSize) {
            FAIL(log,
                 "image array accessor for read is not constructed correctly "
                 "(get_size)");
          }

          if (a.get_count() != getElementsCount<dims + 1>(range)) {
            FAIL(log,
                 "image array accessor for read is not constructed correctly "
                 "(get_count)");
          }
        }

        /** check (image, handler) constructor for writing image
         */
        {
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::write,
                             cl::sycl::access::target::image_array,
                             cl::sycl::access::placeholder::false_t>
              a(image, h);

          if (a.get_size() < getElementsCount<dims + 1>(range) * elementSize) {
            FAIL(log,
                 "image array accessor for write is not constructed correctly "
                 "(get_size)");
          }

          if (a.get_count() != getElementsCount<dims + 1>(range)) {
            FAIL(log,
                 "image array accessor for write is not constructed correctly "
                 "(get_count)");
          }
        }

        /** check (image, handler) constructor for discard_write image
         */
        {
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::discard_write,
                             cl::sycl::access::target::image_array,
                             cl::sycl::access::placeholder::false_t>
              a(image, h);

          if (a.get_size() < getElementsCount<dims + 1>(range) * elementSize) {
            FAIL(log,
                 "image array accessor for discard_write is not constructed "
                 "correctly (get_size)");
          }

          if (a.get_count() != getElementsCount<dims + 1>(range)) {
            FAIL(log,
                 "image array accessor for discard_write is not constructed "
                 "correctly (get_count)");
          }
        }

        /** check accessor is Copy Constructible
         */
        {
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                             cl::sycl::access::target::image_array,
                             cl::sycl::access::placeholder::false_t>
              a(image, h);
          auto b{a};

          if (a.get_size() < b.get_size()) {
            FAIL(log,
                 "image array accessor is not copy constructible (get_size)");
          }

          if (a.get_count() != b.get_count()) {
            FAIL(log,
                 "image array accessor is not copy constructible (get_count)");
          }
        }

        /** check accessor is Copy Assignable
         */
        {
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                             cl::sycl::access::target::image_array,
                             cl::sycl::access::placeholder::false_t>
              a(image, h);
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                             cl::sycl::access::target::image_array,
                             cl::sycl::access::placeholder::false_t>
              b(image, h);
          b = a;

          if (a.get_size() < b.get_size()) {
            FAIL(log, "image array accessor is not copy assignable (get_size)");
          }

          if (a.get_count() != b.get_count()) {
            FAIL(log,
                 "image array accessor is not copy assignable (get_count)");
          }
        }

        /** check accessor is Move Constructible
         */
        {
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                             cl::sycl::access::target::image_array,
                             cl::sycl::access::placeholder::false_t>
              a(image, h);
          auto b{std::move(a)};

          if (b.get_size() < getElementsCount<dims + 1>(range) * elementSize) {
            FAIL(log,
                 "image array accessor is not move constructible (get_size)");
          }

          if (b.get_count() != getElementsCount<dims + 1>(range)) {
            FAIL(log,
                 "image array accessor is not move constructible (get_count)");
          }
        }

        /** check accessor is Move Assignable
         */
        {
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                             cl::sycl::access::target::image_array,
                             cl::sycl::access::placeholder::false_t>
              a(image, h);
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                             cl::sycl::access::target::image_array,
                             cl::sycl::access::placeholder::false_t>
              b(image, h);
          b = std::move(a);

          if (b.get_size() < getElementsCount<dims + 1>(range) * elementSize) {
            FAIL(log, "image array accessor is not move assignable (get_size)");
          }

          if (b.get_count() != getElementsCount<dims + 1>(range)) {
            FAIL(log,
                 "image array accessor is not move assignable (get_count)");
          }
        }

        /** dummy kernel as no kernel is required for these checks
         */
        h.single_task(dummy_functor{});
      });
      queue.wait_and_throw();
    }
  }
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

  void check_all_dims(util::logger &log, cl::sycl::queue &queue) {}

  template <typename T>
  void checkBufferAndLocal(util::logger &log, cl::sycl::queue &queue) {
    buffer_accessor_half_dims<cl::sycl::cl_half, 0>::check(log, queue);
    buffer_accessor_half_dims<cl::sycl::cl_half, 1>::check(log, queue);
    buffer_accessor_half_dims<cl::sycl::cl_half, 2>::check(log, queue);
    buffer_accessor_half_dims<cl::sycl::cl_half, 3>::check(log, queue);

    placeholder_accessor_half_dims<cl::sycl::cl_half, 0>::check(log, queue);
    placeholder_accessor_half_dims<cl::sycl::cl_half, 1>::check(log, queue);
    placeholder_accessor_half_dims<cl::sycl::cl_half, 2>::check(log, queue);
    placeholder_accessor_half_dims<cl::sycl::cl_half, 3>::check(log, queue);

    local_accessor_half_dims<cl::sycl::cl_half, 0>::check(log, queue);
    local_accessor_half_dims<cl::sycl::cl_half, 1>::check(log, queue);
    local_accessor_half_dims<cl::sycl::cl_half, 2>::check(log, queue);
    local_accessor_half_dims<cl::sycl::cl_half, 3>::check(log, queue);
  }

  void checkImage(util::logger &log, cl::sycl::queue &queue) {
    image_accessor_half_dims<cl::sycl::cl_half4, 1>::check(log, queue);
    image_accessor_half_dims<cl::sycl::cl_half4, 2>::check(log, queue);
    image_accessor_half_dims<cl::sycl::cl_half4, 3>::check(log, queue);
    image_array_accessor_half_dims<cl::sycl::cl_half4, 1>::check(log, queue);
    image_array_accessor_half_dims<cl::sycl::cl_half4, 2>::check(log, queue);
  }
  /** execute this test
   */
  void run(util::logger &log) override {
    try {
      auto queue = util::get_cts_object::queue();

      /** check accessor constructors for cl_half
       */
	  checkBufferAndLocal<cl::sycl::cl_half>(log, queue);

      /** check accessor constructors for cl_half
       */
      checkBufferAndLocal<cl::sycl::cl_half2>(log, queue);

      /** check accessor constructors for cl_half
       */
      checkBufferAndLocal<cl::sycl::cl_half3>(log, queue);

      /** check accessor constructors for cl_half
       */
      checkBufferAndLocal<cl::sycl::cl_half4>(log, queue);

      /** check accessor constructors for cl_half
       */
      checkBufferAndLocal<cl::sycl::cl_half8>(log, queue);

      /** check accessor constructors for cl_half
       */
      checkBufferAndLocal<cl::sycl::cl_half16>(log, queue);

      /** check image accessor half variant
       */
      checkImage(log, queue);

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
