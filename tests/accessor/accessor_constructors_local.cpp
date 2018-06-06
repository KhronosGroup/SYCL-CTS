/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2018 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#define TEST_NAME accessor_constructors_local

#include "../common/common.h"
#include "accessor_constructors_utility.h"

namespace TEST_NAMESPACE {
/** unique dummy_functor per file
 *  this is a hack until the CMake script is fixed; kill both the alias and the
 *  dummy class once it is fixed
 */
class dummy_accessor_constructors_local {};
using dummy_functor = ::dummy_functor<dummy_accessor_constructors_local>;

template <typename T, size_t dims>
class local_accessor_dims {
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
        h.parallel_for_work_group(cl::sycl::range<1>(1), dummy_functor{});
      });
      queue.wait_and_throw();
    }
  }
};

template <typename T>
class local_accessor_dims<T, 0> {
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
          h.parallel_for_work_group(cl::sycl::range<1>(1), dummy_functor{});
        }
      });
      queue.wait_and_throw();
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
    local_accessor_dims<T, 0>::check(log, queue);
    local_accessor_dims<T, 1>::check(log, queue);
    local_accessor_dims<T, 2>::check(log, queue);
    local_accessor_dims<T, 3>::check(log, queue);
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
