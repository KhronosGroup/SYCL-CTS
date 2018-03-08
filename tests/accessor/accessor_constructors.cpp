/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#define TEST_NAME accessor_constructors

#include "../common/common.h"
#include "accessor_utility.h"

namespace TEST_NAMESPACE {

/** unique dummy_functor per file
 *  this is a hack until the CMake script is fixed; kill both the alias and the
 *  dummy class once it is fixed
 */
class dummy_accessor_constructors {};
using dummy_functor = ::dummy_functor<dummy_accessor_constructors>;

using namespace sycl_cts;

template <typename T, int dims, int size, cl::sycl::access::mode mode,
          cl::sycl::access::target target,
          cl::sycl::access::placeholder placeholder, typename... Args,
          REQUIRES(is_buffer<target>::value)>
void check_implemenation(
    util::logger &log, const std::string &op,
    cl::sycl::accessor<T, dims, mode, target, placeholder> &a,
    const std::tuple<Args...> &args) {
  if (!(std::get<0>(args) == a.get_size())) {
    FAIL(log, op + "(get_size failed)");
  }

  if (!(std::get<1>(args) == a.get_count())) {
    FAIL(log, op + "(get_count failed)");
  }

  if (!(std::get<2>(args) == a.get_range())) {
    FAIL(log, op + "(get_range failed)");
  }

  if (!(std::get<3>(args) == a.get_offset())) {
    FAIL(log, op + "(get_offset failed)");
  }
}

template <typename T, int dims, int size, cl::sycl::access::mode mode,
          cl::sycl::access::target target,
          cl::sycl::access::placeholder placeholder, typename... Args,
          REQUIRES(is_image<target>::value)>
void check_implemenation(
    util::logger &log, const std::string &op,
    const cl::sycl::accessor<T, dims, mode, target, placeholder> &a,
    const std::tuple<Args...> &args) {
  if (!(std::get<0>(args) == a.get_size())) {
    FAIL(log, op + "(get_size failed)");
  }

  if (!(std::get<1>(args) == a.get_count())) {
    FAIL(log, op + "(get_count failed)");
  }
}

template <typename A, typename... Args>
void check_special_members(util::logger &log, A &a1,
                           const std::tuple<Args...> &args) {
  /** check accessor is MoveConstructible
  */
  auto a2 = std::move(a1);
  check_implemenation(log, "accessor is not move constructible", a2, args);

  /** check accessor is MoveAssignable
  */
  a1 = std::move(a2);
  check_implemenation(log, "accessor is not move assignable", a1, args);

  /** check accessor is Swappable
  */
  {
    using std::swap;
    swap(a1, a2);
    check_implemenation(log, "accessor is not swappable", a2, args);
  }

  /** check accessor is CopyConstructible
  */
  auto a3 = a2;
  check_implemenation(log, "accessor is not copy constructible", a3, args);

  /** check accessor is CopyAssignable
  */
  a1 = a3;
  check_implemenation(log, "accessor is not copy assignable", a1, args);
}

template <typename A>
void check_equality_comparable(util::logger &log, const A &a) {
  /** check for reflexivity
  */
  if (!(a == a)) {
    FAIL(log,
         "accessor is not equality-comparable (operator== reflexivity "
         "failed)");
  } else if (a != a) {
    FAIL(log,
         "accessor is not equality-comparable (operator!= reflexivity "
         "failed)");
  }

  /** check for symmetry
  */
  auto b = a;
  if (!(a == b)) {
    FAIL(log,
         "accessor is not equality-comparable (operator==, copy constructor)");
  } else if (!(b == a)) {
    FAIL(log,
         "accessor is not equality-comparable (operator== symmetry failed)");
  } else if (a != b) {
    FAIL(log,
         "accessor is not equality-comparable (operator!=, copy constructor)");
  } else if (b != a) {
    FAIL(log,
         "accessor is not equality-comparable (operator!= symmetry failed)");
  }

  /** check for transitivity
  */
  auto c = b;
  if (!(a == c)) {
    FAIL(log,
         "accessor is not equality-comparable (operator== transitivity "
         "failed)");
  } else if (a != c) {
    FAIL(log,
         "accessor is not equality-comparable (operator!= transitivity "
         "failed)");
  }

  /** check copy-assignment
  */
  b = a;
  if (!(b == a)) {
    FAIL(log,
         "accessor is not equality-comparable (operator==, copy assignment)");
  } else if (a != b) {
    FAIL(log,
         "accessor is not equality-comparable (operator!=, copy assignment)");
  }

  std::hash<A> hasher;

  if (hasher(a) != hasher(b)) {
    FAIL(log, "accessor hashing of equal failed");
  }
}

template <typename T, int dims, int size, cl::sycl::access::mode mode,
          cl::sycl::access::target target,
          cl::sycl::access::placeholder placeholder =
              cl::sycl::access::placeholder::false_t>
class buffer_accessor_constructors {
 public:
  void operator()(util::logger &log, cl::sycl::queue &queue,
                  cl::sycl::range<dims> &range) {
    auto data = [] {
      auto data = std::array<T, size>{};
      std::fill(data.begin(), data.end(), T{});
      return data;
    }();
    cl::sycl::buffer<T, dims> buffer(data.data(), range);
    const auto offset = cl::sycl::id<dims>(range / 2);
    const auto r = range / 2;

    const auto exposed_interface =
        [&](const cl::sycl::accessor<T, dims, mode, target> &a) {
          return std::make_tuple(a.get_size(), a.get_count(), a.get_range(),
                                 a.get_offset());
        };

    /** check host_buffer-only constructors
    */
    if_constexpr<(target == cl::sycl::access::target::host_buffer)>(
        [&] {
          /** check (buffer) constructor
          */
          auto a1 = make_accessor<T, dims, mode, target>(buffer);

          /** check (buffer, range, offset) constructor
          */
          auto a2 = make_accessor<T, dims, mode, target>(buffer, r, offset);

          /** check host_buffer accessor is Copyable
          */
          check_special_members(log, a1, exposed_interface(a1));

          /** check host_buffer accessor is EqualityComparable and hashable
          */
          check_equality_comparable(log, a1);
        },
        [&] {
          cl::sycl::buffer<T, dims> buffer2(data.data(), range);
          queue.submit([&](cl::sycl::handler &h) {
            /** check (buffer, handler) constructor
            */
            auto a1 =
                make_accessor<T, dims, mode, target, placeholder>(buffer, h);

            /** check (buffer, handler, range, offset) constructor
            */
            auto a2 = make_accessor<T, dims, mode, target, placeholder>(
                buffer, h, r, offset);

            /** check accessor is Copyable
            */
            check_special_members(log, a1, exposed_interface(a1));

            /* check accessor is EqualityComparable and hashable
            */
            check_equality_comparable(log, a1);

            /** dummy kernel as no kernel is required for these checks
            */
            h.single_task(dummy_functor());
          });
        });
  }
};

template <typename T, int dims, int size, cl::sycl::access::mode mode,
          cl::sycl::access::target target>
class local_accessor_constructors {
 public:
  void operator()(util::logger &log, cl::sycl::queue &queue,
                  cl::sycl::range<dims> range) {
    queue.submit([&](cl::sycl::handler &h) {
      /** check (range, handler) constructor
      */
      auto a1 = make_accessor<T, dims, mode, target>(range, h);

      /** check local accessor is Copyable
      */
      check_special_members(log, a1, exposed_interface(a1));

      /** check accessor is EqualityComparable and hashable
      */
      check_equality_comparable(log, a1);

      /** dummy kernel as no kernel is required for these checks
      */
      h.single_task(dummy_functor());
    });
  }
};

template <typename T, int dims, int size, cl::sycl::access::mode mode,
          cl::sycl::access::target target>
class image_accessor_constructors {
 public:
  void operator()(util::logger &log, cl::sycl::queue &queue,
                  cl::sycl::range<dims> &range) {
    auto data = [] {
      auto data = std::array<char, size>{};
      std::fill(data.begin(), data.end(), static_cast<char>(0));
      return data;
    }();
    auto image = cl::sycl::image<(
        (target == cl::sycl::access::target::image_array) ? (dims + 1) : dims)>(
        data.data(), cl::sycl::image_channel_order::rgba,
        cl::sycl::image_channel_type::unsigned_int8, range);

    const auto exposed_interface =
        [&](const cl::sycl::accessor<T, dims, mode, target> &a) {
          return std::make_tuple(a.get_size(), a.get_count());
        };
    /** check target-specific constructors
    */
    if_constexpr<(target == cl::sycl::access::target::host_image)>(
        [&] {
          /** check (image) constructor
          */
          auto a = make_accessor<T, dims, mode, target>(image);

          /** check host image accessor is Copyable
          */
          check_special_members(log, a, exposed_interface(a));

          /** check host image accessor is EqualityComparable and hashable
          */
          check_equality_comparable(log, a, exposed_interface(a));
        },
        [&] {
          queue.submit([&](cl::sycl::handler &handler) {
            /** check (image, handler) constructor
            */
            auto a = make_accessor<T, dims, mode, target>(image, handler);

            /** check device image accessor is Copyable
            */
            check_special_members(log, a);

            /** check device image accessor is EqualityComparable and hashable
            */
            check_equality_comparable(log, a);

            /** dummy kernel as no kernel is required for these checks
            */
            handler.single_task(dummy_functor());
          });
        });
  }
};

template <typename T, int dims, int size, cl::sycl::access::target target,
          cl::sycl::access::placeholder placeholder =
              cl::sycl::access::placeholder::false_t>
class buffer_accessor_modes {
 public:
  void operator()(util::logger &log, cl::sycl::queue &queue,
                  cl::sycl::range<dims> &range) {
    /** check buffer accessor constructors for read
    */
    buffer_accessor_constructors<T, dims, size, cl::sycl::access::mode::read,
                                 target>
        readTests;
    readTests(log, queue, range);

    /** check buffer accessor constructors for write
    */
    buffer_accessor_constructors<T, dims, size, cl::sycl::access::mode::write,
                                 target>
        writeTests;
    writeTests(log, queue, range);

    /** check buffer accessor constructors for read_write
    */
    buffer_accessor_constructors<T, dims, size,
                                 cl::sycl::access::mode::read_write, target>
        readWriteTests;
    readWriteTests(log, queue, range);

    /** check buffer accessor constructors for discard_write
    */
    buffer_accessor_constructors<T, dims, size,
                                 cl::sycl::access::mode::discard_write, target>
        discardWriteTests;
    discardWriteTests(log, queue, range);

    /** check buffer accessor constructors for discard_read_write
    */
    buffer_accessor_constructors<
        T, dims, size, cl::sycl::access::mode::discard_read_write, target>
        discardReadWriteTests;
    discardReadWriteTests(log, queue, range);

    /** check target-specific constructors
     */
    if_constexpr<target == cl::sycl::access::target::constant_buffer>([&] {
      /** check buffer accessor constructors for read
      */
      buffer_accessor_constructors<T, dims, size, cl::sycl::access::mode::read,
                                   target>
          readTests;
      readTests(log, queue, range);
    });

    if_constexpr<target == cl::sycl::access::target::host_buffer>([&] {
      /** check buffer accessor constructors for read
      */
      buffer_accessor_constructors<T, dims, size, cl::sycl::access::mode::read,
                                   target>
          readTests;
      readTests(log, queue, range);

      /** check buffer accessor constructors for write
      */
      buffer_accessor_constructors<T, dims, size, cl::sycl::access::mode::write,
                                   target>
          writeTests;
      writeTests(log, queue, range);

      /** check buffer accessor constructors for read_write
      */
      buffer_accessor_constructors<T, dims, size,
                                   cl::sycl::access::mode::read_write, target>
          readWriteTests;
      readWriteTests(log, queue, range);
    });
  }
};

template <typename T, int dims, int size>
class local_accessor_modes {
 public:
  void operator()(util::logger &log, cl::sycl::queue &queue,
                  cl::sycl::range<dims> &range) {
    constexpr auto target = cl::sycl::access::target::local;
    /** check local accessor constructors for read_write
    */
    local_accessor_constructors<T, dims, size,
                                cl::sycl::access::mode::read_write, target>
        readWriteTests;
    readWriteTests(log, queue, range);
  }
};

/**
 * @brief  Checks that image accessors can be constructed using various modes.
 *
 *         Note: OpenCL 1.2 exclusively permits only read access mode or write
 *         access, so this test does not consider non-mode::read and
 *         non-mode::write.
 * @tparam T      The type of the object the accessor points to.
 * @tparam dims   The number of dimensions the accessor has.
 * @tparam size   The number of elements in each dimension.
 * @tparam target Determines whether this is an image accessor, etc.
 */
template <typename T, int dims, int size, cl::sycl::access::target target>
class image_accessor_modes {
 public:
  void operator()(util::logger &log, cl::sycl::queue &queue,
                  cl::sycl::range<dims> &range) {
    /** check image accessor constructors for read
    */
    image_accessor_constructors<T, dims, size, cl::sycl::access::mode::read,
                                target>
        readTests;
    readTests(log, queue, range);

    /** check image accessor constructors for write
    */
    image_accessor_constructors<T, dims, size, cl::sycl::access::mode::write,
                                target>
        writeTests;
    writeTests(log, queue, range);
  }
};

template <typename T, int dims, int size, cl::sycl::access::target target>
class buffer_accessor_placeholders {
 public:
  void operator()(util::logger &log, cl::sycl::queue &queue,
                  cl::sycl::range<dims> &range) {
    using cl::sycl::access::placeholder;
    buffer_accessor_modes<T, dims, size, target, placeholder::true_t>{}(
        log, queue, range);

    buffer_accessor_modes<T, dims, size, target, placeholder::false_t>{}(
        log, queue, range);
  }
};

template <typename T, int dims, int size>
class buffer_accessor_targets {
 public:
  void operator()(util::logger &log, cl::sycl::queue &queue,
                  cl::sycl::range<dims> &range) {
    /** check buffer accessor constructors for global_buffer
    */
    buffer_accessor_placeholders<T, dims, size,
                                 cl::sycl::access::target::global_buffer>
        globalBufferTests;
    globalBufferTests(log, queue, range);

    /** check buffer accessor constructors for constant_buffer
    */
    buffer_accessor_placeholders<T, dims, size,
                                 cl::sycl::access::target::constant_buffer>
        constantTests;
    constantTests(log, queue, range);

    /** check buffer accessor constructors for host_buffer
    */
    buffer_accessor_modes<T, dims, size, cl::sycl::access::target::host_buffer>
        hostBufferTests;
    hostBufferTests(log, queue, range);
  }
};

template <typename T, int dims, int size>
class local_accessor_targets {
 public:
  void operator()(util::logger &log, cl::sycl::queue &queue,
                  cl::sycl::range<dims> &range) {
    /** check local accessor constructor
    */
    local_accessor_modes<T, dims, size> localTests;
    localTests(log, queue, range);
  }
};

template <typename T, int dims, int size>
class image_accessor_targets {
 public:
  void operator()(util::logger &log, cl::sycl::queue &queue,
                  cl::sycl::range<dims> &range) {
    /** check image accessor constructors for image
    */
    image_accessor_modes<T, dims, size, cl::sycl::access::target::image>
        imageTests;
    imageTests(log, queue, range);

    /** check image accessor constructors for host_image
    */
    image_accessor_modes<T, dims, size, cl::sycl::access::target::host_image>
        hostImageTests;
    hostImageTests(log, queue, range);

    /** check image array constructors for image_array
    */
    image_accessor_modes<T, dims, size, cl::sycl::access::target::image_array>
        imageArrayTests;
    imageArrayTests(log, queue, range);
  }
};

template <typename T>
class buffer_accessor_dims {
 public:
  void operator()(util::logger &log, cl::sycl::queue &queue) {
    constexpr int size = 32;

    /** check buffer accessor constructors for 1 dimension
    */
    cl::sycl::range<1> range1d(size);
    buffer_accessor_targets<T, 1, size> acc1d;
    acc1d(log, queue, range1d);

    /** check buffer accessor constructors for 2 dimension
    */
    cl::sycl::range<2> range2d(size, size);
    buffer_accessor_targets<T, 2, (size * size)> acc2d;
    acc2d(log, queue, range2d);

    /** check buffer accessor constructors for 3 dimension
    */
    cl::sycl::range<3> range3d(size, size, size);
    buffer_accessor_targets<T, 3, (size * size * size)> acc3d;
    acc3d(log, queue, range3d);
  }
};

template <typename T>
class local_accessor_dims {
 public:
  static constexpr auto size = 32;
  void operator()(util::logger &log, cl::sycl::queue &queue) {
    /** check local accessor constructors for 1 dimension
    */
    check(log, queue, cl::sycl::range<1>(size));

    /** check local accessor constructors for 2 dimensions
    */
    check(log, queue, cl::sycl::range<2>(size, size));

    /** check local accessor constructors for 3 dimensions
    */
    check(log, queue, cl::sycl::range<3>(size, size, size));
  }

 private:
  template <int dim>
  void check(util::logger &log, cl::sycl::queue &queue,
             cl::sycl::range<dim> r) {
    auto a = local_accessor_targets<T, dim, size>{};
    a(log, queue, r);
  }
};

template <typename T>
class image_accessor_dims {
 public:
  void operator()(util::logger &log, cl::sycl::queue &queue) {
    constexpr int count = 32;
    constexpr int size = count * 4;

    /** check image accessor constructors for 1 dimension
    */
    cl::sycl::range<1> range1d(count);
    image_accessor_targets<T, 1, size> acc1d;
    acc1d(log, queue, range1d);

    /** check image accessor constructors for 2 dimension
    */
    cl::sycl::range<2> range2d(count / 4, 4);
    image_accessor_targets<T, 2, size> acc2d;
    acc2d(log, queue, range2d);

    /** check image accessor constructors for 3 dimension
    */
    cl::sycl::range<3> range3d(count / 8, 4, 2);
    image_accessor_targets<T, 3, size> acc3d;
    acc3d(log, queue, range3d);
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

  /** execute this test
   */
  void run(util::logger &log) override {
    try {
      auto queue = util::get_cts_object::queue();

      /** check all buffer accessor constructors
      */
      accessor_check<buffer_accessor_dims>(log, queue);

      /** check all local accessor constructors
      */
      accessor_check<local_accessor_dims>(log, queue);

      /** check image accessor constructors for int4
      */
      image_accessor_dims<cl::sycl::int4>()(log, queue);

      /** check image accessor constructors for uint4
      */
      image_accessor_dims<cl::sycl::uint4>()(log, queue);

      /** check image accessor constructors for float4
      */
      image_accessor_dims<cl::sycl::float4>()(log, queue);

      queue.wait_and_throw();
    } catch (const cl::sycl::exception &e) {
      log_exception(log, e);
      cl::sycl::string_class errorMsg =
          "a SYCL exception was caught: " + cl::sycl::string_class(e.what());
      FAIL(log, errorMsg.c_str());
    }
  }

 private:
  template <template <typename> class A>
  void accessor_check(util::logger &log, cl::sycl::queue &queue) {
    /** check accessor constructors for int
       */
    A<int>()(log, queue);

    /** check accessor constructors for float
    */
    A<float>()(log, queue);

    /** check accessor constructors for double
    */
    A<double>()(log, queue);

    /** check accessor constructors for char
    */
    A<char>()(log, queue);

    /** check accessor constructors for vec
    */
    A<cl::sycl::int2>()(log, queue);

    /** check accessor constructors for user_struct
    */
    A<user_struct>()(log, queue);
  }
};

// register this test with the test_collection
util::test_proxy<TEST_NAME> proxy;

} /* namespace accessor_constructors__ */
