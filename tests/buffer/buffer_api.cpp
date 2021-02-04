/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME buffer_api

namespace TEST_NAMESPACE {
using namespace sycl_cts;

/** empty_kernel.
 * Empty kernel, required since command groups
 * are required to have a kernel.
 */
class empty_kernel {
 public:
  void operator()() const {}
};

/*!
@brief used to calculate the ranges based on the dimensionality of the buffer
*/
template <size_t dims>
void precalculate(cl::sycl::range<dims>& rangeIn,
                  cl::sycl::range<dims>& rangeOut, size_t& elementsCount,
                  unsigned elementsIn, unsigned elementsOut);

template <>
void precalculate<1>(cl::sycl::range<1>& rangeIn, cl::sycl::range<1>& rangeOut,
                     size_t& elementsCount, unsigned elementsIn,
                     unsigned elementsOut) {
  rangeIn = cl::sycl::range<1>(elementsIn);
  rangeOut = cl::sycl::range<1>(elementsOut);
  elementsCount = elementsOut;
}

template <>
void precalculate<2>(cl::sycl::range<2>& rangeIn, cl::sycl::range<2>& rangeOut,
                     size_t& elementsCount, unsigned elementsIn,
                     unsigned elementsOut) {
  rangeIn = cl::sycl::range<2>(elementsIn, elementsIn);
  rangeOut = cl::sycl::range<2>(elementsOut, elementsIn);
  elementsCount = (elementsOut * elementsIn);
}

template <>
void precalculate<3>(cl::sycl::range<3>& rangeIn, cl::sycl::range<3>& rangeOut,
                     size_t& elementsCount, unsigned elementsIn,
                     unsigned elementsOut) {
  rangeIn = cl::sycl::range<3>(elementsIn, elementsIn, elementsIn);
  rangeOut = cl::sycl::range<3>(elementsOut, elementsIn, elementsIn);
  elementsCount = (elementsOut * elementsIn * elementsIn);
}

/*!
@brief Used to produce and test the reinterpreted buffer denoted by the template
arguments. It does so by using the provided data array as a multidimensional
buffer
@tparam TIn the type of the original buffer
@tparam TOut the type of the reinterpreted buffer
*/
template <typename TIn, typename TOut>
class test_buffer_reinterpret {
 public:
  unsigned elementsIn, elementsOut;
  /*!
  @brief constructor
  @param elementsIn the dimension used to create the range for the original
  buffer
  @param elementsOut the dimension used to create the range for the
  reinterpreted
  buffer
  */
  test_buffer_reinterpret(unsigned ElementsIn, unsigned ElementsOut)
      : elementsIn(ElementsIn), elementsOut(ElementsOut) {}

  template <size_t dims>
  void check(TIn* data, util::logger& log) {
    cl::sycl::range<dims> rangeIn = getRange<dims>(1);
    cl::sycl::range<dims> rangeOut = getRange<dims>(1);
    size_t elementsCount = 0;
    precalculate<dims>(rangeIn, rangeOut, elementsCount, elementsIn,
                       elementsOut);

    cl::sycl::buffer<TIn, dims> a(data, rangeIn);
    auto r = a.template reinterpret<TOut, dims>(rangeOut);

    if (r.get_size() != (elementsCount * sizeof(TOut))) {
      FAIL(log, "Reinterpretation failed! The buffers have different size");
    }
  }
};

/**
 * Generic buffer API test function
 */
template <typename T, int size, int dims>
void test_buffer(util::logger& log, cl::sycl::range<dims>& r,
                 cl::sycl::id<dims>& i) {
  try {
    cl::sycl::unique_ptr_class<T[]> data(new T[size]);
    std::fill(data.get(), (data.get() + size), 0);

    // Create a default offset with indices 0.
    cl::sycl::id<dims> offset;

    /* create a SYCL buffer from the host buffer */
    cl::sycl::buffer<T, dims> buf(data.get(), r);

    /* check the buffer returns a range */
    auto ret_range = buf.get_range();
    check_return_type<cl::sycl::range<dims>>(log, ret_range,
                                             "cl::sycl::buffer::get_range()");

    /* Check alias types */
    {
      {
        check_type_existence<typename cl::sycl::buffer<T, dims>::value_type>
            typeCheck;
      }
      {
        check_type_existence<typename cl::sycl::buffer<T, dims>::reference>
            typeCheck;
      }
      {
        check_type_existence<
            typename cl::sycl::buffer<T, dims>::const_reference>
            typeCheck;
      }
      {
        check_type_existence<typename cl::sycl::buffer<
            T, dims, cl::sycl::buffer_allocator>::allocator_type>
            typeCheck;
      }
    }

    /* Check that ret_range is the correct size */
    for (int i = 0; i < dims; ++i) {
      if (ret_range[i] != r[i]) {
        FAIL(log,
             "cl::sycl::buffer::get_range does not return "
             "the correct range size!");
      }
    }

    /* check the buffer returns the correct element count */
    auto count = buf.get_count();
    check_return_type<size_t>(log, count, "cl::sycl::buffer::get_count()");

    if (count != size) {
      FAIL(log,
           "cl::sycl::buffer::get_count() does not return "
           "the correct number of elements");
    }

    /* check the buffer returns the correct byte size */
    auto ret_size = buf.get_size();
    check_return_type<size_t>(log, ret_size, "cl::sycl::buffer::get_size()");

    if (ret_size != size * sizeof(T)) {
      FAIL(log,
           "cl::sycl::buffer::get_size() does not return "
           "the correct size of the buffer");
    }

    auto q = util::get_cts_object::queue();

    /* check the buffer returns the correct type of accessor */
    q.submit([&](cl::sycl::handler& cgh) {
      auto acc =
          buf.template get_access<cl::sycl::access::mode::read_write>(cgh);
      check_return_type<
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::read_write,
                             cl::sycl::access::target::global_buffer>>(
          log, acc, "cl::sycl::buffer::get_access<read_write>(handler&)");
      cgh.single_task(empty_kernel());
    });

    /* check the buffer returns the correct type of accessor */
    q.submit([&](cl::sycl::handler& cgh) {
      auto acc =
          buf.template get_access<cl::sycl::access::mode::read,
                                  cl::sycl::access::target::constant_buffer>(
              cgh);
      check_return_type<
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                             cl::sycl::access::target::constant_buffer>>(
          log, acc,
          "cl::sycl::buffer::get_access<read, constant_buffer>(handler&)");
      cgh.single_task(empty_kernel());
    });

    /* check the buffer returns the correct type of accessor */
    {
      auto acc = buf.template get_access<cl::sycl::access::mode::read_write>();
      check_return_type<
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::read_write,
                             cl::sycl::access::target::host_buffer>>(
          log, acc, "cl::sycl::buffer::get_access<read_write, host_buffer>()");
    }

    /* check the buffer returns the correct type of accessor */
    q.submit([&](cl::sycl::handler& cgh) {
      auto acc = buf.template get_access<cl::sycl::access::mode::read_write>(
          cgh, r, offset);
      check_return_type<
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::read_write,
                             cl::sycl::access::target::global_buffer>>(
          log, acc,
          "cl::sycl::buffer::get_access<read_write, global_buffer>(handler&, "
          "range<>, id<>)");
      cgh.single_task(empty_kernel());
    });

    /* check the buffer returns the correct type of accessor */
    {
      auto acc = buf.template get_access<cl::sycl::access::mode::read_write>(
          r, offset);
      check_return_type<
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::read_write,
                             cl::sycl::access::target::host_buffer>>(
          log, acc,
          "cl::sycl::buffer::get_access<read_write, host_buffer>(range<>, "
          "id<>)");
    }

    /* check get_allocator() */
    {
      using AllocatorT = std::allocator<T>;

      /* create another buffer with a custom allocator */
      cl::sycl::buffer<T, dims, AllocatorT> bufAlloc(data.get(), r);

      auto allocator = bufAlloc.get_allocator();

      check_return_type<AllocatorT>(log, allocator, "get_allocator()");

      auto ptr = allocator.allocate(1);
      if (ptr == nullptr) {
        FAIL(log, "get_allocator() returned an invalid allocator ");
      }
      allocator.deallocate(ptr, 1);
    }

    /* check is_sub_buffer() */
    {
      cl::sycl::buffer<T, dims> buf(r);
      cl::sycl::range<dims> sub_r = r;
      sub_r[0] = r[0] - i[0];
      cl::sycl::buffer<T, dims> buf_sub(buf, i, sub_r);
      auto isSubBuffer = buf_sub.is_sub_buffer();
      check_return_type<bool>(log, isSubBuffer, "is_sub_buffer()");
    }

    /* check buffer properties */
    {
      cl::sycl::mutex_class mutex;
      auto context = util::get_cts_object::context();
      const cl::sycl::property_list pl{
          cl::sycl::property::buffer::use_mutex(mutex),
          cl::sycl::property::buffer::context_bound(context)};

      cl::sycl::buffer<T, dims> buf(r, pl);

      /* check has_property() */

      auto hasUseMutexProperty =
          buf.template has_property<cl::sycl::property::buffer::use_mutex>();
      check_return_type<bool>(log, hasUseMutexProperty,
                              "has_property<use_mutex>()");

      auto hasContentBoundProperty = buf.template has_property<
          cl::sycl::property::buffer::context_bound>();
      check_return_type<bool>(log, hasContentBoundProperty,
                              "has_property<context_bound>()");

      /* check get_property() */

      auto useMutexProperty =
          buf.template get_property<cl::sycl::property::buffer::use_mutex>();
      check_return_type<cl::sycl::property::buffer::use_mutex>(
          log, useMutexProperty, "get_property<use_mutex>()");
      check_return_type<cl::sycl::mutex_class*>(
          log, useMutexProperty.get_mutex_ptr(),
          "cl::sycl::property::buffer::use_mutex::get_mutex_ptr()");

      auto contentBoundProperty = buf.template get_property<
          cl::sycl::property::buffer::context_bound>();
      check_return_type<cl::sycl::property::buffer::context_bound>(
          log, contentBoundProperty, "get_property<context_bound>()");
      check_return_type<cl::sycl::context>(
          log, contentBoundProperty.get_context(),
          "cl::sycl::property::buffer::context_bound::get_context()");
    }

    q.wait_and_throw();
  } catch (const cl::sycl::exception& e) {
    log_exception(log, e);
    cl::sycl::string_class errorMsg =
        "a SYCL exception was caught: " + cl::sycl::string_class(e.what());
    FAIL(log, errorMsg.c_str());
  }
}

/** test cl::sycl::buffer API
 */
class TEST_NAME : public util::test_base {
 public:
  /** return information about this test
   */
  void get_info(test_base::info& out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  template <typename T>
  void test_type(util::logger& log) {
    const int size = 8;
    cl::sycl::range<1> range1d(size);
    cl::sycl::range<2> range2d(size, size);
    cl::sycl::range<3> range3d(size, size, size);

    cl::sycl::id<1> id1d(2);
    cl::sycl::id<2> id2d(2, 0);
    cl::sycl::id<3> id3d(2, 0, 0);

    test_buffer<T, size, 1>(log, range1d, id1d);
    test_buffer<T, size * size, 2>(log, range2d, id2d);
    test_buffer<T, size * size * size, 3>(log, range3d, id3d);

    /* check reinterpret() */

    {
      cl::sycl::vector_class<uint8_t> data(sizeof(T));
      test_buffer_reinterpret<uint8_t, T>(sizeof(T), 1)
          .template check<1>(data.data(), log);
    }
    {
      cl::sycl::vector_class<uint8_t> data(sizeof(T)*sizeof(T));
      test_buffer_reinterpret<uint8_t, T>(sizeof(T), 1)
          .template check<2>(data.data(), log);
    }
    {
      cl::sycl::vector_class<uint8_t> data(sizeof(T)*sizeof(T)*sizeof(T));
      test_buffer_reinterpret<uint8_t, T>(sizeof(T), 1)
          .template check<3>(data.data(), log);
    }
  }

  /** execute the test
   */
  void run(util::logger& log) override {
    /* test signed types */
    log.note("testing: int8_t");
    test_type<int8_t>(log);
    log.note("testing: int16_t");
    test_type<int16_t>(log);
    log.note("testing: int32_t");
    test_type<int32_t>(log);
    log.note("testing: int64_t");
    test_type<int64_t>(log);

    /* test unsigned types */
    log.note("testing: uint8_t");
    test_type<uint8_t>(log);
    log.note("testing: uint16_t");
    test_type<uint16_t>(log);
    log.note("testing: uint32_t");
    test_type<uint32_t>(log);
    log.note("testing: uint64_t");
    test_type<uint64_t>(log);

    /* test float types */
    log.note("testing: float");
    test_type<float>(log);
    log.note("testing: double");
    test_type<double>(log);

    /* test vector types */
    log.note("testing: float3");
    test_type<cl::sycl::float3>(log);
    log.note("testing: float2");
    test_type<cl::sycl::float2>(log);
    log.note("testing: float4");
    test_type<cl::sycl::float4>(log);
    log.note("testing: float8");
    test_type<cl::sycl::float8>(log);
    log.note("testing: float16");
    test_type<cl::sycl::float16>(log);

    log.note("testing: double2");
    test_type<cl::sycl::double2>(log);
    log.note("testing: double3");
    test_type<cl::sycl::double3>(log);
    log.note("testing: double4");
    test_type<cl::sycl::double4>(log);
    log.note("testing: double8");
    test_type<cl::sycl::double8>(log);
    log.note("testing: double16");
    test_type<cl::sycl::double16>(log);

    log.note("testing: char2");
    test_type<cl::sycl::char2>(log);
    log.note("testing: char3");
    test_type<cl::sycl::char3>(log);
    log.note("testing: char4");
    test_type<cl::sycl::char4>(log);
    log.note("testing: char8");
    test_type<cl::sycl::char8>(log);
    log.note("testing: char16");
    test_type<cl::sycl::char16>(log);

    log.note("testing: int2");
    test_type<cl::sycl::int2>(log);
    log.note("testing: int3");
    test_type<cl::sycl::int3>(log);
    log.note("testing: int4");
    test_type<cl::sycl::int4>(log);
    log.note("testing: int8");
    test_type<cl::sycl::int8>(log);
    log.note("testing: int16");
    test_type<cl::sycl::int16>(log);

    log.note("testing: short2");
    test_type<cl::sycl::short2>(log);
    log.note("testing: short3");
    test_type<cl::sycl::short3>(log);
    log.note("testing: short4");
    test_type<cl::sycl::short4>(log);
    log.note("testing: short8");
    test_type<cl::sycl::short8>(log);
    log.note("testing: short16");
    test_type<cl::sycl::short16>(log);

    log.note("testing: long2");
    test_type<cl::sycl::long2>(log);
    log.note("testing: long3");
    test_type<cl::sycl::long3>(log);
    log.note("testing: long4");
    test_type<cl::sycl::long4>(log);
    log.note("testing: long8");
    test_type<cl::sycl::long8>(log);
    log.note("testing: long16");
    test_type<cl::sycl::long16>(log);

    log.note("testing: uchar2");
    test_type<cl::sycl::uchar2>(log);
    log.note("testing: uchar3");
    test_type<cl::sycl::uchar3>(log);
    log.note("testing: uchar4");
    test_type<cl::sycl::uchar4>(log);
    log.note("testing: uchar8");
    test_type<cl::sycl::uchar8>(log);
    log.note("testing: uchar16");
    test_type<cl::sycl::uchar16>(log);

    log.note("testing: uint2");
    test_type<cl::sycl::uint2>(log);
    log.note("testing: uint3");
    test_type<cl::sycl::uint3>(log);
    log.note("testing: uint4");
    test_type<cl::sycl::uint4>(log);
    log.note("testing: uint8");
    test_type<cl::sycl::uint8>(log);
    log.note("testing: uint16");
    test_type<cl::sycl::uint16>(log);

    log.note("testing: ushort2");
    test_type<cl::sycl::ushort2>(log);
    log.note("testing: ushort3");
    test_type<cl::sycl::ushort3>(log);
    log.note("testing: ushort4");
    test_type<cl::sycl::ushort4>(log);
    log.note("testing: ushort8");
    test_type<cl::sycl::ushort8>(log);
    log.note("testing: ushort16");
    test_type<cl::sycl::ushort16>(log);

    log.note("testing: ulong2");
    test_type<cl::sycl::ulong2>(log);
    log.note("testing: ulong3");
    test_type<cl::sycl::ulong3>(log);
    log.note("testing: ulong4");
    test_type<cl::sycl::ulong4>(log);
    log.note("testing: ulong8");
    test_type<cl::sycl::ulong8>(log);
    log.note("testing: ulong16");
    test_type<cl::sycl::ulong16>(log);
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

}  // namespace TEST_NAMESPACE
