/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME buffer_api

namespace TEST_NAMESPACE {
using namespace cl::sycl;
using namespace cl::sycl::access;
using namespace sycl_cts;

/** empty_kernel.
 * Empty kernel, required since command grups
 * are required to have a kernel.
 */
class empty_kernel {
 public:
  void operator()() const {}
};

/**
 * Generic buffer API test function
 */
template <typename T, int size, int dims>
void test_buffer(util::logger &log, cl::sycl::range<dims> &r) {
  try {
    unique_ptr_class<T[]> data(new T[size]);
    std::fill(data.get(), (data.get() + size), 0);
    const id<dims> offset;

    /* create a sycl buffer from the host buffer */
    cl::sycl::buffer<T, dims> buf(data.get(), r);

    /* check the buffer returns a range */
    auto ret_range = buf.get_range();
    check_return_type<cl::sycl::range<dims>>(log, ret_range,
                                             "cl::sycl::buffer::get_range()");

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
    q.submit([&](handler &cgh) {
      auto acc =
          buf.template get_access<cl::sycl::access::mode::read_write>(cgh);
      check_return_type<
          accessor<T, dims, mode::read_write, target::global_buffer>>(
          log, acc, "cl::sycl::buffer::get_access()");
      cgh.single_task(empty_kernel());
    });

    /* check the buffer returns the correct type of accessor */
    q.submit([&](handler &cgh) {
      auto acc =
          buf.template get_access<mode::read, target::constant_buffer>(cgh);
      check_return_type<accessor<T, dims, mode::read, target::constant_buffer>>(
          log, acc, "cl::sycl::buffer::get_access()");
      cgh.single_task(empty_kernel());
    });

    /* check the buffer returns the correct type of accessor */
    {
      auto acc =
          buf.template get_access<mode::read_write, target::host_buffer>();
      check_return_type<
          accessor<T, dims, mode::read_write, target::host_buffer>>(
          log, acc, "cl::sycl::buffer::get_access()");
    }

    /* check the buffer returns the correct type of accessor */
    q.submit([&](handler &cgh) {
      auto acc = buf.template get_access<mode::read_write>(cgh, 0, r);
      check_return_type<
          accessor<T, dims, mode::read_write, target::global_buffer>>(
          log, acc, "cl::sycl::buffer::get_access()");
      cgh.single_task(empty_kernel());
    });

    /* check the buffer returns the correct type of accessor */
    {
      auto acc = buf.template get_access<mode::read_write>(0, r);
      check_return_type<
          accessor<T, dims, mode::read_write, target::host_buffer>>(
          log, acc, "cl::sycl::buffer::get_access()");
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

    q.wait_and_throw();
  } catch (cl::sycl::exception e) {
    log_exception(log, e);
    cl::sycl::string_class errorMsg =
        "a SYCL exception was caught: " + cl::sycl::string_class(e.what());
    FAIL(log, errorMsg.c_str());
  }
}

/** test cl::sycl::buffer api
 */
class TEST_NAME : public util::test_base {
 public:
  /** return information about this test
   */
  virtual void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  template <typename T>
  void test_type(util::logger &log) {
    const int size = 8;
    cl::sycl::range<1> range1d(size);
    cl::sycl::range<2> range2d(size, size);
    cl::sycl::range<3> range3d(size, size, size);

    test_buffer<T, size, 1>(log, range1d);
    test_buffer<T, size * size, 2>(log, range2d);
    test_buffer<T, size * size * size, 3>(log, range3d);
  }

  /** execute the test
   */
  virtual void run(util::logger &log) override {
    /* test signed types */
    test_type<int8_t>(log);
    test_type<int16_t>(log);
    test_type<int32_t>(log);
    test_type<int64_t>(log);

    /* test unsigned types */
    test_type<uint8_t>(log);
    test_type<uint16_t>(log);
    test_type<uint32_t>(log);
    test_type<uint64_t>(log);

    /* test float types */
    test_type<float>(log);
    test_type<double>(log);

    /* test vector types */
    test_type<cl::sycl::float3>(log);
    test_type<cl::sycl::float2>(log);
    test_type<cl::sycl::float4>(log);
    test_type<cl::sycl::float8>(log);
    test_type<cl::sycl::float16>(log);

    test_type<cl::sycl::double2>(log);
    test_type<cl::sycl::double3>(log);
    test_type<cl::sycl::double4>(log);
    test_type<cl::sycl::double8>(log);
    test_type<cl::sycl::double16>(log);

    test_type<cl::sycl::char2>(log);
    test_type<cl::sycl::char3>(log);
    test_type<cl::sycl::char4>(log);
    test_type<cl::sycl::char8>(log);
    test_type<cl::sycl::char16>(log);

    test_type<cl::sycl::int2>(log);
    test_type<cl::sycl::int3>(log);
    test_type<cl::sycl::int4>(log);
    test_type<cl::sycl::int8>(log);
    test_type<cl::sycl::int16>(log);

    test_type<cl::sycl::short2>(log);
    test_type<cl::sycl::short3>(log);
    test_type<cl::sycl::short4>(log);
    test_type<cl::sycl::short8>(log);
    test_type<cl::sycl::short16>(log);

    test_type<cl::sycl::long2>(log);
    test_type<cl::sycl::long3>(log);
    test_type<cl::sycl::long4>(log);
    test_type<cl::sycl::long8>(log);
    test_type<cl::sycl::long16>(log);

    test_type<cl::sycl::uchar2>(log);
    test_type<cl::sycl::uchar3>(log);
    test_type<cl::sycl::uchar4>(log);
    test_type<cl::sycl::uchar8>(log);
    test_type<cl::sycl::uchar16>(log);

    test_type<cl::sycl::uint2>(log);
    test_type<cl::sycl::uint3>(log);
    test_type<cl::sycl::uint4>(log);
    test_type<cl::sycl::uint8>(log);
    test_type<cl::sycl::uint16>(log);

    test_type<cl::sycl::ushort2>(log);
    test_type<cl::sycl::ushort3>(log);
    test_type<cl::sycl::ushort4>(log);
    test_type<cl::sycl::ushort8>(log);
    test_type<cl::sycl::ushort16>(log);

    test_type<cl::sycl::ulong2>(log);
    test_type<cl::sycl::ulong3>(log);
    test_type<cl::sycl::ulong4>(log);
    test_type<cl::sycl::ulong8>(log);
    test_type<cl::sycl::ulong16>(log);
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

} /* namespace buffer_api__ */
