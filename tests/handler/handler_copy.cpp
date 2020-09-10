/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME handler_copy

namespace TEST_NAMESPACE {
using namespace sycl_cts;

using mode_t = cl::sycl::access::mode;
using target_t = cl::sycl::access::target;

/**
 * @brief Helps with getting the buffer range and filling the buffer with data
 */
template <typename dataT, int dims>
struct buffer_helper;

template <typename dataT>
struct buffer_helper<dataT, 1> {
  static cl::sycl::range<1> construct_range(size_t elemsPerDim) {
    return {elemsPerDim};
  }
  static void fill(cl::sycl::buffer<dataT, 1>& buf, const dataT& value) {
    auto r = buf.get_range();
    auto acc = buf.template get_access<mode_t::discard_write>();
    for (size_t i = 0; i < r[0]; ++i) {
      acc[i] = value;
    }
  }
};

template <typename dataT>
struct buffer_helper<dataT, 2> {
  static cl::sycl::range<2> construct_range(size_t elemsPerDim) {
    return {elemsPerDim, elemsPerDim};
  }
  static void fill(cl::sycl::buffer<dataT, 2>& buf, const dataT& value) {
    auto r = buf.get_range();
    auto acc = buf.template get_access<mode_t::discard_write>();
    for (size_t r0 = 0; r0 < r[0]; ++r0) {
      for (size_t r1 = 0; r1 < r[1]; ++r1) {
        acc[r0][r1] = value;
      }
    }
  }
};

template <typename dataT>
struct buffer_helper<dataT, 3> {
  static cl::sycl::range<3> construct_range(size_t elemsPerDim) {
    return {elemsPerDim, elemsPerDim, elemsPerDim};
  }
  static void fill(cl::sycl::buffer<dataT, 3>& buf, const dataT& value) {
    auto r = buf.get_range();
    auto acc = buf.template get_access<mode_t::discard_write>();
    for (size_t r0 = 0; r0 < r[0]; ++r0) {
      for (size_t r1 = 0; r1 < r[1]; ++r1) {
        for (size_t r2 = 0; r2 < r[2]; ++r2) {
          acc[r0][r1][r2] = value;
        }
      }
    }
  }
};

/**
 * @brief Test context that stores some data that helps with checks
 */
template <typename dataT, int dim_src, int dim_dst>
struct copy_test_context {
  using host_shared_ptr = cl::sycl::shared_ptr_class<dataT>;
  using buffer_src_t = cl::sycl::buffer<dataT, dim_src>;
  using buffer_dst_t = cl::sycl::buffer<dataT, dim_dst>;

  static constexpr int get_elems(int dim) {
    return (dim == 1) ? 64 : (dim == 2) ? 8 : 4;
  }

  static constexpr size_t numElems = 64;
  static constexpr size_t elemsSrcPerDim = get_elems(dim_src);
  static constexpr size_t elemsDstPerDim = get_elems(dim_dst);
  static constexpr size_t bufferInitValue = 17;

  buffer_src_t srcBuf;
  buffer_dst_t dstBuf;

  host_shared_ptr srcHostPtr;
  host_shared_ptr dstHostPtr;

  copy_test_context()
      : srcBuf(buffer_helper<dataT, dim_src>::construct_range(elemsSrcPerDim)),
        dstBuf(buffer_helper<dataT, dim_dst>::construct_range(elemsDstPerDim)),
        srcHostPtr(new dataT[numElems], std::default_delete<dataT[]>()),
        dstHostPtr(new dataT[numElems], std::default_delete<dataT[]>()) {
    buffer_helper<dataT, dim_src>::fill(srcBuf,
                                        static_cast<dataT>(bufferInitValue));
    buffer_helper<dataT, dim_dst>::fill(dstBuf, static_cast<dataT>(0));
    for (size_t i = 0; i < numElems; ++i) {
      srcHostPtr.get()[i] = static_cast<dataT>(i);
      dstHostPtr.get()[i] = static_cast<dataT>(0);
    }
  }
};

template <typename dataT, int dimSrc, int dimDst, typename testFunction>
copy_test_context<dataT, dimSrc, dimDst> submit_test_function(
    cl::sycl::queue& queue, testFunction fn) {
  copy_test_context<dataT, dimSrc, dimDst> ctx;
  queue.submit([&](cl::sycl::handler& cgh) { fn(ctx, cgh); });
  queue.wait_and_throw();
  return ctx;
}

/**
 * @brief Creates lambdas with the actual tested functionality and passes them
 *        on to be tested. This doesn't include functions that expect a write
 *        accessor.
 */
template <typename dataT, int dim, mode_t mode_src, target_t target>
void test_read_acc_copy_functions(cl::sycl::queue& queue) {
  {
    // Check copy(accessor, shared_ptr_class)
    submit_test_function<dataT, dim, dim>(
        queue,
        [](copy_test_context<dataT, dim, dim>& ctx, cl::sycl::handler& cgh) {
          auto r = ctx.srcBuf.template get_access<mode_src, target>(cgh);
          cgh.copy(r, ctx.dstHostPtr);
        });
  }
  {
    // Check copy(accessor, dataT*)
    submit_test_function<dataT, dim, dim>(
        queue,
        [](copy_test_context<dataT, dim, dim>& ctx, cl::sycl::handler& cgh) {
          auto r = ctx.srcBuf.template get_access<mode_src, target>(cgh);
          cgh.copy(r, ctx.dstHostPtr.get());
        });
  }
  {
    // Check update_host(accessor)
    submit_test_function<dataT, dim, dim>(
        queue,
        [](copy_test_context<dataT, dim, dim>& ctx, cl::sycl::handler& cgh) {
          auto r = ctx.srcBuf.template get_access<mode_src, target>(cgh);
          cgh.update_host(r);
        });
  }
}

/**
 * @brief Creates lambdas with the actual tested functionality and passes them
 *        on to be tested. This includes functions that expect a write accessor.
 */
template <typename dataT, int dim_src, int dim_dst, mode_t mode_src,
          mode_t mode_dst, target_t target>
void test_write_acc_copy_functions(cl::sycl::queue& queue) {
  {
    // Check copy(shared_ptr_class, accessor)
    submit_test_function<dataT, dim_src, dim_dst>(
        queue, [](copy_test_context<dataT, dim_src, dim_dst>& ctx,
                  cl::sycl::handler& cgh) {
          auto w = ctx.dstBuf.template get_access<mode_dst, target>(cgh);
          cgh.copy(ctx.srcHostPtr, w);
        });
  }
  {
    // Check copy(dataT*, accessor)
    submit_test_function<dataT, dim_src, dim_dst>(
        queue, [](copy_test_context<dataT, dim_src, dim_dst>& ctx,
                  cl::sycl::handler& cgh) {
          auto w = ctx.dstBuf.template get_access<mode_dst, target>(cgh);
          cgh.copy(ctx.srcHostPtr.get(), w);
        });
  }
  {
    // Check copy(accessor, accessor)
    submit_test_function<dataT, dim_src, dim_dst>(
        queue, [](copy_test_context<dataT, dim_src, dim_dst>& ctx,
                  cl::sycl::handler& cgh) {
          auto r = ctx.srcBuf.template get_access<mode_src, target>(cgh);
          auto w = ctx.dstBuf.template get_access<mode_dst, target>(cgh);
          cgh.copy(r, w);
        });
  }
  {
    // Check fill(accessor, dataT)
    submit_test_function<dataT, dim_src, dim_dst>(
        queue, [](copy_test_context<dataT, dim_src, dim_dst>& ctx,
                  cl::sycl::handler& cgh) {
          auto w = ctx.dstBuf.template get_access<mode_dst, target>(cgh);
          const auto pattern = dataT(117);
          cgh.fill(w, pattern);
        });
  }
}

/**
 * @brief Tests all valid combinations of access modes and buffer targets.
 */
template <typename dataT, int dim_src>
void test_all_read_acc_copy_functions(cl::sycl::queue& queue) {
  {
    constexpr auto target = target_t::global_buffer;
    test_read_acc_copy_functions<dataT, dim_src, mode_t::read, target>(queue);
    test_read_acc_copy_functions<dataT, dim_src, mode_t::read_write, target>(
        queue);
  }
  {
    constexpr auto target = target_t::constant_buffer;
    test_read_acc_copy_functions<dataT, dim_src, mode_t::read, target>(queue);
  }
}

/**
 * @brief Tests all valid combinations of source and destination access modes.
 */
template <typename dataT, int dim_src, int dim_dst>
void test_all_write_acc_copy_functions(cl::sycl::queue& queue) {
  constexpr auto target = target_t::global_buffer;

  test_write_acc_copy_functions<dataT, dim_src, dim_dst, mode_t::read,
                                mode_t::write, target>(queue);
  test_write_acc_copy_functions<dataT, dim_src, dim_dst, mode_t::read,
                                mode_t::read_write, target>(queue);
  test_write_acc_copy_functions<dataT, dim_src, dim_dst, mode_t::read,
                                mode_t::discard_write, target>(queue);
  test_write_acc_copy_functions<dataT, dim_src, dim_dst, mode_t::read,
                                mode_t::discard_read_write, target>(queue);

  test_write_acc_copy_functions<dataT, dim_src, dim_dst, mode_t::read_write,
                                mode_t::write, target>(queue);
  test_write_acc_copy_functions<dataT, dim_src, dim_dst, mode_t::read_write,
                                mode_t::read_write, target>(queue);
  test_write_acc_copy_functions<dataT, dim_src, dim_dst, mode_t::read_write,
                                mode_t::discard_write, target>(queue);
  test_write_acc_copy_functions<dataT, dim_src, dim_dst, mode_t::read_write,
                                mode_t::discard_read_write, target>(queue);
}

template <typename dataT>
void test_all_dimensions(cl::sycl::queue& queue) {
  test_all_read_acc_copy_functions<dataT, 1>(queue);
  test_all_write_acc_copy_functions<dataT, 1, 1>(queue);
  test_all_write_acc_copy_functions<dataT, 1, 2>(queue);
  test_all_write_acc_copy_functions<dataT, 1, 3>(queue);

  test_all_read_acc_copy_functions<dataT, 2>(queue);
  test_all_write_acc_copy_functions<dataT, 2, 1>(queue);
  test_all_write_acc_copy_functions<dataT, 2, 2>(queue);
  test_all_write_acc_copy_functions<dataT, 2, 3>(queue);

  test_all_read_acc_copy_functions<dataT, 3>(queue);
  test_all_write_acc_copy_functions<dataT, 3, 1>(queue);
  test_all_write_acc_copy_functions<dataT, 3, 2>(queue);
  test_all_write_acc_copy_functions<dataT, 3, 3>(queue);
}

/** tests the API for cl::sycl::handler
 */
class TEST_NAME : public util::test_base {
 public:
  /** return information about this test
   */
  void get_info(test_base::info& out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute the test
   */
  void run(util::logger& log) override {
    try {
      auto queue = util::get_cts_object::queue();

      test_all_dimensions<char>(queue);
      test_all_dimensions<short>(queue);
      test_all_dimensions<int>(queue);
      test_all_dimensions<long>(queue);
      test_all_dimensions<float>(queue);
      test_all_dimensions<double>(queue);

      test_all_dimensions<cl::sycl::char2>(queue);
      test_all_dimensions<cl::sycl::short3>(queue);
      test_all_dimensions<cl::sycl::int4>(queue);
      test_all_dimensions<cl::sycl::long8>(queue);
      test_all_dimensions<cl::sycl::float8>(queue);
      test_all_dimensions<cl::sycl::double16>(queue);

    } catch (const cl::sycl::exception& e) {
      log_exception(log, e);
      cl::sycl::string_class errorMsg =
          "a SYCL exception was caught: " + cl::sycl::string_class(e.what());
      FAIL(log, errorMsg.c_str());
    }
  }
};

// register this test with the test_collection
util::test_proxy<TEST_NAME> proxy;

} /* namespace TEST_NAMESPACE */
