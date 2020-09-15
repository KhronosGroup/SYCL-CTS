/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include <regex>
#include <mutex>
#include <sstream>
#include <memory>

#include "../common/common.h"

#define TEST_NAME handler_copy

namespace TEST_NAMESPACE {
using namespace sycl_cts;

using mode_t = cl::sycl::access::mode;
using target_t = cl::sycl::access::target;

/**
 * @brief Helper class designed to construct useful failure messages for all
 * test case permutations.
 */
class log_helper {
 public:
  log_helper(util::logger* logger) : logger(logger) {}

  template <typename dataT>
  log_helper set_data_type() const {
    auto result = *this;
    if (std::is_same<dataT, char>::value) result.dataType = "char";
    if (std::is_same<dataT, short>::value) result.dataType = "short";
    if (std::is_same<dataT, int>::value) result.dataType = "int";
    if (std::is_same<dataT, long>::value) result.dataType = "long";
    if (std::is_same<dataT, float>::value) result.dataType = "float";
    if (std::is_same<dataT, double>::value) result.dataType = "double";
    if (std::is_same<dataT, cl::sycl::char2>::value)
      result.dataType = "cl::sycl::char2";
    if (std::is_same<dataT, cl::sycl::short3>::value)
      result.dataType = "cl::sycl::short3";
    if (std::is_same<dataT, cl::sycl::int4>::value)
      result.dataType = "cl::sycl::int4";
    if (std::is_same<dataT, cl::sycl::long8>::value)
      result.dataType = "cl::sycl::long8";
    if (std::is_same<dataT, cl::sycl::float8>::value)
      result.dataType = "cl::sycl::float8";
    if (std::is_same<dataT, cl::sycl::double16>::value)
      result.dataType = "cl::sycl::double16";
    return result;
  }

  log_helper set_dim_src(int dim) const {
    auto result = *this;
    result.dimSrc = dim;
    return result;
  }

  log_helper set_dim_dst(int dim) const {
    auto result = *this;
    result.dimDst = dim;
    return result;
  }

  log_helper set_mode_src(mode_t mode) const {
    auto result = *this;
    result.modeSrc = get_mode_string(mode);
    return result;
  }

  log_helper set_mode_dst(mode_t mode) const {
    auto result = *this;
    result.modeDst = get_mode_string(mode);
    return result;
  }

  log_helper set_target(target_t target) const {
    auto result = *this;
    result.target = get_target_string(target);
    return result;
  }

  log_helper set_line(int line) const {
    auto result = *this;
    result.line = line;
    return result;
  }

  log_helper set_op(const std::string& pattern) const {
    auto result = *this;
    result.pattern = pattern;
    return result;
  }

  void fail(const std::string& reason) const {
    logger->fail(make_description() + " failed: " + reason, line);
  }

  void note(const std::string& message) const {
    logger->note(make_description() + " info: " + message);
  }

 private:
  util::logger* logger;
  std::string dataType = "(unknown data type)";
  int dimSrc = -1;
  int dimDst = -1;
  std::string modeSrc = "(unknown mode)";
  std::string modeDst = "(unknown mode)";
  std::string target = "(unknown target)";
  int line = __LINE__;
  std::string pattern = "";

  static std::string get_mode_string(mode_t mode) {
    switch (mode) {
      case mode_t::read:
        return "read";
      case mode_t::write:
        return "write";
      case mode_t::read_write:
        return "read_write";
      case mode_t::discard_write:
        return "discard_write";
      case mode_t::discard_read_write:
        return "discard_read_write";
      case mode_t::atomic:
        return "atomic";
      default:
        return "(unknown mode)";
    }
  }

  static std::string get_target_string(target_t target) {
    switch (target) {
      case target_t::global_buffer:
        return "global_buffer";
      case target_t::constant_buffer:
        return "constant_buffer";
      default:
        return "(unknown target)";
    }
  }

  std::string make_description() const {
    auto desc = pattern;
    desc = std::regex_replace(desc, std::regex("\\$dataT"), dataType);
    desc = std::regex_replace(desc, std::regex("\\$dim_src"),
                              std::to_string(dimSrc));
    desc = std::regex_replace(desc, std::regex("\\$dim_dst"),
                              std::to_string(dimDst));
    desc = std::regex_replace(desc, std::regex("\\$mode_src"), modeSrc);
    desc = std::regex_replace(desc, std::regex("\\$mode_dst"), modeDst);
    desc = std::regex_replace(desc, std::regex("\\$target"), target);
    return desc;
  }
};

template <typename T>
struct type_helper {
  static T make(size_t v) { return static_cast<T>(v); }
  static size_t value(const T& v) { return static_cast<size_t>(v); }
  static bool equal(const T& lhs, const T& rhs) {
    return value(lhs) == value(rhs);
  }
};

template <typename dataT, int numElements>
struct type_helper<cl::sycl::vec<dataT, numElements>> {
  using T = cl::sycl::vec<dataT, numElements>;
  static T make(size_t v) {
    return T{static_cast<typename T::element_type>(v)};
  }
  static size_t value(const T& v) { return static_cast<size_t>(v.s0()); }
  static bool equal(const T& lhs, const T& rhs) {
    // Ideally we'd check that all components are equal, however unfortunately
    // SYCL 1.2.1 doesn't specify a generic subscript operator for vector types.
    return value(lhs) == value(rhs);
  }
};

template <typename dataT, int dims>
class fill_kernel;

template <typename dataT, int dims>
void fill_buffer(cl::sycl::queue& queue, cl::sycl::buffer<dataT, dims>& buf,
                 dataT value) {
  queue.submit([&](cl::sycl::handler& cgh) {
    auto acc = buf.template get_access<mode_t::discard_write>(cgh);
    cgh.parallel_for<fill_kernel<dataT, dims>>(
        buf.get_range(), [=](cl::sycl::id<dims> id) { acc[id] = value; });
  });
}

template <template <int> class T, int dims, size_t default_value>
struct range_id_helper {};

template <template <int> class T, size_t default_value>
struct range_id_helper<T, 1, default_value> {
  static T<1> make(size_t d0, size_t, size_t) { return T<1>{d0}; }

  template <template <int> class other, int other_dims,
            size_t dv = default_value>
  static T<1> cast(const other<other_dims>& o) {
    return T<1>{o[0]};
  }
};

template <template <int dims> class T, size_t default_value>
struct range_id_helper<T, 2, default_value> {
  static T<2> make(size_t d0, size_t d1, size_t) { return T<2>{d0, d1}; }

  template <template <int> class other, int other_dims,
            size_t dv = default_value>
  static T<2> cast(const other<other_dims>& o) {
    return T<2>{o[0], other_dims >= 2 ? o[1] : dv};
  }
};

template <template <int dims> class T, size_t default_value>
struct range_id_helper<T, 3, default_value> {
  static T<3> make(size_t d0, size_t d1, size_t d2) { return T<3>{d0, d1, d2}; }

  template <template <int> class other, int other_dims,
            size_t dv = default_value>
  static T<3> cast(const other<other_dims>& o) {
    return T<3>{o[0], other_dims >= 2 ? o[1] : dv, other_dims == 3 ? o[2] : dv};
  }
};

template <int dims>
using range_helper = range_id_helper<cl::sycl::range, dims, 1>;

template <int dims>
using id_helper = range_id_helper<cl::sycl::id, dims, 0>;

/**
 * @brief The copy_test_context encapsulates all host and device data required
 * for testing, and provides utility functions for verifying the result
 * of the various explicit memory operations.
 */
template <typename dataT, int dim_src, int dim_dst>
class copy_test_context {
  using host_shared_ptr = cl::sycl::shared_ptr_class<dataT>;
  using buffer_src_t = cl::sycl::buffer<dataT, dim_src>;
  using buffer_dst_t = cl::sycl::buffer<dataT, dim_dst>;
  using th = type_helper<dataT>;

 public:
  explicit copy_test_context(cl::sycl::queue& queue) : queue(queue) {
    setup_ranges();

    srcBufHostMemory =
        host_shared_ptr(new dataT[numElems], std::default_delete<dataT[]>());
    srcHostPtr =
        host_shared_ptr(new dataT[numElems], std::default_delete<dataT[]>());
    dstHostPtr =
        host_shared_ptr(new dataT[numElems], std::default_delete<dataT[]>());

    for (size_t i = 0; i < numElems; ++i) {
      srcBufHostMemory.get()[i] = hostCanary;
      srcHostPtr.get()[i] = bufferInitValue;
      dstHostPtr.get()[i] = hostCanary;
    }

    srcBuf = std::unique_ptr<buffer_src_t>(new buffer_src_t(
        srcBufHostMemory, srcBufRange,
        cl::sycl::property_list{
            cl::sycl::property::buffer::use_mutex{srcBufHostMemoryMutex}}));
    dstBuf = std::unique_ptr<buffer_dst_t>(new buffer_dst_t(dstBufRange));

    fill_buffer(queue, *srcBuf, bufferInitValue);
    fill_buffer(queue, *dstBuf, deviceCanary);

    queue.wait_and_throw();
  }

  /**
   * @brief Verifies a device to host copy.
   */
  template <typename test_fn>
  void verify_d2h_copy(test_fn fn, const log_helper& lh) const {
    run_test_function(fn, lh);

    for (size_t i = 0; i < numElems; ++i) {
      const auto expected = bufferInitValue;
      const auto received = dstHostPtr.get()[i];
      if (!th::equal(received, expected)) {
        log_error(lh, cl::sycl::id<3>(i, 0, 0), received, expected);
        return;
      }
    }
  }

  /**
   * @brief Verifies that the host memory backing the source buffer has been
   * updated correctly.
   */
  template <typename test_fn>
  void verify_update_host(test_fn fn, const log_helper& lh) const {
    run_test_function(fn, lh);

    std::lock_guard<cl::sycl::mutex_class> lock(srcBufHostMemoryMutex);
    for (size_t i = 0; i < numElems; ++i) {
      const auto expected = bufferInitValue;
      const auto received = srcBufHostMemory.get()[i];
      if (!th::equal(received, expected)) {
        log_error(lh, cl::sycl::id<3>(i, 0, 0), received, expected);
        return;
      }
    }
  }

  /**
   * @brief Verifies a host to device or device to device copy.
   */
  template <typename test_fn>
  void verify_device_copy(test_fn fn, const log_helper& lh) {
    run_test_function(fn, lh);

    // TODO: Consider verifying directly on device.
    auto acc = dstBuf->template get_access<cl::sycl::access::mode::read>();
    for (size_t i = 0; i < numElems; ++i) {
      const auto expected = bufferInitValue;
      const auto dstIndex = reconstruct_index(dstBufRange, i);
      const auto received = acc[dstIndex];

      if (!th::equal(received, expected)) {
        log_error(lh, id_helper<3>::cast(dstIndex), received, expected);
        return;
      }
    }
  }

  /**
   * @brief Verifies that the device buffer has been filled correctly.
   *
   * @param fn
   * @param expected The value that was used to fill the region.
   * @param lh
   */
  template <typename test_fn>
  void verify_fill(test_fn fn, dataT expected, const log_helper& lh) {
    run_test_function(fn, lh);

    // TODO: Consider verifying directly on device.
    auto acc = dstBuf->template get_access<cl::sycl::access::mode::read>();
    for (size_t i = 0; i < numElems; ++i) {
      const auto idx = reconstruct_index(dstBufRange, i);
      const auto received = acc[idx];
      if (!th::equal(received, expected)) {
        log_error(lh, id_helper<3>::cast(idx), received, expected);
        return;
      }
    }
  }

  buffer_src_t getSrcBuf() const { return *srcBuf; }
  buffer_dst_t getDstBuf() const { return *dstBuf; }

  host_shared_ptr getSrcHostPtr() const { return srcHostPtr; }
  host_shared_ptr getDstHostPtr() const { return dstHostPtr; }

 private:
  cl::sycl::queue& queue;

  const dataT bufferInitValue = th::make(17);
  const dataT hostCanary = th::make(12345);
  const dataT deviceCanary = th::make(54321);

  cl::sycl::range<dim_src> srcBufRange = range_helper<dim_src>::make(0, 0, 0);
  cl::sycl::range<dim_dst> dstBufRange = range_helper<dim_dst>::make(0, 0, 0);

  std::unique_ptr<buffer_src_t> srcBuf;
  std::unique_ptr<buffer_dst_t> dstBuf;

  host_shared_ptr srcHostPtr;
  host_shared_ptr dstHostPtr;

  size_t numElems = 0;

  // Host memory region backing srcBuf,
  // used for testing handler::update_host().
  host_shared_ptr srcBufHostMemory = nullptr;
  mutable cl::sycl::mutex_class srcBufHostMemoryMutex;

  template <int dim>
  static cl::sycl::id<dim> reconstruct_index(cl::sycl::range<dim> range,
                                             size_t linearIndex) {
    assert(range.size() > 0);
    const auto r3 = cl::sycl::range<3>(range[0], dim > 1 ? range[1] : 1,
                                       dim > 2 ? range[2] : 1);
    const auto d0 = linearIndex / (r3[1] * r3[2]);
    const auto d1 = linearIndex % (r3[1] * r3[2]) / r3[2];
    const auto d2 = linearIndex % (r3[1] * r3[2]) % r3[2];
    return range_helper<dim>::make(d0, d1, d2);
  }

  static void log_error(const log_helper& lh, cl::sycl::id<3> index,
                        dataT received, dataT expected) {
    std::stringstream ss;
    ss << "Unexpected value at index ";
    ss << "[" << index[0] << "," << index[1] << "," << index[2] << "]: ";
    ss << th::value(received) << " (received) != " << th::value(expected)
       << " (expected)\n";
    lh.fail(ss.str());
  }

  void setup_ranges() {
    const auto elemsSrcPerDim = (dim_src == 1) ? 64 : (dim_src == 2) ? 8 : 4;
    const auto elemsDstPerDim = (dim_dst == 1) ? 64 : (dim_dst == 2) ? 8 : 4;

    srcBufRange = range_helper<dim_src>::make(elemsSrcPerDim, elemsSrcPerDim,
                                              elemsSrcPerDim);
    dstBufRange = range_helper<dim_dst>::make(elemsDstPerDim, elemsDstPerDim,
                                              elemsDstPerDim);

    assert(srcBufRange.size() == dstBufRange.size());
    numElems = srcBufRange.size();
  }

  template <typename test_fn>
  void run_test_function(test_fn fn, const log_helper& lh) const {
    // lh.note("Running...");  // Enable for verbose debugging output
    try {
      queue.submit([&](cl::sycl::handler& cgh) { fn(cgh); });
      queue.wait_and_throw();
    } catch (cl::sycl::exception&) {
      lh.fail("Exception thrown during call:");
      throw;
    }
  }
};

/**
 * @brief Creates lambdas with the actual tested functionality and passes them
 *        on to be tested. This doesn't include functions that expect a write
 *        accessor.
 */
template <typename dataT, int dim, mode_t mode_src, target_t target>
void test_read_acc_copy_functions(log_helper lh, cl::sycl::queue& queue) {
  lh = lh.set_mode_src(mode_src).set_target(target);
  {
    // Check copy(accessor, shared_ptr_class)
    copy_test_context<dataT, dim, dim> ctx(queue);
    ctx.verify_d2h_copy(
        [&](cl::sycl::handler& cgh) {
          auto r = ctx.getSrcBuf().template get_access<mode_src, target>(cgh);
          cgh.copy(r, ctx.getDstHostPtr());
        },
        lh.set_line(__LINE__).set_op(
            "copy(accessor<$dataT, $dim_src, $mode_src, $target>, "
            "shared_ptr_class<$dataT>)"));
  }
  {
    // Check copy(accessor, dataT*)
    copy_test_context<dataT, dim, dim> ctx(queue);
    ctx.verify_d2h_copy(
        [&](cl::sycl::handler& cgh) {
          auto r = ctx.getSrcBuf().template get_access<mode_src, target>(cgh);
          cgh.copy(r, ctx.getDstHostPtr().get());
        },
        lh.set_line(__LINE__).set_op(
            "copy(accessor<$dataT, $dim_src, $mode_src, $target>, "
            "$dataT*)"));
  }
  {
    // Check update_host(accessor)
    copy_test_context<dataT, dim, dim> ctx(queue);
    ctx.verify_update_host(
        [&](cl::sycl::handler& cgh) {
          auto r = ctx.getSrcBuf().template get_access<mode_src, target>(cgh);
          cgh.update_host(r);
        },
        lh.set_line(__LINE__).set_op(
            "update_host(accessor<$dataT, $dim_src, $mode_src, $target>)"));
  }
}

/**
 * @brief Creates lambdas with the actual tested functionality and passes them
 *        on to be tested. This includes functions that expect a write accessor.
 */
template <typename dataT, int dim_src, int dim_dst, mode_t mode_src,
          mode_t mode_dst, target_t target>
void test_write_acc_copy_functions(log_helper lh, cl::sycl::queue& queue) {
  lh = lh.set_mode_src(mode_src).set_mode_dst(mode_dst).set_target(target);
  {
    // Check copy(shared_ptr_class, accessor)
    copy_test_context<dataT, dim_src, dim_dst> ctx(queue);
    ctx.verify_device_copy(
        [&](cl::sycl::handler& cgh) {
          auto w = ctx.getDstBuf().template get_access<mode_dst, target>(cgh);
          cgh.copy(ctx.getSrcHostPtr(), w);
        },
        lh.set_line(__LINE__).set_op(
            "copy(shared_ptr_class<$dataT>, accessor<$dataT, $dim_dst, "
            "$mode_dst, $target>)"));
  }
  {
    // Check copy(dataT*, accessor)
    copy_test_context<dataT, dim_src, dim_dst> ctx(queue);
    ctx.verify_device_copy(
        [&](cl::sycl::handler& cgh) {
          auto w = ctx.getDstBuf().template get_access<mode_dst, target>(cgh);
          cgh.copy(ctx.getSrcHostPtr().get(), w);
        },
        lh.set_line(__LINE__).set_op(
            "copy($dataT*, accessor<$dataT, $dim_dst, $mode_dst, $target>)"));
  }
  {
    // Check copy(accessor, accessor)
    copy_test_context<dataT, dim_src, dim_dst> ctx(queue);
    ctx.verify_device_copy(
        [&](cl::sycl::handler& cgh) {
          auto r = ctx.getSrcBuf().template get_access<mode_src, target>(cgh);
          auto w = ctx.getDstBuf().template get_access<mode_dst, target>(cgh);
          cgh.copy(r, w);
        },
        lh.set_line(__LINE__).set_op(
            "copy(accessor<$dataT, $dim_src, $mode_src, $target>, "
            "accessor<$dataT, $dim_dst, $mode_dst, $target>)"));
  }
  {
    // Check fill(accessor, dataT)
    const auto pattern = type_helper<dataT>::make(117);
    copy_test_context<dataT, dim_src, dim_dst> ctx(queue);
    ctx.verify_fill(
        [&](cl::sycl::handler& cgh) {
          auto w = ctx.getDstBuf().template get_access<mode_dst, target>(cgh);
          cgh.fill(w, pattern);
        },
        pattern,
        lh.set_line(__LINE__).set_op(
            "fill(accessor<$dataT, $dim_dst, $mode_dst, $target>)"));
  }
}

/**
 * @brief Tests all valid combinations of access modes and buffer targets.
 */
template <typename dataT, int dim_src>
void test_all_read_acc_copy_functions(log_helper lh, cl::sycl::queue& queue) {
  lh = lh.set_dim_src(dim_src);
  {
    constexpr auto target = target_t::global_buffer;
    test_read_acc_copy_functions<dataT, dim_src, mode_t::read, target>(lh,
                                                                       queue);
    test_read_acc_copy_functions<dataT, dim_src, mode_t::read_write, target>(
        lh, queue);
  }
  {
    constexpr auto target = target_t::constant_buffer;
    test_read_acc_copy_functions<dataT, dim_src, mode_t::read, target>(lh,
                                                                       queue);
  }
}

/**
 * @brief Tests all valid combinations of source and destination access modes.
 */
template <typename dataT, int dim_src, int dim_dst>
void test_all_write_acc_copy_functions(log_helper lh, cl::sycl::queue& queue) {
  lh = lh.set_dim_src(dim_src).set_dim_dst(dim_dst);
  constexpr auto target = target_t::global_buffer;

  test_write_acc_copy_functions<dataT, dim_src, dim_dst, mode_t::read,
                                mode_t::write, target>(lh, queue);
  test_write_acc_copy_functions<dataT, dim_src, dim_dst, mode_t::read,
                                mode_t::read_write, target>(lh, queue);
  test_write_acc_copy_functions<dataT, dim_src, dim_dst, mode_t::read,
                                mode_t::discard_write, target>(lh, queue);
  test_write_acc_copy_functions<dataT, dim_src, dim_dst, mode_t::read,
                                mode_t::discard_read_write, target>(lh, queue);

  test_write_acc_copy_functions<dataT, dim_src, dim_dst, mode_t::read_write,
                                mode_t::write, target>(lh, queue);
  test_write_acc_copy_functions<dataT, dim_src, dim_dst, mode_t::read_write,
                                mode_t::read_write, target>(lh, queue);
  test_write_acc_copy_functions<dataT, dim_src, dim_dst, mode_t::read_write,
                                mode_t::discard_write, target>(lh, queue);
  test_write_acc_copy_functions<dataT, dim_src, dim_dst, mode_t::read_write,
                                mode_t::discard_read_write, target>(lh, queue);
}

/**
 * @brief Tests all valid combinations of source and destination dimensions.
 */
template <typename dataT>
void test_all_dimensions(log_helper lh, cl::sycl::queue& queue) {
  lh = lh.set_data_type<dataT>();
  test_all_read_acc_copy_functions<dataT, 1>(lh, queue);
  test_all_write_acc_copy_functions<dataT, 1, 1>(lh, queue);
  test_all_write_acc_copy_functions<dataT, 1, 2>(lh, queue);
  test_all_write_acc_copy_functions<dataT, 1, 3>(lh, queue);

  test_all_read_acc_copy_functions<dataT, 2>(lh, queue);
  test_all_write_acc_copy_functions<dataT, 2, 1>(lh, queue);
  test_all_write_acc_copy_functions<dataT, 2, 2>(lh, queue);
  test_all_write_acc_copy_functions<dataT, 2, 3>(lh, queue);

  test_all_read_acc_copy_functions<dataT, 3>(lh, queue);
  test_all_write_acc_copy_functions<dataT, 3, 1>(lh, queue);
  test_all_write_acc_copy_functions<dataT, 3, 2>(lh, queue);
  test_all_write_acc_copy_functions<dataT, 3, 3>(lh, queue);
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

      log_helper lh(&log);

      test_all_dimensions<char>(lh, queue);
      test_all_dimensions<short>(lh, queue);
      test_all_dimensions<int>(lh, queue);
      test_all_dimensions<long>(lh, queue);
      test_all_dimensions<float>(lh, queue);
      test_all_dimensions<double>(lh, queue);

      test_all_dimensions<cl::sycl::char2>(lh, queue);
      test_all_dimensions<cl::sycl::short3>(lh, queue);
      test_all_dimensions<cl::sycl::int4>(lh, queue);
      test_all_dimensions<cl::sycl::long8>(lh, queue);
      test_all_dimensions<cl::sycl::float8>(lh, queue);
      test_all_dimensions<cl::sycl::double16>(lh, queue);

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
