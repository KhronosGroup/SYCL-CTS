/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2015 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"
#include "../../oclmath/mt19937.h"
#include "./../../util/math_vector.h"
#include "./../../util/math_helper.h"

#define TEST_NAME math_geometric_fast_normalize

namespace TEST_NAMESPACE {
using namespace sycl_cts;

/* list of special edge cases to test stored as floating
 * point bit patterns.
 *
 * each value in this list will be used for both
 * x and y arguments to a math function.
 *
 * for ( int i=0; i<nEdgeCases*nEdgeCases; i++ )
 * {
 *     xarg = gEdgeCases[ i % nEdgeCases ];
 *     yarg = gEdgeCases[ i / nEdgeCases ];
 *     ...
 * }
 */
// Disable formatting of the array so that it is still aligned correctly
// clang-format off
uint32_t gEdgeCases[] = {
0x7fc00000, 0xff800000, 0xff7fffff, 0xdf800001, 0xdf800000, 0xdf7fffff, 0xdf000001, 0xdf000000, 0xdeffffff,
0xcf800001, 0xcf800000, 0xcf7fffff, 0xcf000001, 0xcf000000, 0xceffffff, 0xc47a0000, 0xc2c80000, 0xc0800000,
0xc0600000, 0xc0400000, 0xc0400001, 0xc0200000, 0xc03fffff, 0xc0000000, 0xbfc00001, 0xbfc00000, 0xbfbfffff,
0xbf800001, 0xbf800000, 0xbf7fffff, 0xbf000001, 0xbf000000, 0xbeffffff, 0xbe800001, 0xbe800000, 0xbe7fffff,
0x80800001, 0x80800000, 0x807fffff, 0x800007ff, 0x8000007f, 0x80000007, 0x80000006, 0x80000005, 0x80000004,
0x80000003, 0x80000002, 0x80000001, 0x80000000, 0xffc00000, 0x7f800000, 0x7f7fffff, 0x5f800001, 0x5f800000,
0x5f7fffff, 0x5f000001, 0x5f000000, 0x5effffff, 0x4f800001, 0x4f800000, 0x4f7fffff, 0x4f000001, 0x4f000000,
0x4effffff, 0x447a0000, 0x42c80000, 0x40800000, 0x40600000, 0x40400000, 0x40400001, 0x40200000, 0x403fffff,
0x40000000, 0x3fc00001, 0x3fc00000, 0x3fbfffff, 0x3f800001, 0x3f800000, 0x3f7fffff, 0x3f000001, 0x3f000000,
0x3effffff, 0x3e800001, 0x3e800000, 0x3e7fffff, 0x00800001, 0x00800000, 0x007fffff, 0x000007ff, 0x0000007f,
0x00000007, 0x00000006, 0x00000005, 0x00000004, 0x00000003, 0x00000002, 0x00000001, 0x00000000
};
// clang-format on

const uint32_t nEdgeCases = sizeof(gEdgeCases) / sizeof(uint32_t);

template <class T, int length>
uint32_t type_length(const cl::sycl::vec<T, length> &ref) {
  return ref.get_count();
}

uint32_t type_length(const float &ref) { return 1; }

uint32_t type_length(const double &ref) { return 1; }

template <class type_t, int length>
void type_set(cl::sycl::vec<type_t, length> &ref, const uint32_t index,
              type_t val) {
  assert(index < length);
  ::setElement(ref, index, val);
}

void type_set(float &ref, uint32_t index, float val) {
  assert(index == 0);
  ref = val;
}

void type_set(double &ref, uint32_t index, double val) {
  assert(index == 0);
  ref = val;
}

template <typename T, int n>
T length(cl::sycl::vec<T, n> v) {
  T total = 0.0;
  for (int i = 0; i < n; ++i) total += ::getElement(v, i);
  return sqrt(total);
}

template <typename T, int n>
cl::sycl::vec<T, n> normalize(cl::sycl::vec<T, n> v) {
  cl::sycl::vec<T, n> ret;
  T len = length(v);
  for (int i = 0; i < n; ++i) {
    ::setElement(ret, i, (::getElement(v, i) / len));
  }
  return ret;
}

/**
 */
template <typename type_t, uint32_t size>
struct test_buffer {
  type_t m_buffer[size];

  test_buffer() { memset(m_buffer, 0, sizeof(m_buffer)); }

  type_t &operator[](uint32_t ix) {
    assert(ix >= 0 && ix < size);
    return m_buffer[ix];
  }
};

template <typename T>
bool verify_func(cl::sycl::vec<T, 2> &input, cl::sycl::vec<T, 2> &param) {
  float ulpTolerance = 8194.f;

  cl::sycl::float2 res = TEST_NAMESPACE::normalize(param);
  float errorTolerance;

  errorTolerance = ulpTolerance *
                   fmaxf(fmaxf(::getElement(param, 1), ::getElement(param, 1)),
                         ::getElement(param, 2));

  cl::sycl::float2 errs;
  for (int i = 0; i < 2; ++i) {
    ::setElement(errs, i, fabsf(::getElement(input, i) - ::getElement(res, i)));
    if (::getElement(errs, i) > errorTolerance) return false;
  }

  return true;
}

template <typename T>
bool verify_func(cl::sycl::vec<T, 4> &input, cl::sycl::vec<T, 4> &param) {
  float ulpTolerance = 8194.f;

  cl::sycl::float4 res = TEST_NAMESPACE::normalize(param);
  float errorTolerance;

  errorTolerance = ulpTolerance *
                   fmaxf(fmaxf(::getElement(param, 1), ::getElement(param, 1)),
                         ::getElement(param, 2));

  cl::sycl::float4 errs;
  for (int i = 0; i < 3; ++i) {
    ::setElement(errs, i, fabsf(::getElement(input, i) - ::getElement(res, i)));
    if (::getElement(errs, i) > errorTolerance) return false;
  }

  return true;
}

/**
 */
template <typename type_t>
struct test_class {
#if WIMPY_MODE
  /* wimpy mode constant */
  static const uint32_t nScale = 0x100;
#else
  /* full (brutal) mode constant */
  static const uint32_t nScale = 0x1;
#endif
  /* size of the host side buffer */
  static const uint32_t nBufferSize = 0x1000;

  /* size of the device side buffer */
  static const uint32_t nSubBufferSize = 0x100;

  static const uint32_t num_params_k = 2;
  static const uint32_t buffer_size_k = 1024;

  /* parameter buffer */
  util::UNIQUE_PTR<type_t[]> m_param_1;

  /* results buffer */
  util::UNIQUE_PTR<type_t[]> m_output;

  /* record the max ULP error */
  float m_max_ulp;

  MTdata m_randData;

  uint64_t m_index;
  uint64_t m_edge_index;

  uint64_t type_mask() {
    if (typeid(typename type_t::element_type) == typeid(float))
      return ~((1ull << 32) - 1);

#if SYCL_CTS_TEST_DOUBLE
    if (typeid(typename type_t::element_type) == typeid(double))
      return ~((1ull << 32) - 1);
#endif

#if SYCL_CTS_TEST_HALF
    if (typeid(typename type_t::element_type) == typeid(cl::sycl::half))
      return ~((1ull << 16) - 1);
#endif

    assert(!"Should not reach here");
    return 0;
  }

  test_class()
      : m_param_1(nullptr),
        m_output(nullptr),
        m_max_ulp(0.f),
        m_randData(),
        m_index(0),
        m_edge_index(0) {}

  /**
   */
  typename type_t::element_type generate_scalar(uint32_t x) {
    return typename type_t::element_type(math::int_to_float(x));
  }

  void generate(util::logger &log) {
    assert(m_param_1.get() != nullptr);
    assert(m_output.get() != nullptr);

    const uint32_t nElms = type_length(type_t());

    /* fill data buffer */
    for (uint32_t i = 0; i < nBufferSize; i++) {
      /* access buffers */
      type_t &e_param1 = m_param_1[i];
      type_t &e_output = m_output[i];

      /* for each element */
      for (uint32_t j = 0; j < nElms; j++) {
        /* truncate the index value to type range */
        uint64_t ix = uint64_t(m_index & ~type_mask());

        if (m_edge_index < (nEdgeCases * nEdgeCases)) {
          ix = m_edge_index++;
          /* use the edge case list */
          type_set(e_param1, j, generate_scalar(gEdgeCases[ix % nEdgeCases]));
        } else {
          /* generate new parameter value */
          type_set(e_param1, j, generate_scalar(genrand_int32(m_randData)));
        }

        m_index += nScale;

        /* clear the output buffer */
        type_set(e_output, j, static_cast<typename type_t::element_type>(1.f));
      }
    }
  }

  void execute(util::logger &log, cl::sycl::queue &sycl_queue) {
    cl::sycl::buffer<type_t, 1> buf_output(m_output.get(),
                                           cl::sycl::range<1>(buffer_size_k));
    cl::sycl::buffer<type_t, 1> buf_param_1(m_param_1.get(),
                                            cl::sycl::range<1>(buffer_size_k));

    sycl_queue.submit([&](cl::sycl::handler &cgh) {
      auto acc_output =
          buf_output
              .template get_access<cl::sycl::access::mode::read_write,
                                   cl::sycl::access::target::global_buffer>(
                  cgh);
      auto acc_param_1 =
          buf_param_1
              .template get_access<cl::sycl::access::mode::read,
                                   cl::sycl::access::target::global_buffer>(
                  cgh);

      cgh.parallel_for<test_class>(cl::sycl::range<1>(buffer_size_k),
                                   [=](cl::sycl::id<1> id) {
                                     type_t &out = acc_output[id];
                                     type_t &pr1 = acc_param_1[id];

                                     out = cl::sycl::fast_normalize(pr1);
                                   });
    });
  }

  void verify(util::logger &log) {
    for (uint32_t i = 0; i < buffer_size_k; i++) {
      if (!verify_func(m_output[i], m_param_1[i])) {
        FAIL(log, "verification failed");
        break;
      }
    }
  }

  void run(util::logger &log) {
    try {
      const uint32_t num_itts = 100;

      /* create device selector */
      cts_selector l_selector;

      /* create command queue */
      cl::sycl::queue l_queue(l_selector);

      for (int i = 0; i < num_itts; i++) {
        generate(log);
        execute(log, l_queue);
        verify(log);

        if (log.has_failed()) break;
      }
      l_queue.wait_and_throw();
    } catch (cl::sycl::exception e) {
      e.what();
    }
  }

  /** clear values required during testing
   */
  bool setup(util::logger &log) {
    m_max_ulp = 0.f;

    m_param_1.reset(new type_t[nBufferSize]);
    assert(m_param_1.get() != nullptr);
    memset(m_param_1.get(), 0, nBufferSize * sizeof(type_t));

    m_output.reset(new type_t[nBufferSize]);
    assert(m_output.get() != nullptr);
    memset(m_output.get(), 0,
           nBufferSize * sizeof(typename type_t::element_type));

    m_randData = init_genrand(0);

    return true;
  }

  /** release all test resources
   */
  void cleanup() {
    m_param_1.reset(nullptr);
    m_output.reset(nullptr);
  }
};

/**
 */
class TEST_NAME : public util::test_base {
 public:
  /** return information about this test
   */
  virtual void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute the test
   */
  virtual void run(util::logger &log) override {
    using namespace cl::sycl;

    try {
      /* create device selector */
      cts_selector l_selector;

      /* create command queue */
      cl::sycl::queue sycl_queue(l_selector);

      test_class<cl::sycl::float2> testf2;
      if (!testf2.setup(log)) return;
      testf2.run(log);
      testf2.cleanup();

      test_class<cl::sycl::float4> testf4;
      if (!testf4.setup(log)) return;
      testf4.run(log);
      testf4.cleanup();

#ifdef SYCL_CTS_TEST_DOUBLE

      test_class<cl::sycl::double2> testd2;
      if (!testd2.setup(log)) return;
      testd2.run(log);
      testd2.cleanup();

      test_class<cl::sycl::double4> testd4;
      if (!testd4.setup(log)) return;
      testd4.run(log);
      testd4.cleanup();
    }

#endif

#ifdef SYCL_CTS_TEST_HALF

    test_class<cl::sycl::half2> testh2;
    if (!testh2.setup(log)) return;
    testh2.run(log);
    testh2.cleanup();

    test_class<cl::sycl::half4> testh4;
    if (!testh4.setup(log)) return;
    testh4.run(log);
    testh4.cleanup();
#endif

    sycl_queue.wait_and_throw();
  }
  catch (cl::sycl::exception e) {
    log_exception(log, e);
    FAIL(log, "sycl exception caught");
  }
}
};

util::test_proxy<TEST_NAME> proxy;

} /* namespace math_geometric_normalize__ */
