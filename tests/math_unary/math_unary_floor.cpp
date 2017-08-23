
/*************************************************************************
//
//  This file was AUTOMATICALLY GENERATED via generate_math_unary.py
//
**************************************************************************/
/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:  (c) 2014 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#include "./../../util/stl.h"
#include "./../../util/math_helper.h"
#include "./../../util/math_vector.h"
#include "./../../util/type_names.h"

#include "./../../oclmath/reference_math.h"
#include "./../../oclmath/Utility.h"

/** test specifiers
 */
#define TEST_NAME math_unary_floor
#define WIMPY_MODE 1
#define MAX_ULPS 0.0f
#define HOST_FUNC floor
#define REF_FUNC reference_floor

namespace math_unary_floor__ {
using namespace sycl_cts;
using namespace cl::sycl;

/**
 */
enum {
  e_ulp_func_half = 0,
  e_ulp_func_float,
  e_ulp_func_double,
};

/** kernel functor
 */
template <typename T>
class test_kernel {
 public:
  typedef accessor<T, 1, cl::sycl::access::mode::read,
                   cl::sycl::access::target::global_buffer>
      t_read;
  typedef accessor<T, 1, cl::sycl::access::mode::write,
                   cl::sycl::access::target::global_buffer>
      t_write;

  t_write m_out;
  t_read m_in;

  test_kernel(t_write w_, t_read r_) : m_out(w_), m_in(r_) {}

  void operator()(id<1> item) {
    auto in = m_in[item];
    auto &out = m_out[item];

    out = cl::sycl::HOST_FUNC(in);
  }
};

template <class T, int length>
uint32_t type_length(const vec<T, length> &ref) {
  return ref.get_count();
}

uint32_t type_length(const float &ref) { return 1; }

uint32_t type_length(const double &ref) { return 1; }

template <class type_t, int length>
type_t type_get(vec<type_t, length> &ref, const uint32_t index) {
  assert(index < length);
  return getElement(ref, index);
}

float &type_get(float &ref, uint32_t index) {
  assert(index == 0);
  return ref;
}

double &type_get(double &ref, uint32_t index) {
  assert(index == 0);
  return ref;
}

template <class type_t, int length>
void type_set(vec<type_t, length> &ref, const uint32_t index, type_t val) {
  assert(index < length);
  setElement(ref, static_cast<int>(index), val);
}

void type_set(float &ref, uint32_t index, float val) {
  assert(index == 0);
  ref = val;
}

void type_set(double &ref, uint32_t index, double val) {
  assert(index == 0);
  ref = val;
}

/* ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
 * ---- ---- ---- ---- ---- */

/** unary brute force math test
 */
template <typename type_t, typename base_t, int ulp_test_func>
class test_class {
 protected:
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

  uint64_t m_index;

  /* parameter buffer */
  util::UNIQUE_PTR<type_t[]> m_param_1;

  /* results buffer */
  util::UNIQUE_PTR<type_t[]> m_output;

  /* record the max ULP error */
  float m_max_ulp;

  /**
   */
  base_t generate_scalar(uint32_t x) { return base_t(math::int_to_float(x)); }

  /** fill the buffer with new values
   */
  uint32_t generate(util::logger &log) {
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
        uint32_t ix = uint32_t(m_index & ~type_mask());

        /* generate new parameter value */
        type_set(e_param1, j, generate_scalar(ix));

        m_index += nScale;

        /* clear the output buffer */
        type_set(e_output, j, base_t(1.f));

        /* if we have exceeded 32 bit range */
        if ((m_index & type_mask()) != 0)
          /* the buffer may be only partially filled */
          return i;
      }
    }

    /* we filled the entire buffer */
    return nBufferSize;
  }

  /** pass the buffer through the kernel
   */
  bool execute_chunk(util::logger &log, queue &a_queue,
                     const uint32_t chunkStart, const uint32_t chunkSize) {
    try {
      buffer<type_t, 1> l_buf_param1(m_param_1.get() + chunkStart,
                                     range<1>(chunkSize));
      buffer<type_t, 1> l_buf_output(m_output.get() + chunkStart,
                                     range<1>(chunkSize));

      /* add command to queue */
      a_queue.submit([&](handler &cgh) {
        auto acc_param1 =
            l_buf_param1.template get_access<cl::sycl::access::mode::read>(cgh);
        auto acc_output =
            l_buf_output.template get_access<cl::sycl::access::mode::write>(
                cgh);

        /* instantiate the kernel */
        auto kern = test_kernel<type_t>(acc_output, acc_param1);

        /* execute the kernel */
        cgh.parallel_for(range<1>(chunkSize), kern);
      });

      a_queue.wait_and_throw();

    } catch (cl::sycl::exception e) {
      log_exception(log, e);
      FAIL(log, "");
      return false;
    }

    return true;
  }

  /** process an entire buffer
   */
  bool execute(util::logger &log, queue &a_queue, uint32_t nItems) {
    const uint32_t nChunkSize = 64;

    const uint32_t nItts = (nItems / nChunkSize);

    /* step through chunks */
    for (uint32_t pos = 0; pos < nItts; pos += nChunkSize) {
      /* find a fitting chunk */
      int32_t size = nItems - pos;
      size = (size < nChunkSize) ? size : nChunkSize;

      if (size <= 0) break;

      /* process this chunk */
      if (!execute_chunk(log, a_queue, pos, size)) return false;
    }

    return true;
  }

  /**
   */
  bool verify_scalar(util::logger &log, base_t vut, base_t param) {
    double ref = REF_FUNC((double)param);

    static_assert(ulp_test_func == e_ulp_func_half ||
                      ulp_test_func == e_ulp_func_float ||
                      ulp_test_func == e_ulp_func_double,
                  "unknown ulp reference function");

    float ulp = 100.f;

    if (ulp_test_func == e_ulp_func_half)
      assert(!"half reference not yet implemented");

    if (ulp_test_func == e_ulp_func_float) ulp = Ulp_Error(vut, ref);

    if (ulp_test_func == e_ulp_func_double) ulp = Ulp_Error_Double(vut, ref);

    m_max_ulp = fmaxf(fabsf(ulp), m_max_ulp);

    if (m_max_ulp > MAX_ULPS) return false;

    return true;
  }

  /**
   */
  bool verify(util::logger &log) {
    uint32_t num_elms = type_length(m_param_1[0]);

    for (uint32_t i = 0; i < nBufferSize; i++) {
      for (uint32_t j = 0; j < num_elms; j++) {
        base_t param_1 = type_get(m_param_1[i], j);

        base_t res = type_get(m_output[i], j);

        if (!verify_scalar(log, res, param_1)) return false;
      }
    }

    return true;
  }

 public:
  test_class()
      : m_param_1(nullptr), m_output(nullptr), m_max_ulp(0.f), m_index(0) {}

  /** clear values required during testing
   */
  bool setup(util::logger &log) {
    m_max_ulp = 0.f;

    m_param_1.reset(new type_t[nBufferSize]);
    assert(m_param_1.get() != nullptr);
    memset(m_param_1.get(), 0, nBufferSize * sizeof(type_t));

    m_output.reset(new type_t[nBufferSize]);
    assert(m_output.get() != nullptr);
    memset(m_output.get(), 0, nBufferSize * sizeof(type_t));

    return true;
  }

  /** return a mask with bits being set
   */
  uint64_t type_mask() {
    if (typeid(base_t) == typeid(float)) return ~((1ull << 32) - 1);

#if SYCL_CTS_TEST_DOUBLE
    if (typeid(base_t) == typeid(double)) return ~((1ull << 32) - 1);
#endif

#if SYCL_CTS_TEST_HALF
    if (typeid(base_t) == typeid(half)) return ~((1ull << 16) - 1);
#endif

    assert(!"Should not reach here");
    return 0;
  }

  /** execute this test
   */
  void run(util::logger &log) {
    try {
      /* create device selector */
      cts_selector l_selector;

      /* create command queue */
      queue l_queue(l_selector);

      const uint64_t mask = type_mask();

      /*  */
      do {
        /* send progress report */
        log.progress(int32_t(m_index >> 8), int32_t(mask >> 8));

        /* generate a buffer of x values */
        uint32_t nItems = generate(log);

        /* convert x values to y values */
        if (!execute(log, l_queue, nItems)) {
          FAIL(log, "kernel failed execution");
          break;
        }

        /* compare y values to reference y values */
        if (!verify(log)) {
          FAIL(log, "ULP exceeds tolerance");
          break;
        }
      } while ((m_index & type_mask()) != 0);

      /* send 100% progress report */
      log.progress(int(1), int(1));

      /* record the max ulps */
      log.note("max_ulp = %8.2f", m_max_ulp);
    } catch (cl::sycl::exception e) {
      FAIL(log, "big fail");
    }
  }

  /** release all test resources
   */
  void cleanup() {
    m_param_1.reset(nullptr);
    m_output.reset(nullptr);
  }
};

/* ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
 * ---- ---- ---- ---- ---- */

class TEST_NAME : public sycl_cts::util::test_base {
 public:
  template <typename type_t>
  bool execute(util::logger &log) {
    type_t test;
    if (!test.setup(log)) return false;
    test.run(log);
    test.cleanup();
    return !log.has_failed();
  }

  /** return information about this test
   */
  virtual void get_info(test_base::info &out) const override {
    using sycl_cts::util::STRING;
    STRING name = STRING(TOSTRING(TEST_NAME));
    set_test_info(out, name.c_str(), TEST_FILE);
  }

  /** execute this test
   */
  virtual void run(util::logger &log) override {
#if SYCL_CTS_TEST_HALF
    if (!execute<test_class<half, half, e_ulp_func_half>>(log)) return;
    if (!execute<test_class<half2, half, e_ulp_func_half>>(log)) return;
    if (!execute<test_class<half3, half, e_ulp_func_half>>(log)) return;
    if (!execute<test_class<half4, half, e_ulp_func_half>>(log)) return;
    if (!execute<test_class<half8, half, e_ulp_func_half>>(log)) return;
    if (!execute<test_class<half16, half, e_ulp_func_half>>(log)) return;
#endif

    if (!execute<test_class<float, float, e_ulp_func_float>>(log)) return;
    if (!execute<test_class<float2, float, e_ulp_func_float>>(log)) return;
    if (!execute<test_class<float3, float, e_ulp_func_float>>(log)) return;
    if (!execute<test_class<float4, float, e_ulp_func_float>>(log)) return;
    if (!execute<test_class<float8, float, e_ulp_func_float>>(log)) return;
    if (!execute<test_class<float16, float, e_ulp_func_float>>(log)) return;

#if SYCL_CTS_TEST_DOUBLE
    if (!execute<test_class<double, double, e_ulp_func_double>>(log)) return;
    if (!execute<test_class<double2, double, e_ulp_func_double>>(log)) return;
    if (!execute<test_class<double3, double, e_ulp_func_double>>(log)) return;
    if (!execute<test_class<double4, double, e_ulp_func_double>>(log)) return;
    if (!execute<test_class<double8, double, e_ulp_func_double>>(log)) return;
    if (!execute<test_class<double16, double, e_ulp_func_double>>(log)) return;
#endif
  }
};

util::test_proxy<TEST_NAME> proxy;

} /* namespace math_unary_cos__ */
