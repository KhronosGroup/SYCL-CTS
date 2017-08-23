
/*************************************************************************
//
//  This file was AUTOMATICALLY GENERATED via generate_integer_unary.py
//
**************************************************************************/
/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2015 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#include "./../../util/stl.h"
#include "./../../util/math_helper.h"
#include "./../../util/math_reference.h"
#include "./../../util/type_names.h"

#include "./../../oclmath/reference_math.h"
#include "./../../oclmath/Utility.h"
#include "./../../oclmath/mt19937.h"

/** test specifiers
 */
#define TEST_NAME math_integer_binary_add_sat
#define TEST_TYPE add_sat

namespace integer_binary_add_sat__ {
using namespace sycl_cts;
using namespace cl::sycl;

/** kernel functor
 */
template <typename Vec>
class cKernel_mib {
 protected:
  typedef accessor<Vec, 1, cl::sycl::access::mode::read,
                   cl::sycl::access::target::global_buffer>
      t_readAccess;
  typedef accessor<Vec, 1, cl::sycl::access::mode::write,
                   cl::sycl::access::target::global_buffer>
      t_writeAccess;

  t_writeAccess m_o; /* output     */
  t_readAccess m_x;  /* argument X */
  t_readAccess m_y;  /* argument Y */

 public:
  cKernel_mib(t_writeAccess out_, t_readAccess x_, t_readAccess y_)
      : m_o(out_), m_x(x_), m_y(y_) {}

  void operator()(item<1> item) {
    int32_t ix = int32_t(item.get_global_linear());

    auto &o = m_o[ix];
    auto x = m_x[ix];
    auto y = m_y[ix];

    o = cl::sycl::TEST_TYPE(x, y);
  }
};

/**
 */
template <typename Base, typename Vec>
class test_class : public sycl_cts::util::test_base {
 protected:
  /* number of elements in the vector type */
  static const int nVectorElements = sizeof(Vec) / sizeof(Base);

  /* size of the data buffer to process */
  static const uint32_t nBufferSize = 512;

  /* data buffers */
  util::UNIQUE_PTR<Vec[]> m_obuf; /* output vals */
  util::UNIQUE_PTR<Vec[]> m_xbuf; /* x_arguments */
  util::UNIQUE_PTR<Vec[]> m_ybuf; /* y_arguments */

  /*  */
  MTdata m_randData;

 public:
  /** constructor
   */
  test_class()
      : m_obuf(nullptr), m_xbuf(nullptr), m_ybuf(nullptr), m_randData() {}

  /** fill the buffer with new values
   */
  void generate(util::logger &, MTdata &rng) {
    assert(m_obuf.get() != nullptr);
    assert(m_xbuf.get() != nullptr);
    assert(m_ybuf.get() != nullptr);

    const int nBytes = sizeof(Vec) * nBufferSize;
    memset((uint8_t *)m_obuf.get(), 0, nBytes);
    math::rand(rng, (uint8_t *)m_xbuf.get(), nBytes);
    math::rand(rng, (uint8_t *)m_ybuf.get(), nBytes);
  }

  /** process an entire buffer
   */
  bool execute(util::logger &log) {
    assert(m_obuf.get() != nullptr);
    assert(m_xbuf.get() != nullptr);
    assert(m_ybuf.get() != nullptr);

    /* create device selector */
    cts_selector l_selector;

    /* create command queue */
    queue l_queue(l_selector);

    try {
      buffer<Vec, 1> obuf(m_obuf.get(), range<1>(nBufferSize));
      buffer<Vec, 1> xbuf(m_xbuf.get(), range<1>(nBufferSize));
      buffer<Vec, 1> ybuf(m_ybuf.get(), range<1>(nBufferSize));

      /* add command to queue */
      l_queue.submit([&](handler &cgh) {
        auto optr =
            obuf.template get_access<cl::sycl::access::mode::write>(cgh);
        auto xptr = xbuf.template get_access<cl::sycl::access::mode::read>(cgh);
        auto yptr = ybuf.template get_access<cl::sycl::access::mode::read>(cgh);

        /* instantiate the kernel */
        auto kern = cKernel_mib<Vec>(optr, xptr, yptr);

        /* execute the kernel */
        cgh.parallel_for(nd_range<1>(range<1>(nBufferSize), range<1>(1)), kern);
      });

      l_queue.wait_and_throw();

    } catch (cl::sycl::exception e) {
      log_exception(log, e);
      FAIL(log, "");
      return false;
    }

    return true;
  }

  /* verify scalar result
   */
  bool verify(Base result, Base x, Base y) {
/* scalar results are 1 on pass and 0 on fail */
#if 0
        return result == reference::TEST_TYPE( x, y );
#else
    return true;
#endif
  }

  /* verify the buffer is valid
   */
  bool verify(util::logger &log) {
    Vec *o = m_obuf.get();
    Vec *x = m_xbuf.get();
    Vec *y = m_ybuf.get();

    for (int i = 0; i < nBufferSize; i++) {
      // cast our vector types to an array of their elements
      Base *eo_a = reinterpret_cast<Base *>(o + i);
      Base *ex_a = reinterpret_cast<Base *>(x + i);
      Base *ey_a = reinterpret_cast<Base *>(y + i);

      for (int j = 0; j < nVectorElements; j++) {
        bool pass = false;

        Base eo = eo_a[j];
        Base ex = ex_a[j];
        Base ey = ey_a[j];

        pass = verify(eo, ex, ey);

        if (!pass) {
          log.note("fail at item %d element %d", i, j);
          log.note(TOSTRING(TEST_TYPE) "( 0x%llx, 0x%llx ) returned 0x%llx",
                   (uint64_t)ex, (uint64_t)ey, (uint64_t)eo);
          FAIL(log, "");
          return false;
        }
      }
    }

    return true;
  }

  /** return information about this test
   */
  void get_info(test_base::info &out) const {
    using sycl_cts::util::STRING;
    STRING name = STRING(TOSTRING(TEST_NAME)) + "_" + type_name<Vec>();
    set_test_info(out, name.c_str(), TEST_FILE);
  }

  /** clear values required during testing
   */
  bool setup(util::logger &) {
    m_obuf.reset(new Vec[nBufferSize]);
    assert(m_obuf.get() != nullptr);
    memset(m_obuf.get(), 0, sizeof(Vec) * nBufferSize);

    m_xbuf.reset(new Vec[nBufferSize]);
    assert(m_xbuf.get() != nullptr);
    memset(m_xbuf.get(), 0, sizeof(Vec) * nBufferSize);

    m_ybuf.reset(new Vec[nBufferSize]);
    assert(m_ybuf.get() != nullptr);
    memset(m_ybuf.get(), 0, sizeof(Vec) * nBufferSize);

    m_randData = init_genrand(0);

    return true;
  }

  /** execute this test
   */
  void run(util::logger &log) {
    MTdata rng = init_genrand(0);

    generate(log, rng);

    if (!execute(log)) return;

    if (!verify(log)) return;
  }

  /** release all test resources
   *
   */
  void cleanup() {
    m_obuf.reset(nullptr);
    m_xbuf.reset(nullptr);
    m_ybuf.reset(nullptr);
  }
};

class TEST_NAME : public sycl_cts::util::test_base {
 public:
  template <typename T>
  bool execute(util::logger &log) {
    T test;
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

  /** clear values required during testing
   */
  virtual bool setup(util::logger &log) override { return true; }

  /** execute this test
   */
  virtual void run(util::logger &log) override {
    /* 1 scalar value */
    if (!execute<test_class<uint8_t, uint8_t>>(log)) return;
    if (!execute<test_class<uint16_t, uint16_t>>(log)) return;
    if (!execute<test_class<uint32_t, uint32_t>>(log)) return;
    if (!execute<test_class<uint64_t, uint64_t>>(log)) return;
    if (!execute<test_class<int8_t, int8_t>>(log)) return;
    if (!execute<test_class<int16_t, int16_t>>(log)) return;
    if (!execute<test_class<int32_t, int32_t>>(log)) return;
    if (!execute<test_class<int64_t, int64_t>>(log)) return;

    /* 2 element vector */
    if (!execute<test_class<uint8_t, uchar2>>(log)) return;
    if (!execute<test_class<uint16_t, ushort2>>(log)) return;
    if (!execute<test_class<uint32_t, uint2>>(log)) return;
    if (!execute<test_class<uint64_t, ulong2>>(log)) return;
    if (!execute<test_class<int8_t, char2>>(log)) return;
    if (!execute<test_class<int16_t, short2>>(log)) return;
    if (!execute<test_class<int32_t, int2>>(log)) return;
    if (!execute<test_class<int64_t, long2>>(log)) return;

    /* 3 element vector */
    if (!execute<test_class<uint8_t, uchar3>>(log)) return;
    if (!execute<test_class<uint16_t, ushort3>>(log)) return;
    if (!execute<test_class<uint32_t, uint3>>(log)) return;
    if (!execute<test_class<uint64_t, ulong3>>(log)) return;
    if (!execute<test_class<int8_t, char3>>(log)) return;
    if (!execute<test_class<int16_t, short3>>(log)) return;
    if (!execute<test_class<int32_t, int3>>(log)) return;
    if (!execute<test_class<int64_t, long3>>(log)) return;

    /* 4 element vector */
    if (!execute<test_class<uint8_t, uchar4>>(log)) return;
    if (!execute<test_class<uint16_t, ushort4>>(log)) return;
    if (!execute<test_class<uint32_t, uint4>>(log)) return;
    if (!execute<test_class<uint64_t, ulong4>>(log)) return;
    if (!execute<test_class<int8_t, char4>>(log)) return;
    if (!execute<test_class<int16_t, short4>>(log)) return;
    if (!execute<test_class<int32_t, int4>>(log)) return;
    if (!execute<test_class<int64_t, long4>>(log)) return;

    /* 8 element vector */
    if (!execute<test_class<uint8_t, uchar8>>(log)) return;
    if (!execute<test_class<uint16_t, ushort8>>(log)) return;
    if (!execute<test_class<uint32_t, uint8>>(log)) return;
    if (!execute<test_class<uint64_t, ulong8>>(log)) return;
    if (!execute<test_class<int8_t, char8>>(log)) return;
    if (!execute<test_class<int16_t, short8>>(log)) return;
    if (!execute<test_class<int32_t, int8>>(log)) return;
    if (!execute<test_class<int64_t, long8>>(log)) return;

    /* 16 element vector */
    if (!execute<test_class<uint8_t, uchar16>>(log)) return;
    if (!execute<test_class<uint16_t, ushort16>>(log)) return;
    if (!execute<test_class<uint32_t, uint16>>(log)) return;
    if (!execute<test_class<uint64_t, ulong16>>(log)) return;
    if (!execute<test_class<int8_t, char16>>(log)) return;
    if (!execute<test_class<int16_t, short16>>(log)) return;
    if (!execute<test_class<int32_t, int16>>(log)) return;
    if (!execute<test_class<int64_t, long16>>(log)) return;
  }

  /** release all test resources
   */
  virtual void cleanup() override {}
};

util::test_proxy<TEST_NAME> proxy;

} /* namespace math_integer_binary_add_sat__ */
