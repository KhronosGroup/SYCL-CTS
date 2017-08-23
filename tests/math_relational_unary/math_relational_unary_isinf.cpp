
/*************************************************************************
//
//  This file was AUTOMATICALLY GENERATED via generate_math_relational_unary.py
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
#define SCALAR_FUDGE 1
#define TEST_NAME math_relational_unary_isinf
#define TEST_TYPE isinf
#define KERNEL_NAME cKernel_isinf

namespace math_relational_unary_isinf__ {
using namespace sycl_cts;
using namespace cl::sycl;

/** kernel functor
 */
template <typename T_in, typename T_out>
class KERNEL_NAME {
 protected:
  typedef accessor<T_in, 1, cl::sycl::access::mode::read,
                   cl::sycl::access::target::global_buffer>
      t_readAccess;
  typedef accessor<T_out, 1, cl::sycl::access::mode::write,
                   cl::sycl::access::target::global_buffer>
      t_writeAccess;

  t_writeAccess m_o; /* output     */
  t_readAccess m_x;  /* argument X */

 public:
  KERNEL_NAME(t_writeAccess out_, t_readAccess x_) : m_o(out_), m_x(x_) {}

  void operator()(item<1> item) {
    auto &o = m_o[item.get_global()];
    auto x = m_x[item.get_global()];

    o = cl::sycl::TEST_TYPE(x);
  }
};

/**
 */
template <typename T_in, typename T_out>
class test_class {
 protected:
  /* size of the data buffer to process */
  static const uint32_t nBufferSize = 512;

  /* data buffers */
  util::UNIQUE_PTR<T_in[]> m_xbuf;  /* x_arguments */
  util::UNIQUE_PTR<T_out[]> m_obuf; /* output vals */

  /*  */
  MTdata m_randData;

 public:
  /** constructor
   */
  test_class() : m_obuf(nullptr), m_xbuf(nullptr), m_randData() {}

  /** fill the buffer with new values
   */
  void generate(util::logger &log, MTdata &rng) {
    assert(m_obuf.get() != nullptr);
    assert(m_xbuf.get() != nullptr);

    math::rand(rng, m_xbuf.get(), nBufferSize);

    memset(m_xbuf.get(), 0, sizeof(T_out) * nBufferSize);
  }

  /** process an entire buffer
   */
  bool execute(util::logger &log) {
    /* create device selector */
    cts_selector l_selector;

    /* create command queue */
    queue l_queue(l_selector);

    try {
      buffer<T_out, 1> obuf(m_obuf.get(), range<1>(nBufferSize));
      buffer<T_in, 1> xbuf(m_xbuf.get(), range<1>(nBufferSize));

      /* add command to queue */
      l_queue.submit([&](handler &cgh) {
        auto optr =
            obuf.template get_access<cl::sycl::access::mode::write>(cgh);
        auto xptr = xbuf.template get_access<cl::sycl::access::mode::read>(cgh);

        /* instantiate the kernel */
        auto kern = KERNEL_NAME<T_in, T_out>(optr, xptr);

        /* execute the kernel */
        cgh.parallel_for(
            nd_range<1>(range<1>(nBufferSize), range<1>(nBufferSize / 8)),
            kern);
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
  bool verifyScl(int result, float x) {
    /* scalar results are 1 on pass
     * and 0 on fail */
    return result == reference::TEST_TYPE(x);
  }

  /* verify vector result
   */
  bool verifyVec(int result, float x) {
    /* vector results are -1 on pass (all bits set)
     * and 0 on fail */
    return result == -reference::TEST_TYPE(x);
  }

  /* verify the buffer is valid
   */
  bool verify(util::logger &log) {
    auto o = m_obuf.get();
    auto x = m_xbuf.get();

    bool isVectorType = math::numElements(*x) > 1;

    for (int i = 0; i < nBufferSize; i++) {
      for (int j = 0; j < math::numElements(*x); j++) {
        bool pass = false;

        auto eo = math::getElement(o[i], j);
        auto ex = math::getElement(x[i], j);

        /* if the base data type is a vector type */
        if (isVectorType && (!SCALAR_FUDGE)) pass = verifyVec(eo, ex);
        /* if the base data type is a scalar type */
        else
          pass = verifyScl(eo, ex);

        if (!pass) {
          log.note("fail at item %d element %d", i, j);
          log.note(TOSTRING(TEST_TYPE) "( %f ) returned %d", ex, eo);

          FAIL(log, "");
          return false;
        }
      }
    }

    return true;
  }

  /** clear values required during testing
   */
  bool setup(util::logger &log) {
    m_xbuf.reset(new T_in[nBufferSize]);
    assert(m_xbuf.get() != nullptr);

    m_obuf.reset(new T_out[nBufferSize]);
    assert(m_obuf.get() != nullptr);

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
   */
  void cleanup() {
    m_xbuf.reset(nullptr);
    m_obuf.reset(nullptr);
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
    if (!execute<test_class<float, int>>(log)) return;
    if (!execute<test_class<float2, int2>>(log)) return;
    if (!execute<test_class<float3, int3>>(log)) return;
    if (!execute<test_class<float4, int4>>(log)) return;
    if (!execute<test_class<float8, int8>>(log)) return;
    if (!execute<test_class<float16, int16>>(log)) return;
  }

  /** release all test resources
   */
  virtual void cleanup() override {}
};

util::test_proxy<TEST_NAME> proxy;

} /* namespace math_relational_unary_isinf_$NUM_ELMS$__ */
