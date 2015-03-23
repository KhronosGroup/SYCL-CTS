/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2015 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME atomic_api

namespace cl {
namespace sycl {
template <typename T>
using atomic = std::atomic<T>;

using namespace std;
}
}

namespace TEST_NAMESPACE {
using namespace sycl_cts;
using namespace cl::sycl;

template <typename T, int target>
class functor_single;

template <typename T, int target>
class functor_parallel;

/* Specialsation for float because most operations are int/uint only */
template <int target>
class functor_single<float, target> {
  cl::sycl::accessor<float, 1, access::atomic, target> m_acc;

 public:
  functor_single(cl::sycl::accessor<float, 1, access::atomic, target> acc)
      : m_acc(acc) {}

  void operator()() {
    std::memory_order mor = std::memory_order_relaxed;
    cl::sycl::atomic<float> a = m_acc[0];
    float old = a.load(mor);
    a.store(0.f, mor);
    old = a.exchange(1.f, mor);

    old = atomic_load_explicit(&a, mor);
    atomic_store_explicit(&a, 0.f, mor);
    old = atomic_exchange_explicit(&a, 1.f, mor);
  }
};

template <int target>
class functor_parallel<float, target> {
  cl::sycl::accessor<float, 1, access::atomic, target> m_acc;

 public:
  functor_parallel(cl::sycl::accessor<float, 1, access::atomic, target> acc)
      : m_acc(acc) {}

  void operator()(id<1> i) {
    std::memory_order mor = std::memory_order_relaxed;
    volatile cl::sycl::atomic<float> a = m_acc[0];
    float old = a.load(mor);
    a.store(0.f, mor);
    old = a.exchange(1.f, mor);

    old = atomic_load_explicit(&a, mor);
    atomic_store_explicit(&a, 0.f, mor);
    old = atomic_exchange_explicit(&a, 1.f, mor);
  }
};

template <typename T, int target>
class functor_single {
  cl::sycl::accessor<T, 1, access::atomic, target> m_acc;

 public:
  functor_single(cl::sycl::accessor<T, 1, access::atomic, target> acc)
      : m_acc(acc) {}

  void operator()() {
    std::memory_order mor = std::memory_order_relaxed;
    cl::sycl::atomic<T> a = m_acc[0];
    T old = a.load(mor);
    a.store(0, mor);
    old = a.exchange(1, mor);
    bool res = a.compare_exchange_strong(old, 1, mor, mor);
    old = a.fetch_add(1, mor);
    old = a.fetch_sub(1, mor);
    old = a.fetch_and(0xFFFFFFFF, mor);
    old = a.fetch_or(0x00000000, mor);
    old = a.fetch_xor(0xFFFFFFFE, mor);
    old = a.fetch_min(0xFFFFFFFF, mor);
    old = a.fetch_max(0x00000000, mor);

    old = atomic_load_explicit(&a, mor);
    atomic_store_explicit(&a, 0, mor);
    old = atomic_exchange_explicit(&a, 1, mor);
    old = atomic_compare_exchange_strong_explicit(&a, &old, 1, mor, mor);
    old = atomic_fetch_add_explicit(&a, 1, mor);
    old = atomic_fetch_sub_explicit(&a, 1, mor);
    old = atomic_fetch_and_explicit(&a, 0xFFFFFFFF, mor);
    old = atomic_fetch_or_explicit(&a, 0x00000000, mor);
    old = atomic_fetch_xor_explicit(&a, 0xFFFFFFFE, mor);
    old = atomic_fetch_min_explicit(&a, 0xFFFFFFFF, mor);
    old = atomic_fetch_max_explicit(&a, 0x00000000, mor);
  }
};

template <typename T, int target>
class functor_parallel {
  cl::sycl::accessor<T, 1, access::atomic, target> m_acc;

 public:
  functor_parallel(cl::sycl::accessor<T, 1, access::atomic, target> acc)
      : m_acc(acc) {}

  void operator()(id<1> i) {
    std::memory_order mor = std::memory_order_relaxed;
    volatile cl::sycl::atomic<T> a = m_acc[0];
    bool res = false;

    T old = a.load(mor);
    a.store(0, mor);
    old = a.exchange(1, mor);
    res = a.compare_exchange_strong(old, 1, mor, mor);
    old = a.fetch_add(1, mor);
    old = a.fetch_sub(1, mor);
    old = a.fetch_and(0xFFFFFFFF, mor);
    old = a.fetch_or(0x00000000, mor);
    old = a.fetch_xor(0xFFFFFFFE, mor);
    old = a.fetch_min(0xFFFFFFFF, mor);
    old = a.fetch_max(0x00000000, mor);

    old = atomic_load_explicit(&a, mor);
    atomic_store_explicit(&a, 0, mor);
    old = atomic_exchange_explicit(&a, 1, mor);
    res = atomic_compare_exchange_strong_explicit(&a, &old, 1, mor, mor);
    old = atomic_fetch_add_explicit(&a, 1, mor);
    old = atomic_fetch_sub_explicit(&a, 1, mor);
    old = atomic_fetch_and_explicit(&a, 0xFFFFFFFF, mor);
    old = atomic_fetch_or_explicit(&a, 0x00000000, mor);
    old = atomic_fetch_xor_explicit(&a, 0xFFFFFFFE, mor);
    old = atomic_fetch_min_explicit(&a, 0xFFFFFFFF, mor);
    old = atomic_fetch_max_explicit(&a, 0x00000000, mor);
  }
};

template <typename T, int target>
class atomics {
 public:
  void operator()(util::logger &log, queue &q) {
    T data = 0;
    memset(&data, 0xFF, sizeof(T));

    // Single element to test concurrent access
    range<1> r(1);
    buffer<T, 1> buf(&data, r);

    q.submit([&](cl::sycl::handler &cgh) {
      accessor<T, 1, access::atomic, target> acc(buf, cgh);

      functor_parallel<T, target> f(acc);

      cgh.parallel_for(r, f);
    });

    q.submit([&](cl::sycl::handler &cgh) {
      accessor<T, 1, access::atomic, target> acc(buf, cgh);

      functor_single<T, target> f(acc);

      cgh.single_task(f);
    });

    accessor<T, 1, access::atomic, cl::sycl::access::target::host_buffer> h_acc(
        buf);

    auto a = h_acc[0];
    if (typeid(a) != typeid(cl::sycl::atomic<T>))
      FAIL(log,
           "cl::sycl::accessor does not return atomic type "
           "for access mode == atomic");
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

  template <typename T>
  void test_type(util::logger &log, cl::sycl::queue q) {
    atomics<T, cl::sycl::access::target::global_buffer> a_g;
    a_g(log, q);
  }

  /** execute the test
   */
  virtual void run(util::logger &log) override {
    try {
      cl::sycl::queue q;
      test_type<int>(log, q);
      test_type<unsigned int>(log, q);
      test_type<float>(log, q);

      q.wait_and_throw();

    } catch (cl::sycl::exception e) {
      log_exception(log, e);
      FAIL(log, "sycl exception caught");
    }
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

} /* namespace atomic_api__ */
