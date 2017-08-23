/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"
#include <sstream>

#define TEST_NAME atomic_api

namespace TEST_NAMESPACE {
using namespace sycl_cts;
using namespace cl::sycl;

template <typename T, cl::sycl::access::target target>
class functor_single;

template <typename T, cl::sycl::access::target target>
class functor_parallel;

/* Specialisation for float because most operations are int/uint only */
template <cl::sycl::access::target target>
class functor_single<float, target> {
  cl::sycl::accessor<float, 1, access::mode::atomic, target> m_acc;

 public:
  functor_single(cl::sycl::accessor<float, 1, access::mode::atomic, target> acc)
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

template <cl::sycl::access::target target>
class functor_parallel<float, target> {
  cl::sycl::accessor<float, 1, access::mode::atomic, target> m_acc;

 public:
  functor_parallel(
      cl::sycl::accessor<float, 1, access::mode::atomic, target> acc)
      : m_acc(acc) {}

  void operator()(id<1> i) {
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

template <typename T, cl::sycl::access::target target>
class functor_single {
  cl::sycl::accessor<T, 1, access::mode::atomic, target> m_acc;

 public:
  functor_single(cl::sycl::accessor<T, 1, access::mode::atomic, target> acc)
      : m_acc(acc) {}

  void operator()() {
    std::memory_order mor = std::memory_order_relaxed;
    cl::sycl::atomic<T> a = m_acc[0];
    T old = a.load(mor);
    a.store(static_cast<T>(0), mor);
    old = a.exchange(static_cast<T>(1), mor);
    bool res = a.compare_exchange_strong(old, static_cast<T>(1), mor, mor);
    old = a.fetch_add(static_cast<T>(1), mor);
    old = a.fetch_sub(static_cast<T>(1), mor);
    old = a.fetch_and(static_cast<T>(0xFFFFFFFF), mor);
    old = a.fetch_or(static_cast<T>(0), mor);
    old = a.fetch_xor(static_cast<T>(0xFFFFFFFE), mor);
    old = a.fetch_min(static_cast<T>(0xFFFFFFFF), mor);
    old = a.fetch_max(static_cast<T>(0), mor);

    old = atomic_load_explicit(&a, mor);
    atomic_store_explicit(&a, static_cast<T>(0), mor);
    old = atomic_exchange_explicit(&a, static_cast<T>(1), mor);
    old = atomic_compare_exchange_strong_explicit(&a, &old, static_cast<T>(1),
                                                  mor, mor);
    old = atomic_fetch_add_explicit(&a, static_cast<T>(1), mor);
    old = atomic_fetch_sub_explicit(&a, static_cast<T>(1), mor);
    old = atomic_fetch_and_explicit(&a, static_cast<T>(0xFFFFFFFF), mor);
    old = atomic_fetch_or_explicit(&a, static_cast<T>(0), mor);
    old = atomic_fetch_xor_explicit(&a, static_cast<T>(0xFFFFFFFE), mor);
    old = atomic_fetch_min_explicit(&a, static_cast<T>(0xFFFFFFFF), mor);
    old = atomic_fetch_max_explicit(&a, static_cast<T>(0), mor);
  }
};

template <typename T, cl::sycl::access::target target>
class functor_parallel {
  cl::sycl::accessor<T, 1, access::mode::atomic, target> m_acc;

 public:
  functor_parallel(cl::sycl::accessor<T, 1, access::mode::atomic, target> acc)
      : m_acc(acc) {}

  void operator()(id<1> i) {
    std::memory_order mor = std::memory_order_relaxed;
    cl::sycl::atomic<T> a = m_acc[0];
    bool res = false;

    T old = a.load(mor);
    a.store(static_cast<T>(0), mor);
    old = a.exchange(static_cast<T>(1), mor);
    res = a.compare_exchange_strong(old, static_cast<T>(1), mor, mor);
    old = a.fetch_add(static_cast<T>(1), mor);
    old = a.fetch_sub(static_cast<T>(1), mor);
    old = a.fetch_and(static_cast<T>(0xFFFFFFFF), mor);
    old = a.fetch_or(static_cast<T>(0), mor);
    old = a.fetch_xor(static_cast<T>(0xFFFFFFFE), mor);
    old = a.fetch_min(static_cast<T>(0xFFFFFFFF), mor);
    old = a.fetch_max(static_cast<T>(0), mor);

    old = atomic_load_explicit(&a, mor);
    atomic_store_explicit(&a, static_cast<T>(0), mor);
    old = atomic_exchange_explicit(&a, static_cast<T>(1), mor);
    res = atomic_compare_exchange_strong_explicit(&a, &old, static_cast<T>(1),
                                                  mor, mor);
    old = atomic_fetch_add_explicit(&a, static_cast<T>(1), mor);
    old = atomic_fetch_sub_explicit(&a, static_cast<T>(1), mor);
    old = atomic_fetch_and_explicit(&a, static_cast<T>(0xFFFFFFFF), mor);
    old = atomic_fetch_or_explicit(&a, static_cast<T>(0), mor);
    old = atomic_fetch_xor_explicit(&a, static_cast<T>(0xFFFFFFFE), mor);
    old = atomic_fetch_min_explicit(&a, static_cast<T>(0xFFFFFFFF), mor);
    old = atomic_fetch_max_explicit(&a, static_cast<T>(0), mor);
  }
};

template <typename T, int dimensions, cl::sycl::access::target target>
class atomics {
 public:
  void operator()(util::logger &log, queue &q) {
    T data = 0;
    ::memset(&data, 0xFF, sizeof(T));

    // Single element to test concurrent access
    range<dimensions> r;
    for (int i = 0; i < dimensions; i++) r[i] = 1;
    buffer<T, dimensions> buf(&data, r);

    q.submit([&](cl::sycl::handler &cgh) {
      accessor<T, dimensions, access::mode::atomic, target> acc(buf, cgh);

      functor_parallel<T, target> f(acc);

      cgh.parallel_for(r, f);
    });

    q.submit([&](cl::sycl::handler &cgh) {
      accessor<T, dimensions, access::mode::atomic, target> acc(buf, cgh);

      functor_single<T, target> f(acc);

      cgh.single_task(f);
    });

    accessor<T, dimensions, access::mode::atomic,
             cl::sycl::access::target::host_buffer>
        h_acc(buf);

    if (dimensions == 1) {
      auto a = h_acc[0];
      check_return_type<cl::sycl::atomic<T>>(log, a, "accessor<T, dimensions, access::mode::atomic,
             cl::sycl::access::target::host_buffer> with dimensions = 1");
    } else {
      id<dimensions> index;
      for (size_t i = 0; i < dimensions; i++) {
        index[i] = 0;
      }
      auto a = h_acc[index];
      check_return_type<cl::sycl::atomic<T>>(log, a, "accessor<T, dimensions, access::mode::atomic,
             cl::sycl::access::target::host_buffer> with dimensions > 1");
    }
  }
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
    {
      atomics<T, 1, cl::sycl::access::target::global_buffer> aG;
      aG(log, q);
    }
    {
      atomics<T, 2, cl::sycl::access::target::global_buffer> aG;
      aG(log, q);
    }
    {
      atomics<T, 3, cl::sycl::access::target::global_buffer> aG;
      aG(log, q);
    }
  }

  /** execute the test
    */
  virtual void run(util::logger &log) override {
    try {
      auto q = util::get_cts_object::queue();
      test_type<int>(log, q);
      test_type<unsigned int>(log, q);
      test_type<float>(log, q);

      q.wait_and_throw();
    } catch (cl::sycl::exception e) {
      log_exception(log, e);
      cl::sycl::string_class errorMsg =
          "a SYCL exception was caught: " + cl::sycl::string_class(e.what());
      FAIL(log, errorMsg.c_str());
    }
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

} /* namespace atomic_api__ */
