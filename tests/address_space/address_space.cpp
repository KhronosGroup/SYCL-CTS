/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

// NOTE: THIS TEST WILL BE REFACTORED

#include "../common/common.h"

#include <cassert>
#include <type_traits>

#define EXPECT_EQUALS(lhs, rhs) \
  if ((lhs) != (rhs)) return false;

#define EXPECT_ADDRSPACE_EQUALS(expr, as) EXPECT_EQUALS((expr).get_value(), as)

#define TEST_NAME address_space

namespace address_space__ {
using namespace sycl_cts;
using namespace cl::sycl::access;

#if defined(__SYCL_DEVICE_ONLY__)

template <cl::sycl::access::address_space kAs>
struct AddrSpace {
  constexpr cl::sycl::access::address_space get_value() const { return kAs; }
};

// global functions
AddrSpace<cl::sycl::access::address_space::global_space>
f(cl::sycl::multi_ptr<int,
                      cl::sycl::access::address_space::global_space>::pointer) {
  return {};
}
AddrSpace<cl::sycl::access::address_space::local_space>
f(cl::sycl::multi_ptr<int,
                      cl::sycl::access::address_space::local_space>::pointer) {
  return {};
}
AddrSpace<cl::sycl::access::address_space::constant_space> f(
    cl::sycl::multi_ptr<
        int, cl::sycl::access::address_space::constant_space>::pointer) {
  return {};
}
AddrSpace<cl::sycl::access::address_space::private_space> f(
    cl::sycl::multi_ptr<
        int, cl::sycl::access::address_space::private_space>::pointer) {
  return {};
}

struct C {
  /*implicit*/
  C() = default;

  // constructors
  cl::sycl::access::address_space m_addrspace;

  explicit C(cl::sycl::multi_ptr<
             int, cl::sycl::access::address_space::global_space>::pointer)
      : m_addrspace(cl::sycl::access::address_space::global_space) {}
  explicit C(cl::sycl::multi_ptr<
             int, cl::sycl::access::address_space::local_space>::pointer)
      : m_addrspace(cl::sycl::access::address_space::local_space) {}
  explicit C(cl::sycl::multi_ptr<
             int, cl::sycl::access::address_space::constant_space>::pointer)
      : m_addrspace(cl::sycl::access::address_space::constant_space) {}
  explicit C(cl::sycl::multi_ptr<
             int, cl::sycl::access::address_space::private_space>::pointer)
      : m_addrspace(cl::sycl::access::address_space::private_space) {}

  // methods
  AddrSpace<cl::sycl::access::address_space::global_space> g(
      cl::sycl::multi_ptr<
          int, cl::sycl::access::address_space::global_space>::pointer) {
    return {};
  }

  AddrSpace<cl::sycl::access::address_space::local_space> g(
      cl::sycl::multi_ptr<
          int, cl::sycl::access::address_space::local_space>::pointer) {
    return {};
  }

  AddrSpace<cl::sycl::access::address_space::constant_space> g(
      cl::sycl::multi_ptr<
          int, cl::sycl::access::address_space::constant_space>::pointer) {
    return {};
  }

  AddrSpace<cl::sycl::access::address_space::private_space> g(
      cl::sycl::multi_ptr<
          int, cl::sycl::access::address_space::private_space>::pointer) {
    return {};
  }

  // operators
  AddrSpace<cl::sycl::access::address_space::global_space> operator+(
      cl::sycl::multi_ptr<
          int, cl::sycl::access::address_space::global_space>::pointer) {
    return {};
  }

  AddrSpace<cl::sycl::access::address_space::local_space> operator+(
      cl::sycl::multi_ptr<
          int, cl::sycl::access::address_space::local_space>::pointer) {
    return {};
  }

  AddrSpace<cl::sycl::access::address_space::constant_space> operator+(
      cl::sycl::multi_ptr<
          int, cl::sycl::access::address_space::constant_space>::pointer) {
    return {};
  }

  AddrSpace<cl::sycl::access::address_space::private_space> operator+(
      cl::sycl::multi_ptr<
          int, cl::sycl::access::address_space::private_space>::pointer) {
    return {};
  }
};

bool test_overload_resolution(
    cl::sycl::multi_ptr<
        int, cl::sycl::access::address_space::global_space>::pointer globalPtr,
    cl::sycl::multi_ptr<
        int, cl::sycl::access::address_space::local_space>::pointer localPtr,
    cl::sycl::multi_ptr<int, cl::sycl::access::address_space::constant_space>::
        pointer constantPtr,
    cl::sycl::multi_ptr<
        int, cl::sycl::access::address_space::private_space>::pointer privPtr) {
  // functions
  EXPECT_ADDRSPACE_EQUALS(f(globalPtr),
                          cl::sycl::access::address_space::global_space);
  EXPECT_ADDRSPACE_EQUALS(f(localPtr),
                          cl::sycl::access::address_space::local_space);
  EXPECT_ADDRSPACE_EQUALS(f(constantPtr),
                          cl::sycl::access::address_space::constant_space);
  EXPECT_ADDRSPACE_EQUALS(f(privPtr),
                          cl::sycl::access::address_space::private_space);

  // methods
  C c;
  EXPECT_ADDRSPACE_EQUALS(c.g(globalPtr),
                          cl::sycl::access::address_space::global_space);
  EXPECT_ADDRSPACE_EQUALS(c.g(localPtr),
                          cl::sycl::access::address_space::local_space);
  EXPECT_ADDRSPACE_EQUALS(c.g(constantPtr),
                          cl::sycl::access::address_space::constant_space);
  EXPECT_ADDRSPACE_EQUALS(c.g(privPtr),
                          cl::sycl::access::address_space::private_space);

  // constructors
  EXPECT_EQUALS(C{globalPtr}.m_addrspace,
                cl::sycl::access::address_space::global_space);
  EXPECT_EQUALS(C{localPtr}.m_addrspace,
                cl::sycl::access::address_space::local_space);
  EXPECT_EQUALS(C{constantPtr}.m_addrspace,
                cl::sycl::access::address_space::constant_space);
  EXPECT_EQUALS(C{privPtr}.m_addrspace,
                cl::sycl::access::address_space::private_space);

  // operators
  EXPECT_ADDRSPACE_EQUALS(c + globalPtr,
                          cl::sycl::access::address_space::global_space);
  EXPECT_ADDRSPACE_EQUALS(c + localPtr,
                          cl::sycl::access::address_space::local_space);
  EXPECT_ADDRSPACE_EQUALS(c + constantPtr,
                          cl::sycl::access::address_space::constant_space);
  EXPECT_ADDRSPACE_EQUALS(c + privPtr,
                          cl::sycl::access::address_space::private_space);

  return true;
}

// the parameter has no address space specification, so it'll
// get duplication based on the argument's address space
// it should call the appropiate overload of f(), and that will
// tell us the address space the parameter was in
template <typename T,
          typename std::enable_if<std::is_pointer<T>::value, void> * = nullptr>
cl::sycl::access::address_space getAddrSpace(T p) {
  return f(p).get_value();
}

struct D {
  template <typename T, typename std::enable_if<std::is_pointer<T>::value, void>
                            * = nullptr>
  cl::sycl::access::address_space getAddrSpace(T p) {
    return address_space__::getAddrSpace(p);
  }

  template <typename T, typename std::enable_if<std::is_pointer<T>::value, void>
                            * = nullptr>
  cl::sycl::access::address_space operator[](T p) {
    return address_space__::getAddrSpace(p);
  }
};

bool test_duplication(
    cl::sycl::multi_ptr<
        int, cl::sycl::access::address_space::global_space>::pointer globalPtr,
    cl::sycl::multi_ptr<
        int, cl::sycl::access::address_space::local_space>::pointer localPtr,
    cl::sycl::multi_ptr<int, cl::sycl::access::address_space::constant_space>::
        pointer constantPtr,
    cl::sycl::multi_ptr<
        int, cl::sycl::access::address_space::private_space>::pointer privPtr) {
  EXPECT_EQUALS(getAddrSpace(globalPtr),
                cl::sycl::access::address_space::global_space);
  EXPECT_EQUALS(getAddrSpace(localPtr),
                cl::sycl::access::address_space::local_space);
  EXPECT_EQUALS(getAddrSpace(constantPtr),
                cl::sycl::access::address_space::constant_space);
  EXPECT_EQUALS(getAddrSpace(privPtr),
                cl::sycl::access::address_space::private_space);

  D d;
  EXPECT_EQUALS(d.getAddrSpace(globalPtr),
                cl::sycl::access::address_space::global_space);
  EXPECT_EQUALS(d.getAddrSpace(localPtr),
                cl::sycl::access::address_space::local_space);
  EXPECT_EQUALS(d.getAddrSpace(constantPtr),
                cl::sycl::access::address_space::constant_space);
  EXPECT_EQUALS(d.getAddrSpace(privPtr),
                cl::sycl::access::address_space::private_space);

  EXPECT_EQUALS(d[globalPtr], cl::sycl::access::address_space::global_space);
  EXPECT_EQUALS(d[localPtr], cl::sycl::access::address_space::local_space);
  EXPECT_EQUALS(d[constantPtr],
                cl::sycl::access::address_space::constant_space);
  EXPECT_EQUALS(d[privPtr], cl::sycl::access::address_space::private_space);

  return true;
}

template <typename T,
          typename std::enable_if<std::is_pointer<T>::value, void> * = nullptr>
T id(T p) {
  return p;
}

bool test_return_type_deduction(
    cl::sycl::multi_ptr<
        int, cl::sycl::access::address_space::global_space>::pointer globalPtr,
    cl::sycl::multi_ptr<
        int, cl::sycl::access::address_space::local_space>::pointer localPtr,
    cl::sycl::multi_ptr<int, cl::sycl::access::address_space::constant_space>::
        pointer constantPtr,
    cl::sycl::multi_ptr<
        int, cl::sycl::access::address_space::private_space>::pointer privPtr) {
  // return type deduction
  EXPECT_EQUALS(getAddrSpace(id(globalPtr)),
                cl::sycl::access::address_space::global_space);
  EXPECT_EQUALS(getAddrSpace(id(localPtr)),
                cl::sycl::access::address_space::local_space);
  EXPECT_EQUALS(getAddrSpace(id(constantPtr)),
                cl::sycl::access::address_space::constant_space);
  EXPECT_EQUALS(getAddrSpace(id(privPtr)),
                cl::sycl::access::address_space::private_space);

  return true;
}

bool test_initialization(
    cl::sycl::multi_ptr<
        int, cl::sycl::access::address_space::global_space>::pointer globalPtr,
    cl::sycl::multi_ptr<
        int, cl::sycl::access::address_space::local_space>::pointer localPtr,
    cl::sycl::multi_ptr<int, cl::sycl::access::address_space::constant_space>::
        pointer constantPtr,
    cl::sycl::multi_ptr<
        int, cl::sycl::access::address_space::private_space>::pointer privPtr) {
  auto p1 = globalPtr;
  auto p2 = localPtr;
  auto p3 = constantPtr;
  auto p4 = privPtr;

  EXPECT_EQUALS(getAddrSpace(p1),
                cl::sycl::access::address_space::global_space);
  EXPECT_EQUALS(getAddrSpace(p2), cl::sycl::access::address_space::local_space);
  EXPECT_EQUALS(getAddrSpace(p3),
                cl::sycl::access::address_space::constant_space);
  EXPECT_EQUALS(getAddrSpace(p4),
                cl::sycl::access::address_space::private_space);

  return true;
}

void test(bool &pass) {
  pass = test_overload_resolution(nullptr, nullptr, nullptr, nullptr);
  if (!pass) return;
  pass = test_duplication(nullptr, nullptr, nullptr, nullptr);
  if (!pass) return;
  pass = test_return_type_deduction(nullptr, nullptr, nullptr, nullptr);
  if (!pass) return;
  pass = test_initialization(nullptr, nullptr, nullptr, nullptr);
  if (!pass) return;
}

#else

void test(bool &pass) {
  // this test is supposed to only work on the device compiler.
  pass = true;
}

#endif

class TEST_NAME : public sycl_cts::util::test_base {
 public:
  virtual void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  virtual void run(util::logger &log) override {
    bool pass = false;
    cl::sycl::range<1> r(1);
    try {
      cl::sycl::buffer<bool, 1> buf(&pass, r);
      auto q = util::get_cts_object::queue();

      q.submit([&](cl::sycl::handler &cgh) {
        auto acc = buf.get_access<cl::sycl::access::mode::read_write>(cgh);

        cgh.single_task<TEST_NAME>([=]() {
          bool b = acc[0];
          test(b);
          acc[0] = b;
        });
      });

      q.wait_and_throw();
    } catch (cl::sycl::exception e) {
      log_exception(log, e);
      cl::sycl::string_class errorMsg =
          "a SYCL exception was caught: " + cl::sycl::string_class(e.what());
      FAIL(log, errorMsg.c_str());
    }

    if (!pass) FAIL(log, "Device compiler failed address space tests");

    return;
  }
};

util::test_proxy<TEST_NAME> proxy;

} /* namespace address_space */
