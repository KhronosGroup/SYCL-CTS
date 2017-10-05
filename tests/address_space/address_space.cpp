/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#include <cassert>
#include <type_traits>

#define EXPECT_EQUALS(lhs, rhs) \
  if ((lhs) != (rhs)) return false;

#define EXPECT_ADDRSPACE_EQUALS(expr, as) \
  EXPECT_EQUALS(expr, (AddrSpace<as>()).get_value())

#define TEST_NAME address_space

namespace TEST_NAME {
using namespace sycl_cts;

template <cl::sycl::access::address_space kAs>
struct AddrSpace {
  constexpr int get_value() const { return static_cast<int>(kAs); }
};

// the parameter has no address space specification, so it'll
// get duplicated based on the argument's address space
// it should call the appropiate overload of f(), and that will
// tell us the address space the parameter was in

int readValue(int *p) { return *p; }

int readValue(int &p) { return p; }

struct SomeStruct {
  int readValue(int *p) { return *p; }

  int readValue(int &p) { return p; }

  int operator[](int *p) { return *p; }

  int operator[](int &p) { return p; }
};

template <cl::sycl::access::address_space AspSpace>
bool test_duplication(cl::sycl::multi_ptr<int, AspSpace> ptr) {
  EXPECT_ADDRSPACE_EQUALS(readValue(ptr), AspSpace);
  EXPECT_ADDRSPACE_EQUALS(readValue(*ptr), AspSpace);

  SomeStruct d;
  EXPECT_ADDRSPACE_EQUALS(d.readValue(ptr), AspSpace);
  EXPECT_ADDRSPACE_EQUALS(d.readValue(*ptr), AspSpace);
  EXPECT_ADDRSPACE_EQUALS(d[ptr], AspSpace);
  EXPECT_ADDRSPACE_EQUALS(d[*ptr], AspSpace);

  return true;
}

bool test_duplication(cl::sycl::global_ptr<int> globalPtr,
                      cl::sycl::local_ptr<int> localPtr,
                      cl::sycl::constant_ptr<int> constantPtr,
                      cl::sycl::private_ptr<int> privPtr) {
  test_duplication(globalPtr);
  test_duplication(localPtr);
  test_duplication(constantPtr);
  test_duplication(privPtr);

  return true;
}

template <cl::sycl::access::address_space AspSpace>
int *id(cl::sycl::multi_ptr<int, AspSpace> p) {
  return p;
}

bool test_return_type_deduction(cl::sycl::global_ptr<int> globalPtr,
                                cl::sycl::local_ptr<int> localPtr,
                                cl::sycl::constant_ptr<int> constantPtr,
                                cl::sycl::private_ptr<int> privPtr) {
  // return type deduction
  EXPECT_ADDRSPACE_EQUALS(*id(globalPtr),
                          cl::sycl::access::address_space::global_space);
  EXPECT_ADDRSPACE_EQUALS(*id(localPtr),
                          cl::sycl::access::address_space::local_space);
  EXPECT_ADDRSPACE_EQUALS(*id(constantPtr),
                          cl::sycl::access::address_space::constant_space);
  EXPECT_ADDRSPACE_EQUALS(*id(privPtr),
                          cl::sycl::access::address_space::private_space);

  return true;
}

bool test_initialization(cl::sycl::global_ptr<int> globalPtr,
                         cl::sycl::local_ptr<int> localPtr,
                         cl::sycl::constant_ptr<int> constantPtr,
                         cl::sycl::private_ptr<int> privPtr) {
  int *p1 = globalPtr;
  int *p2 = localPtr;
  int *p3 = constantPtr;
  int *p4 = privPtr;

  EXPECT_ADDRSPACE_EQUALS(*p1, cl::sycl::access::address_space::global_space);
  EXPECT_ADDRSPACE_EQUALS(*p2, cl::sycl::access::address_space::local_space);
  EXPECT_ADDRSPACE_EQUALS(*p3, cl::sycl::access::address_space::constant_space);
  EXPECT_ADDRSPACE_EQUALS(*p4, cl::sycl::access::address_space::private_space);

  return true;
}

class TEST_NAME : public sycl_cts::util::test_base {
 public:
  void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  void run(util::logger &log) override {
    bool pass = false;
    std::array<int, 4> ASPValues = {
        AddrSpace<cl::sycl::access::address_space::global_space>().get_value(),
        AddrSpace<cl::sycl::access::address_space::constant_space>()
            .get_value(),
        AddrSpace<cl::sycl::access::address_space::local_space>().get_value(),
        AddrSpace<cl::sycl::access::address_space::private_space>()
            .get_value()};

    cl::sycl::range<1> r(1);

    try {
      cl::sycl::buffer<bool, 1> resBuff(&pass, r);
      cl::sycl::buffer<int, 1> initBuff(ASPValues.data(), ASPValues.size());
      cl::sycl::buffer<int, 1> globalBuff(&ASPValues[0], r);
      cl::sycl::buffer<int, 1> constantBuff(&ASPValues[1], r);

      auto q = util::get_cts_object::queue();

      q.submit([&](cl::sycl::handler &cgh) {
        auto resAcc =
            resBuff.get_access<cl::sycl::access::mode::read_write>(cgh);
        auto initAcc = initBuff.get_access<cl::sycl::access::mode::read>(cgh);
        auto globalAcc =
            globalBuff.get_access<cl::sycl::access::mode::read>(cgh);
        cl::sycl::accessor<int, 1, cl::sycl::access::mode::read,
                           cl::sycl::access::target::constant_buffer>
            constAcc(constantBuff, cgh);
        cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write,
                           cl::sycl::access::target::local>
            localAcc(r, cgh);

        cgh.single_task<TEST_NAME>([=]() {
          bool pass = resAcc[0];
          localAcc[0] = initAcc[2];
          int priv = initAcc[3];

          pass = test_duplication(
              globalAcc.get_pointer(), localAcc.get_pointer(),
              constAcc.get_pointer(), cl::sycl::private_ptr<int>(&priv));
          if (!pass) return;
          pass = test_return_type_deduction(
              globalAcc.get_pointer(), localAcc.get_pointer(),
              constAcc.get_pointer(), cl::sycl::private_ptr<int>(&priv));
          if (!pass) return;
          pass = test_initialization(
              globalAcc.get_pointer(), localAcc.get_pointer(),
              constAcc.get_pointer(), cl::sycl::private_ptr<int>(&priv));
          if (!pass) return;
          resAcc[0] = pass;
        });
      });

      q.wait_and_throw();
    } catch (const cl::sycl::exception &e) {
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

} /* namespace TEST_NAME */
