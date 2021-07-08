/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#include <array>
#include <cassert>
#include <type_traits>

#define EXPECT_EQUALS(lhs, rhs) \
  if ((lhs) != (rhs)) return false;

#define EXPECT_ADDRSPACE_EQUALS(expr, as) \
  EXPECT_EQUALS(expr, (AddrSpace<as>()).get_value())

#define TEST_NAME address_space

namespace TEST_NAME {
using namespace sycl_cts;

template <sycl::access::address_space kAs>
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

template <sycl::access::address_space AspSpace>
bool test_duplication(sycl::multi_ptr<int, AspSpace> ptr) {
  EXPECT_ADDRSPACE_EQUALS(readValue(ptr), AspSpace);
  EXPECT_ADDRSPACE_EQUALS(readValue(*ptr), AspSpace);

  SomeStruct d;
  EXPECT_ADDRSPACE_EQUALS(d.readValue(ptr), AspSpace);
  EXPECT_ADDRSPACE_EQUALS(d.readValue(*ptr), AspSpace);
  EXPECT_ADDRSPACE_EQUALS(d[ptr], AspSpace);
  EXPECT_ADDRSPACE_EQUALS(d[*ptr], AspSpace);

  return true;
}

bool test_duplication(sycl::global_ptr<int> globalPtr,
                      sycl::local_ptr<int> localPtr,
                      sycl::constant_ptr<int> constantPtr,
                      sycl::private_ptr<int> privPtr) {
  test_duplication(globalPtr);
  test_duplication(localPtr);
  test_duplication(constantPtr);
  test_duplication(privPtr);

  return true;
}

template <sycl::access::address_space AspSpace>
int *id(sycl::multi_ptr<int, AspSpace> p) {
  return p;
}

bool test_return_type_deduction(sycl::global_ptr<int> globalPtr,
                                sycl::local_ptr<int> localPtr,
                                sycl::constant_ptr<int> constantPtr,
                                sycl::private_ptr<int> privPtr) {
  // return type deduction
  EXPECT_ADDRSPACE_EQUALS(*id(globalPtr),
                          sycl::access::address_space::global_space);
  EXPECT_ADDRSPACE_EQUALS(*id(localPtr),
                          sycl::access::address_space::local_space);
  EXPECT_ADDRSPACE_EQUALS(*id(constantPtr),
                          sycl::access::address_space::constant_space);
  EXPECT_ADDRSPACE_EQUALS(*id(privPtr),
                          sycl::access::address_space::private_space);

  return true;
}

bool test_initialization(sycl::global_ptr<int> globalPtr,
                         sycl::local_ptr<int> localPtr,
                         sycl::constant_ptr<int> constantPtr,
                         sycl::private_ptr<int> privPtr) {
  int *p1 = globalPtr;
  int *p2 = localPtr;
  int *p3 = constantPtr;
  int *p4 = privPtr;

  EXPECT_ADDRSPACE_EQUALS(*p1, sycl::access::address_space::global_space);
  EXPECT_ADDRSPACE_EQUALS(*p2, sycl::access::address_space::local_space);
  EXPECT_ADDRSPACE_EQUALS(*p3, sycl::access::address_space::constant_space);
  EXPECT_ADDRSPACE_EQUALS(*p4, sycl::access::address_space::private_space);

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
        AddrSpace<sycl::access::address_space::global_space>().get_value(),
        AddrSpace<sycl::access::address_space::constant_space>()
            .get_value(),
        AddrSpace<sycl::access::address_space::local_space>().get_value(),
        AddrSpace<sycl::access::address_space::private_space>()
            .get_value()};

    sycl::range<1> r(1);

    try {
      sycl::buffer<bool, 1> resBuff(&pass, r);
      sycl::buffer<int, 1> initBuff(ASPValues.data(), ASPValues.size());
      sycl::buffer<int, 1> globalBuff(&ASPValues[0], r);
      sycl::buffer<int, 1> constantBuff(&ASPValues[1], r);

      auto q = util::get_cts_object::queue();

      q.submit([&](sycl::handler &cgh) {
        auto resAcc =
            resBuff.get_access<sycl::access_mode::read_write>(cgh);
        auto initAcc = initBuff.get_access<sycl::access_mode::read>(cgh);
        auto globalAcc =
            globalBuff.get_access<sycl::access_mode::read>(cgh);
        sycl::accessor<int, 1, sycl::access_mode::read,
                           sycl::target::constant_buffer>
            constAcc(constantBuff, cgh);
        sycl::accessor<int, 1, sycl::access_mode::read_write,
                           sycl::target::local>
            localAcc(r, cgh);

        cgh.single_task<TEST_NAME>([=]() {
          bool pass = resAcc[0];
          localAcc[0] = initAcc[2];
          int priv = initAcc[3];

          pass = test_duplication(
              globalAcc.get_pointer(), localAcc.get_pointer(),
              constAcc.get_pointer(), sycl::private_ptr<int>(&priv));
          if (!pass) return;
          pass = test_return_type_deduction(
              globalAcc.get_pointer(), localAcc.get_pointer(),
              constAcc.get_pointer(), sycl::private_ptr<int>(&priv));
          if (!pass) return;
          pass = test_initialization(
              globalAcc.get_pointer(), localAcc.get_pointer(),
              constAcc.get_pointer(), sycl::private_ptr<int>(&priv));
          if (!pass) return;
          resAcc[0] = pass;
        });
      });

      q.wait_and_throw();
    } catch (const sycl::exception &e) {
      log_exception(log, e);
      std::string errorMsg =
          "a SYCL exception was caught: " + std::string(e.what());
      FAIL(log, errorMsg.c_str());
    }

    if (!pass) FAIL(log, "Device compiler failed address space tests");

    return;
  }
};

util::test_proxy<TEST_NAME> proxy;

} /* namespace TEST_NAME */
