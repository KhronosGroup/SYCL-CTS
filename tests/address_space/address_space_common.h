/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provide common code for address_space verification
//
*******************************************************************************/

#ifndef SYCL_CTS_TESTS_ADDRESS_SPACE_ADDRESS_SPACE_COMMON_H
#define SYCL_CTS_TESTS_ADDRESS_SPACE_ADDRESS_SPACE_COMMON_H

#include "../../util/kernel_names.h"
#include "../common/common.h"

#include <array>
#include <string>
#include <type_traits>

#define EXPECT_EQUALS(lhs, rhs) \
  if ((lhs) != (rhs)) return false;

#define EXPECT_ADDRSPACE_EQUALS(expr, as) \
  EXPECT_EQUALS(expr, (AddrSpace<as>()).get_value())

#ifndef TEST_NAME
#error Invalid test namespace
#endif

namespace TEST_NAMESPACE {
using namespace sycl_cts;

template <typename T>
struct address_space_kernel {};

template <typename T>
class check_types {
  using kernel_name =
      address_space_kernel<sycl_cts::util::kernel_name::adapter_t<T>>;

 public:
  template <sycl::access::address_space kAs>
  struct AddrSpace {
    constexpr T get_value() const { return static_cast<T>(to_integral(kAs)); }
  };

  static T readValue(T *p) { return *p; }

  static T readValue(T &p) { return p; }

  struct SomeStruct {
    T readValue(T *p) { return *p; }

    T readValue(T &p) { return p; }

    T operator[](T *p) { return *p; }

    T operator[](T &p) { return p; }
  };

  /** @brief Verify value in multi_ptr is the same one used for initialization
   */
  template <sycl::access::address_space AspSpace>
  static bool test_duplication(sycl::multi_ptr<T, AspSpace> ptr) {
    // Get pointer by calling “operator pointer() const” in a different ways
    // Dereference raw pointer and verify value

    EXPECT_ADDRSPACE_EQUALS(readValue(ptr), AspSpace);
    EXPECT_ADDRSPACE_EQUALS(readValue(*ptr), AspSpace);

    SomeStruct d;
    EXPECT_ADDRSPACE_EQUALS(d.readValue(ptr), AspSpace);
    EXPECT_ADDRSPACE_EQUALS(d.readValue(*ptr), AspSpace);
    EXPECT_ADDRSPACE_EQUALS(d[ptr], AspSpace);
    EXPECT_ADDRSPACE_EQUALS(d[*ptr], AspSpace);

    return true;
  }

  static bool test_duplication(sycl::global_ptr<T> globalPtr,
                               sycl::local_ptr<T> localPtr,
                               sycl::constant_ptr<T> constantPtr,
                               sycl::private_ptr<T> privPtr) {
    bool result = true;

    result &= test_duplication(globalPtr);
    result &= test_duplication(localPtr);
    result &= test_duplication(constantPtr);
    result &= test_duplication(privPtr);

    return result;
  }

  template <sycl::access::address_space AspSpace>
  static T *id(sycl::multi_ptr<T, AspSpace> p) {
    // Get pointer by calling “operator pointer() const” for multi_ptr instance
    return p;
  }

  /** @brief Verify value in multi_ptr is the same one used for initialization
   */
  static bool test_return_type_deduction(sycl::global_ptr<T> globalPtr,
                                         sycl::local_ptr<T> localPtr,
                                         sycl::constant_ptr<T> constantPtr,
                                         sycl::private_ptr<T> privPtr) {
    using namespace sycl::access;

    // Get pointer by calling id() function
    // Dereference raw pointer and verify value
    EXPECT_ADDRSPACE_EQUALS(*id(globalPtr), address_space::global_space);
    EXPECT_ADDRSPACE_EQUALS(*id(localPtr), address_space::local_space);
    EXPECT_ADDRSPACE_EQUALS(*id(constantPtr), address_space::constant_space);
    EXPECT_ADDRSPACE_EQUALS(*id(privPtr), address_space::private_space);

    return true;
  }

  /** @brief Verify value in multi_ptr is the same one used for initialization
   */
  static bool test_initialization(sycl::global_ptr<T> globalPtr,
                                  sycl::local_ptr<T> localPtr,
                                  sycl::constant_ptr<T> constantPtr,
                                  sycl::private_ptr<T> privPtr) {
    using namespace sycl::access;

    // Get pointer by calling “operator pointer() const” for multi_ptr instance
    T *p1 = globalPtr;
    T *p2 = localPtr;
    T *p3 = constantPtr;
    T *p4 = privPtr;

    // Dereference raw pointer and verify value
    EXPECT_ADDRSPACE_EQUALS(*p1, address_space::global_space);
    EXPECT_ADDRSPACE_EQUALS(*p2, address_space::local_space);
    EXPECT_ADDRSPACE_EQUALS(*p3, address_space::constant_space);
    EXPECT_ADDRSPACE_EQUALS(*p4, address_space::private_space);

    return true;
  }

  /** @brief Run checks for a specific type
   *  @details Every check sequence is as follows:
   *    - Initialize array with N integer values
   *    - Create N multi_ptr instances pointing to N array elements
   *    - Verify value in multi_ptr is the same one used for initialization
   *  @todo We need to review and probably fix the logic of the checks
   */
  bool operator()() {
    using namespace sycl::access;

    bool pass = false;

    std::array<T, 4> ASPValues = {
        AddrSpace<address_space::global_space>().get_value(),
        AddrSpace<address_space::constant_space>().get_value(),
        AddrSpace<address_space::local_space>().get_value(),
        AddrSpace<address_space::private_space>().get_value()};

    auto q = util::get_cts_object::queue();

    {
      auto r = sycl::range<1>(1);
      sycl::buffer<bool, 1> resBuff(&pass, r);
      sycl::buffer<T, 1> initBuff(ASPValues.data(), ASPValues.size());
      sycl::buffer<T, 1> globalBuff(&ASPValues[0], r);
      sycl::buffer<T, 1> constantBuff(&ASPValues[1], r);

      q.submit([&](sycl::handler &cgh) {
        constexpr auto read_only = sycl::access_mode::read;
        constexpr auto read_write = sycl::access_mode::read_write;

        auto resAcc = resBuff.get_access<read_write>(cgh);
        auto initAcc = initBuff.template get_access<read_only>(cgh);
        auto globalAcc = globalBuff.template get_access<read_only>(cgh);
        sycl::accessor<T, 1, read_only, sycl::target::constant_buffer> constAcc(
            constantBuff, cgh);
        sycl::accessor<T, 1, read_write, sycl::target::local> localAcc(r, cgh);

        cgh.single_task<kernel_name>([=]() {
          bool pass = true;
          localAcc[0] = initAcc[2];
          T priv = initAcc[3];

          pass &= test_duplication(
              globalAcc.get_pointer(), localAcc.get_pointer(),
              constAcc.get_pointer(), sycl::private_ptr<T>(&priv));
          pass &= test_return_type_deduction(
              globalAcc.get_pointer(), localAcc.get_pointer(),
              constAcc.get_pointer(), sycl::private_ptr<T>(&priv));
          pass &= test_initialization(
              globalAcc.get_pointer(), localAcc.get_pointer(),
              constAcc.get_pointer(), sycl::private_ptr<T>(&priv));
          resAcc[0] = pass;
        });
      });
    }
    return pass;
  }
};

template <typename T>
void test_types(util::logger &log) {
  {
    check_types<T> verifier;
    if (!verifier()) FAIL(log, "Device compiler failed address space tests");
  }
  return;
}

}  // namespace TEST_NAMESPACE

#endif  // SYCL_CTS_TESTS_ADDRESS_SPACE_ADDRESS_SPACE_COMMON_H
