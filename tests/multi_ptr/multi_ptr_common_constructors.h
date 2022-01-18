/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for multi_ptr common constructors
//
*******************************************************************************/

#ifndef __SYCLCTS_TESTS_MULTI_PTR_COMMON_CONSTRUCTORS_H
#define __SYCLCTS_TESTS_MULTI_PTR_COMMON_CONSTRUCTORS_H

#include "../common/common.h"
#include "../common/get_cts_string.h"
#include "../common/type_list.h"

namespace multi_ptr_common_constructors {

template <typename T, sycl::access::address_space Space,
          sycl::access::decorated Decorated>
struct multi_ptr_kernel_name;

/** @brief Provides text description of test case in case of fail
 *  @tparam Space sycl::access::address_space value
 *  @tparam Decorated sycl::access::decorated value
 */
template <sycl::access::address_space Space, sycl::access::decorated Decorated>
std::string get_case_description(const std::string &info, size_t overload_index,
                                 const std::string &type_name) {
  static std::vector<std::string> overloads{
      "multi_ptr()", "multi_ptr(const multi_ptr&)", "multi_ptr(multi_ptr&&)",
      "multi_ptr(multi_ptr<ElementType, Space, yes>::pointer)",
      "multi_ptr(std::nullptr_t)"};
  std::string addr_space{sycl_cts::get_cts_string::for_address_space<Space>()};
  std::string decor{sycl_cts::get_cts_string::for_decorated<Decorated>()};
  std::string message{info + " of get() for " + overloads[overload_index] +
                      " with tparams: <" + addr_space + "> <" + decor +
                      "> and type: " + type_name};
  return message;
}

template <sycl::access::address_space Space>
constexpr bool use_local_accessor =
    (Space == sycl::access::address_space::local_space) ||
    (Space == sycl::access::address_space::private_space);

/** @brief Provides verification of multi_ptr common constructors with template
 *         parameters given
 *  @tparam T Variable type for type coverage
 *  @tparam Space sycl::access::address_space value
 *  @tparam Decorated sycl::access::decorated value
 */
template <typename T, sycl::access::address_space Space,
          sycl::access::decorated Decorated>
void run_tests(sycl_cts::util::logger &log, const std::string &type_name) {
  using namespace sycl_cts;
  using m_ptr_t = sycl::multi_ptr<T, Space, Decorated>;
  using other_decorated_ptr_t =
      sycl::multi_ptr<T, Space, sycl::access::decorated::yes>;

  // Kernel name
  using k_name = multi_ptr_kernel_name<T, Space, Decorated>;

  // In the test, there are 5 verifies for type correctness and only 4 verifies
  // for value correctness because we can't predict what value will contain
  // multi_ptr created with the default constructor
  constexpr size_t types_size = 5;
  constexpr size_t values_size = 4;
  // Arrays for result values
  bool same_type[types_size]{};
  bool same_value[values_size]{};
  T ref_value{user_def_types::get_init_value_helper<T>(0)};
  auto queue = util::get_cts_object::queue();

  using CommonAccType = sycl::accessor<T, 1, sycl::access_mode::read>;
  using LocalAccType = sycl::local_accessor<T, 1>;

  // Accessor for ref value depending on sycl::access::address_space
  using RefAccType =
      typename std::conditional<use_local_accessor<Space>, LocalAccType,
                                CommonAccType>::type;
  using ResultAccType = sycl::accessor<bool, 1, sycl::access_mode::write>;

  // Main check lambda
  auto run_and_check = [=](RefAccType ref_acc, ResultAccType same_type_acc,
                           ResultAccType same_value_acc) {
    // For indexing result arrays
    size_t type_i = 0;
    size_t value_i = 0;
    // Check default constructor
    {
      m_ptr_t mptr;
      same_type_acc[type_i++] =
          std::is_same_v<decltype(mptr.get()), typename m_ptr_t::pointer>;
    }

    // Check multi_ptr(const multi_ptr&)
    {
      m_ptr_t other(ref_acc);
      m_ptr_t mptr(other);

      same_type_acc[type_i++] =
          std::is_same_v<decltype(mptr.get()), typename m_ptr_t::pointer>;
      same_value_acc[value_i++] = (*(mptr.get()) == *(other.get()));
    }

    // Check multi_ptr(multi_ptr&&)
    {
      m_ptr_t other(ref_acc);
      auto other_get = *(other.get());
      m_ptr_t mptr(std::move(other));

      same_type_acc[type_i++] =
          std::is_same_v<decltype(mptr.get()), typename m_ptr_t::pointer>;
      same_value_acc[value_i++] = (*(mptr.get()) == other_get);
    }

    // Check explicit multi_ptr(multi_ptr<ElementType, Space, yes>::pointer)
    {
      other_decorated_ptr_t other(ref_acc);
      m_ptr_t mptr(other.get());

      same_type_acc[type_i++] =
          std::is_same_v<decltype(mptr.get()), typename m_ptr_t::pointer>;
      same_value_acc[value_i++] = (*(mptr.get()) == *(other.get()));
    }

    // Check multi_ptr(std::nullptr_t)
    {
      m_ptr_t mptr(nullptr);
      same_type_acc[type_i++] =
          std::is_same_v<decltype(mptr.get()), typename m_ptr_t::pointer>;
      same_value_acc[value_i++] = (mptr.get() == nullptr);
    }
  };

  {
    sycl::range<1> types_range(types_size);
    sycl::range<1> values_range(values_size);
    sycl::buffer<T, 1> ref_buf(&ref_value, sycl::range<1>{1});
    sycl::buffer<bool, 1> same_type_buf(same_type, types_range);
    sycl::buffer<bool, 1> same_value_buf(same_value, values_range);

    queue.submit([&](sycl::handler &cgh) {
      auto ref_acc = ref_buf.template get_access<sycl::access_mode::read>(cgh);
      auto same_type_acc =
          same_type_buf.template get_access<sycl::access_mode::write>(cgh);
      auto same_value_acc =
          same_value_buf.template get_access<sycl::access_mode::write>(cgh);

      if constexpr (use_local_accessor<Space>) {
        sycl::local_accessor<T, 1> loc_acc(sycl::range<1>(1), cgh);
        cgh.parallel_for<k_name>(sycl::range<1>(1), [=](sycl::item<1>) {
          loc_acc[0] = ref_acc[0];
          run_and_check(loc_acc, same_type_acc, same_value_acc);
        });
      } else {
        cgh.single_task<k_name>(
            [=] { run_and_check(ref_acc, same_type_acc, same_value_acc); });
      }
    });
  }
  for (size_t i = 0; i < types_size; ++i) {
    if (!same_type[i]) {
      FAIL(log, (get_case_description<Space, Decorated>("Incorrect type", i,
                                                        type_name)));
    }
  }
  for (size_t i = 0; i < values_size; ++i) {
    if (!same_value[i]) {
      // We use i + 1 overload index because we have one extra check for
      // multi_ptr::pointer type for default constructor
      FAIL(log, (get_case_description<Space, Decorated>("Incorrect value",
                                                        i + 1, type_name)));
    }
  }
}

/** @brief Provides verification of multi_ptr common constructors for type given
 *  @tparam T Variable type for type coverage
 */
template <typename T>
class check_multi_ptr_common_constructors_for_type {
 public:
  void operator()(sycl_cts::util::logger &log, const std::string &type_name) {
    run_tests<T, sycl::access::address_space::global_space,
              sycl::access::decorated::yes>(log, type_name);
    run_tests<T, sycl::access::address_space::local_space,
              sycl::access::decorated::yes>(log, type_name);
    run_tests<T, sycl::access::address_space::private_space,
              sycl::access::decorated::yes>(log, type_name);
    run_tests<T, sycl::access::address_space::generic_space,
              sycl::access::decorated::yes>(log, type_name);
    run_tests<T, sycl::access::address_space::constant_space,
              sycl::access::decorated::yes>(log, type_name);
    run_tests<T, sycl::access::address_space::global_space,
              sycl::access::decorated::no>(log, type_name);
    run_tests<T, sycl::access::address_space::local_space,
              sycl::access::decorated::no>(log, type_name);
    run_tests<T, sycl::access::address_space::private_space,
              sycl::access::decorated::no>(log, type_name);
    run_tests<T, sycl::access::address_space::generic_space,
              sycl::access::decorated::no>(log, type_name);
    run_tests<T, sycl::access::address_space::constant_space,
              sycl::access::decorated::no>(log, type_name);
  }
};

}  // namespace multi_ptr_common_constructors

#endif  // __SYCLCTS_TESTS_MULTI_PTR_COMMON_CONSTRUCTORS_H
