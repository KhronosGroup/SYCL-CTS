/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Common code for multi_ptr members tests
//
*******************************************************************************/

#ifndef SYCL_CTS_TEST_MULTI_PTR_MEMBERS_H
#define SYCL_CTS_TEST_MULTI_PTR_MEMBERS_H

#include "../common/common.h"
#include "../common/get_cts_string.h"

namespace multi_ptr_members {

/** @brief Verify that sycl::multi_ptr members are equal to:
 *          1) sycl::multi_ptr::value_type is same as provided element type
 *          2) sycl::multi_ptr::iterator_category is same as
 *            std::random_access_iterator_tag
 *          3) sycl::multi_ptr::difference_type is same
 *            as std::ptrdiff_t If decorate address is access::decorated::no:
 *            1)sycl::multi_ptr::pointer is same as std::add_pointer_t<
 *              value_type>
 *            2)sycl::multi_ptr::reference is same as
 *              std::add_lvalue_reference_t<value_type>
 *  @tparam VariableT Variable type for type coverage
 *  @tparam AddressSpace Field of sycl::access::address_space namespace for
 *          multi ptr
 *  @tparam Decorated Field of sycl::access::decorated namespace for
 *          multi ptr
 *  @param log sycl_cts::util::logger class object
 *  @param type_name a string representing the currently tested type
 */
template <typename VariableT,
          sycl::access::address_space AddressSpace,
          sycl::access::decorated Decorated>
static void verify_members(sycl_cts::util::logger &log,
                           const std::string &type_name) {
  auto multi_ptr{
      sycl::multi_ptr<VariableT, AddressSpace, Decorated>()};

  std::string address_type_str{
      std::string(sycl_cts::get_cts_string::for_address_space<AddressSpace>())};
  std::string decorated_str{
      std::string(sycl_cts::get_cts_string::for_decorated<Decorated>())};
  std::string log_suffix{" with address type: " + address_type_str +
                         ", with decorated type: " + decorated_str +
                         ", with tested type: " + type_name};
  if (!std::is_same_v<typename decltype(multi_ptr)::value_type, VariableT>) {
    FAIL(log,
         "sycl::multi_ptr::value_type doesn't equal to provided value type" +
             log_suffix);
  }
  if (!std::is_same_v<typename decltype(multi_ptr)::difference_type,
                      std::ptrdiff_t>) {
    FAIL(log,
         "sycl::multi_ptr::difference_type doesn't equal to std::ptrdiff_t" +
             log_suffix);
  }
  if (!std::is_same_v<typename decltype(multi_ptr)::iterator_category,
                      std::random_access_iterator_tag>) {
    FAIL(log,
         "sycl::multi_ptr::iterator_category doesn't equal to "
         "std::random_access_iterator_tag" +
             log_suffix);
  }

  if constexpr (Decorated == sycl::access::decorated::no) {
    if (!std::is_same_v<typename decltype(multi_ptr)::pointer,
                        std::add_pointer_t<VariableT>>) {
      FAIL(log,
           "sycl::multi_ptr::pointer doesen't equal to "
           "std::add_pointer_t<value_type>" +
               log_suffix);
    }
    if (!std::is_same_v<typename decltype(multi_ptr)::reference,
                        std::add_lvalue_reference_t<VariableT>>) {
      FAIL(log,
           "sycl::multi_ptr::reference doesn't equal to "
           "std::add_lvalue_reference_t<value_type>" +
               log_suffix);
    }
  }
}

/** @brief Dummy struct with overloaded call operator that will be called in
 *         "for_all_types" function
 *  @tparam VariableT Variable type for type coverage
 */
template <typename VariableT>
struct run_test_with_chosen_data_type {
  /** @brief Run verification's function with provided variable type and
   *         sycl::access::address_space and sycl::access::decorated
   *         enumerations fields
   *  @param log sycl_cts::util::logger class object
   *  @param type_name a string representing the currently tested type
   */
  void operator()(sycl_cts::util::logger &log, const std::string &type_name) {
    verify_members<VariableT, sycl::access::address_space::global_space,
                   sycl::access::decorated::yes>(log, type_name);
    verify_members<VariableT, sycl::access::address_space::global_space,
                   sycl::access::decorated::no>(log, type_name);
    verify_members<VariableT, sycl::access::address_space::local_space,
                   sycl::access::decorated::yes>(log, type_name);
    verify_members<VariableT, sycl::access::address_space::local_space,
                   sycl::access::decorated::no>(log, type_name);
    verify_members<VariableT, sycl::access::address_space::private_space,
                   sycl::access::decorated::yes>(log, type_name);
    verify_members<VariableT, sycl::access::address_space::private_space,
                   sycl::access::decorated::no>(log, type_name);
    verify_members<VariableT, sycl::access::address_space::generic_space,
                   sycl::access::decorated::yes>(log, type_name);
    verify_members<VariableT, sycl::access::address_space::generic_space,
                   sycl::access::decorated::no>(log, type_name);
  }
};

}  // namespace multi_ptr_members

#endif  // SYCL_CTS_TEST_MULTI_PTR_MEMBERS_H
