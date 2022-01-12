/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for multi_ptr accessor constructors
//
*******************************************************************************/
#ifndef __SYCLCTS_TESTS_MULTI_PTR_ACCESSOR_CONSTRUCTORS_H
#define __SYCLCTS_TESTS_MULTI_PTR_ACCESSOR_CONSTRUCTORS_H

#include "../common/common.h"
#include "../common/get_cts_object.h"
#include "../common/get_cts_string.h"
#include "../common/type_list.h"

namespace multi_ptr_accessor_constructors {
template <typename T, sycl::access::address_space Space,
          sycl::access::decorated DecorateAddress, int dimension,
          sycl::access::mode Mode>
struct multi_ptr_kernel_name;

/** @brief Provides text description of the test case on a failure
 *  @tparam Space sycl::access::address_space value
 *  @tparam Mode sycl::access::mode value
 *  @tparam Target sycl::access::target value
 */
template <int dimension, sycl::access::address_space Space,
          sycl::access::mode Mode, sycl::access::target Target>
std::string get_case_description(const std::string& info,
                                 const std::string& type_name) {
  static std::string overload{
      "multi_ptr(accsessor<value_type, dimensions, sycl::access::mode, "
      "access::target::device, sycl::access::placeholder>)"};
  std::string addr_space{sycl_cts::get_cts_string::for_address_space<Space>()};
  std::string dimension_str{std::to_string(dimension)};
  std::string mode{sycl_cts::get_cts_string::for_mode<Mode>()};
  std::string target{sycl_cts::get_cts_string::for_target<Target>()};
  std::string message{info + " of get() for " + overload + " with tparams: <" +
                      type_name + "> <" + dimension_str + "> <" + addr_space +
                      "> <" + mode + "> <" + target + ">" +
                      "> and type: " + type_name};
  return message;
}

/** @brief Provides verification of multi_ptr accessor constructors with
 * template parameters given
 *  @tparam T Variable type for type coverage
 *  @tparam Space sycl::access::address_space value
 *  @tparam Mode sycl::access::mode value
 */
template <typename T, sycl::access::address_space Space, int dimension,
          sycl::access::mode Mode>
void run_tests(sycl_cts::util::logger& log, const std::string& type_name) {
  using namespace sycl_cts;
  using multi_ptr_t = sycl::multi_ptr<T, Space, sycl::access::decorated::no>;
  // Kernel name
  using kernel = multi_ptr_kernel_name<T, Space, sycl::access::decorated::no,
                                       dimension, Mode>;

  // result values
  bool same_type = false;
  bool same_value = false;

  // default value
  T init_value = user_def_types::get_init_value_helper<T>(10);

  auto queue = util::get_cts_object::queue();

  {
    auto init_val_range =
        sycl_cts::util::get_cts_object::range<dimension>::get(1, 1, 1);
    sycl::buffer<bool, 1> same_type_buf(&same_type, sycl::range<1>(1));
    sycl::buffer<bool, 1> same_value_buf(&same_value, sycl::range<1>(1));
    sycl::buffer<T, dimension> init_val_buffer(&init_value, init_val_range);
    queue.submit([&](sycl::handler& cgh) {
      auto same_type_acc =
          same_type_buf.template get_access<sycl::access_mode::write>(cgh);
      auto same_value_acc =
          same_value_buf.template get_access<sycl::access_mode::write>(cgh);
      auto accessor_instance = init_val_buffer.template get_access<Mode>(cgh);

      cgh.single_task<kernel>([=] {
        // Creating multi_ptr object with accessor constructor
        multi_ptr_t mptr(accessor_instance);

        auto id = util::get_cts_object::id<dimension>::get(0, 0, 0);
        // Check that mptr points at same value as accessor
        same_value_acc[0] = (*(mptr.get()) == accessor_instance[id]);

        // Check that type of value is correct
        same_type_acc[0] =
            std::is_same_v<decltype(mptr.get()), typename multi_ptr_t::pointer>;
      });
    });
  }
  if (!same_type) {
    FAIL(log, (get_case_description<dimension, Space, Mode,
                                    sycl::access::target::device>(
                  "Incorrect type", type_name)));
  }
  if (!same_value) {
    FAIL(log, (get_case_description<dimension, Space, Mode,
                                    sycl::access::target::device>(
                  "Incorrect value", type_name)));
  }
}

/** @brief Run tests for all combinations of access mode
 */
template <typename T, sycl::access::address_space Space, int dimension>
void run_tests_for_dimension(sycl_cts::util::logger& log,
                             const std::string& type_name) {
  run_tests<T, Space, dimension, sycl::access::mode::read>(log, type_name);
  run_tests<T, Space, dimension, sycl::access::mode::write>(log, type_name);
  run_tests<T, Space, dimension, sycl::access::mode::read_write>(log,
                                                                 type_name);
}

/** @brief Run tests for all combinations of dimensions
 */
template <typename T, sycl::access::address_space Space>
void run_tests_for_space(sycl_cts::util::logger& log,
                         const std::string& type_name) {
  run_tests_for_dimension<T, Space, 1>(log, type_name);
  run_tests_for_dimension<T, Space, 2>(log, type_name);
  run_tests_for_dimension<T, Space, 3>(log, type_name);
}

/** @brief Provides verification of multi_ptr global_space and generic_space
 * accessor constructor for specific type given
 *  @tparam T Specific type of multi ptr data to use
 */
template <typename T>
class check_multi_ptr_accessor_constructor_for_type {
 public:
  void operator()(sycl_cts::util::logger& log, const std::string& type_name) {
    using namespace sycl::access;
    // Accroding to spec of multi_ptr, constructor can take only
    // global_space and generic_space accessors
    run_tests_for_space<T, address_space::global_space>(log, type_name);
    run_tests_for_space<T, address_space::generic_space>(log, type_name);
  }
};

}  // namespace multi_ptr_accessor_constructors

#endif  // __SYCLCTS_TESTS_MULTI_PTR_ACCESSOR_CONSTRUCTORS_H
