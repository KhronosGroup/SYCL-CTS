/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for the device_global api
//
//  Some of the tests execute in the kernel. These types of tests have result
//  arrays, that kernel uses through a buffer accessor. For result arrays
//  defined enum classes that have to be used for addressing elements in the
//  array. Arrays with strings are used to print errors corresponding to the
//  index in result array.
//
*******************************************************************************/

#include "../../../util/array.h"
#include "../../common/common.h"
#include "../../common/get_cts_object.h"
#include "device_global_common.h"
#include "type_pack.h"

#define TEST_NAME device_global_api_basic

namespace TEST_NAMESPACE {
using namespace sycl_cts;
using namespace sycl_cts::util;
using namespace device_global_common_functions;

#if defined(SYCL_EXT_ONEAPI_PROPERTIES) && \
    defined(SYCL_EXT_ONEAPI_DEVICE_GLOBAL)

using namespace sycl::ext::oneapi::experimental;

template <typename T, test_names name,
          sycl::access::decorated Decorated = sycl::access::decorated::no>
struct device_global_kernel_name;

namespace get_multi_ptr_method {
// Creating device_global instances with default constructor
template <typename T, sycl::access::decorated Decorated>
const device_global<T> const_dev_global;
template <typename T, sycl::access::decorated Decorated>
device_global<T> dev_global;

// Used to address elements in the result array
enum class indx : size_t {
  same_type_const,
  same_type_non_const,
  correct_def_val_const,
  correct_def_val_non_const,
  correct_changed_val,
  size  // must be last
};
constexpr size_t integral(const indx& i) { return to_integral<indx>(i); }

/** @brief The function tests that device_global method get_multi_ptr() returns
 * multi_ptr that points to underlying device_global value
 *  @tparam T Type of underlying device_global value
 *  @tparam Decorated The flag indicates the decorated version of multi_ptr is
 * created
 */
template <typename T, sycl::access::decorated Decorated>
void run_test(util::logger& log, const std::string& type_name) {
  using kernel =
      device_global_kernel_name<T, test_names::get_multi_ptr_method, Decorated>;

  using multi_ptr_t =
      sycl::multi_ptr<T, sycl::access::address_space::global_space, Decorated>;
  using const_multi_ptr_t =
      sycl::multi_ptr<const T, sycl::access::address_space::global_space,
                      Decorated>;

  std::string error_strings[integral(indx::size)]{
      "Wrong type inside const multi_ptr",
      "Wrong type inside non-const multi_ptr",
      "Wrong default value inside const multi_ptr",
      "Wrong default value inside non-const multi_ptr",
      "Wrong value after change inside multi_ptr",
  };

  util::array<bool, integral(indx::size)> result;
  {
    sycl::buffer<bool, 1> result_buf(result.values,
                                     sycl::range<1>(integral(indx::size)));
    auto queue = util::get_cts_object::queue();
    queue.submit([&](sycl::handler& cgh) {
      auto result_acc =
          result_buf.template get_access<sycl::access_mode::read_write>(cgh);

      T value_ref_zero_init{};
      std::memset(&value_ref_zero_init, 0, sizeof(value_ref_zero_init));
      cgh.single_task<kernel>([=] {
        auto cmptr =
            const_dev_global<const T, Decorated>.template get_multi_ptr<Decorated>();
        auto mptr =
            dev_global<T, Decorated>.template get_multi_ptr<Decorated>();

        // Check that underlying type of multi_ptr is same as T
        result_acc[integral(indx::same_type_const)] =
            std::is_same<decltype(cmptr), const_multi_ptr_t>::value;
        result_acc[integral(indx::same_type_non_const)] =
            std::is_same<decltype(mptr), multi_ptr_t>::value;

        // if *mptr and *cmptr equal to default value, then
        // test will be marked as passed, otherwise the test is failed
        result_acc[integral(indx::correct_def_val_const)] =
            value_operations::are_equal(*(cmptr.get()), value_ref_zero_init);
        result_acc[integral(indx::correct_def_val_non_const)] =
            value_operations::are_equal(*(mptr.get()), value_ref_zero_init);
        // Change value, that multi_ptr points to
        value_operations::assign(*mptr, 42);
        // Get current value from device_global, that should change in previous
        // step
        const T& current_value = dev_global<T, Decorated>.get();
        result_acc[integral(indx::correct_changed_val)] =
            value_operations::are_equal(*mptr, current_value);
      });
    });
  }
  for (size_t i = integral(indx::same_type_const); i < integral(indx::size);
       ++i) {
    if (!result[i]) {
      std::string fail_msg = get_case_description<Decorated>(
          "Device global: get_multi_ptr()", error_strings[i], type_name);
      FAIL(log, fail_msg);
    }
  }
}
}  // namespace get_multi_ptr_method

namespace implicit_conversation_to_T {
// Creating device_global instances with default constructor
template <typename T>
const device_global<T> const_dev_global;
template <typename T>
device_global<T> dev_global;

// Setting enum for result array
enum class indx : size_t {
  is_def_value_const,
  is_def_value_non_const,
  correct_changed_val,
  size  // must be last
};
constexpr size_t integral(const indx& i) { return to_integral(i); }

/** @brief The function tests that device_global correctly implicitly converts
 * to underlying value
 *  @tparam T Type of underlying device_global value
 */
template <typename T>
void run_test(util::logger& log, const std::string& type_name) {
  using kernel =
      device_global_kernel_name<T, test_names::implicit_conversation>;

  std::string error_strings[integral(indx::size)]{
      "Wrong default value const type",
      "Wrong default value non-const type",
      "Wrong value after change",
  };

  util::array<bool, integral(indx::size)> result;
  {
    sycl::buffer<bool, 1> result_buf(result.values,
                                     sycl::range<1>(integral(indx::size)));

    auto queue = util::get_cts_object::queue();
    queue.submit([&](sycl::handler& cgh) {
      auto result_acc =
          result_buf.template get_access<sycl::access_mode::read_write>(cgh);

      T value_ref_zero_init{};
      std::memset(&value_ref_zero_init, 0, sizeof(value_ref_zero_init));
      cgh.single_task<kernel>([=] {
        // Call copy constructor of T to access reference to the device_global
        // value
        const T& const_instance(const_dev_global<T>);
        T& instance(dev_global<T>);

        // Check that resulted reference is to default value
        result_acc[integral(indx::is_def_value_const)] =
            value_operations::are_equal(value_ref_zero_init, instance);
        result_acc[integral(indx::is_def_value_non_const)] =
            value_operations::are_equal(value_ref_zero_init, const_instance);

        // Changing value
        value_operations::assign(dev_global<T>, 42);
        // Get current value from the device_global, which should change in
        // previous step
        T& current_value = dev_global<T>.get();
        // Check, that the device_global object contains new value
        result_acc[integral(indx::correct_changed_val)] =
            value_operations::are_equal(dev_global<T>, current_value);
      });
    });
  }
  for (size_t i = integral(indx::is_def_value_const); i < integral(indx::size);
       ++i) {
    if (!result[i]) {
      std::string fail_msg =
          get_case_description("Device global: implicit conversation to T",
                               error_strings[i], type_name);
      FAIL(log, fail_msg);
    }
  }
}
}  // namespace implicit_conversation_to_T

namespace get_method {
// Creating device_global instances with default constructor
template <typename T>
const device_global<T> const_dev_global;
template <typename T>
device_global<T> dev_global;

// Setting enum for result array
enum class indx : size_t {
  same_type_const,
  same_type_non_const,
  correct_def_val_const,
  correct_def_val_non_const,
  correct_changed_val,
  size  // must be last
};
constexpr size_t integral(const indx& i) { return to_integral(i); }

/** @brief The function tests that device_global get() method returns correct
 * underlying value
 *  @tparam T Type of underlying device_global value
 */
template <typename T>
void run_test(util::logger& log, const std::string& type_name) {
  using kernel = device_global_kernel_name<T, test_names::get_method>;

  std::string error_strings[integral(indx::size)]{
      "Wrong type for const value",
      "Wrong type for non-const value",
      "Wrong default value for const value",
      "Wrong default value for non-const value",
      "Wrong value after change",
  };

  util::array<bool, integral(indx::size)> result;
  {
    sycl::buffer<bool, 1> result_buf(result.values,
                                     sycl::range<1>(integral(indx::size)));

    auto queue = util::get_cts_object::queue();
    queue.submit([&](sycl::handler& cgh) {
      auto result_acc =
          result_buf.template get_access<sycl::access_mode::read_write>(cgh);

      T value_ref_zero_init{};
      std::memset(&value_ref_zero_init, 0, sizeof(value_ref_zero_init));
      cgh.single_task<kernel>([=] {
        // Call get() to access reference
        const T& const_instance(const_dev_global<T>.get());
        T& instance(dev_global<T>.get());
        // Check that return type is T& and const T&
        result_acc[integral(indx::same_type_const)] =
            std::is_same<decltype(const_instance), const T&>::value;
        result_acc[integral(indx::same_type_non_const)] =
            std::is_same<decltype(instance), T&>::value;
        // Check that resulted reference is to default value
        result_acc[integral(indx::correct_def_val_const)] =
            value_operations::are_equal(const_instance, value_ref_zero_init);
        result_acc[integral(indx::correct_def_val_non_const)] =
            value_operations::are_equal(instance, value_ref_zero_init);
        // Assign new value an check that device_global instance contains new
        // value
        value_operations::assign(instance, 42);
        result_acc[integral(indx::correct_changed_val)] =
            value_operations::are_equal(dev_global<T>, instance);
      });
    });
  }
  for (size_t i = integral(indx::same_type_const); i < integral(indx::size);
       ++i) {
    if (!result[i]) {
      std::string fail_msg = get_case_description("Device global: get() method",
                                                  error_strings[i], type_name);
      FAIL(log, fail_msg);
    }
  }
}
}  // namespace get_method

namespace has_property_method {
using namespace sycl::ext::oneapi::experimental;
// Creating instance with property prop_value_t with default
// constructor
template <typename T, typename prop_value_t>
device_global<T, decltype(properties{prop_value_t{}})> dev_global;

/** @brief The function tests that the device_global has only props, that was
 * given to properties
 *  @tparam T Type of underlying device_global value
 *  @tparam prop_value_t Property contained by properties
 *  @tparam other_props Properties, which should not be in properties
 */
template <typename T, typename prop_value_t, typename other_prop>
void has_no_other_props(util::logger& log, const std::string& type_name) {
  if (dev_global<T, prop_value_t>.template has_property<other_prop>() !=
      false) {
    std::string fail_msg =
        get_case_description("Device global: has_property()",
                             "Unexpected property returned true", type_name);
    FAIL(log, fail_msg);
  }
}

// Helper for checking has_no_other_props for all properties in a type_pack.
template <typename T, typename prop_value_t, typename other_props>
struct check_has_no_other_props_helper;
template <typename T, typename prop_value_t>
struct check_has_no_other_props_helper<T, prop_value_t, type_pack<>> {
  static void check(util::logger&, const std::string&) {}
};
template <typename T, typename prop_value_t, typename other_prop,
          typename... other_props>
struct check_has_no_other_props_helper<T, prop_value_t,
                                       type_pack<other_prop, other_props...>> {
  static void check(util::logger& log, const std::string& type_name) {
    has_no_other_props<T, prop_value_t, other_prop>(log, type_name);
    check_has_no_other_props_helper<
        T, prop_value_t, type_pack<other_props...>>::check(log, type_name);
  }
};

/** @brief The function tests that device_global method has_property() return
 * true on properties, that was given to properties and false if property was
 * not provided
 *  @tparam T Type of underlying value
 *  @tparam prop_key Name of property that included in device_global property
 * list
 *  @tparam prop_value_t Property contained by properties
 *  @tparam other_props Props, which should not be in properties
 */
template <typename T, typename prop_key, typename prop_value_t,
          typename other_props>
void run_test(util::logger& log, const std::string& type_name) {
  // Check that instance has prop_key property
  if (dev_global<T, prop_value_t>.template has_property<prop_key>() != true) {
    std::string fail_msg = get_case_description("Device global: has_property()",
                                                "Wrong value.", type_name);
    FAIL(log, fail_msg);
  }
  // Check that instance has no other_props
  check_has_no_other_props_helper<T, prop_value_t, other_props>::check(
      log, type_name);
}
}  // namespace has_property_method

namespace get_property_method {
using namespace sycl::ext::oneapi::experimental;
// Creating instance with property prop_value_t with default
// constructor
template <typename T, typename prop_value_t>
device_global<T, decltype(properties{prop_value_t{}})> dev_global;

/** @brief The function tests that device_global get_property method returns
 * correct properties
 *  @tparam T Type of underlying device_global value
 *  @tparam prop_key Name of property that included in device_global property
 * list
 *  @tparam prop_value_t Integral_constant of property
 *  @param expected property that expecting from get_property()
 */
template <typename T, typename prop_key, typename prop_value_t>
void run_test(util::logger& log, const std::string& type_name) {
  bool property_check{};
  // Check that get_property<prop_value_t> returns expected property of
  // expected_prop_type
  property_check =
      (dev_global<T, prop_value_t>.template get_property<prop_key>() ==
       prop_value_t{});

  if (!property_check) {
    std::string fail_msg = get_case_description(
        "Device global: get_property", "Wrong property returned.", type_name);
    FAIL(log, fail_msg);
  }
}
}  // namespace get_property_method

namespace element_type {
// Creating instance with default constructor
template <typename T>
device_global<T> dev_global;

template <typename T>
void run_test(util::logger& log, const std::string& type_name) {
  // Check that member type element_type is_same as std::remove_extent_t<T>
  bool isSame = std::is_same<typename device_global<T>::element_type,
                             std::remove_extent_t<T>>::value;
  if (!isSame) {
    std::string fail_msg = get_case_description(
        "Device global: element_type",
        "Wrong type. element_type != std::remove_extent_t<T>", type_name);
    FAIL(log, fail_msg);
  }
}
}  // namespace element_type
template <typename T>
void run_tests(sycl_cts::util::logger& log, const std::string& type_name) {
  // Tests for different decorated parameters
  get_multi_ptr_method::run_test<T, sycl::access::decorated::no>(log,
                                                                 type_name);
  get_multi_ptr_method::run_test<T, sycl::access::decorated::yes>(log,
                                                                  type_name);

  implicit_conversation_to_T::run_test<T>(log, type_name);
  get_method::run_test<T>(log, type_name);

  // Run test of has_property method with different properties in device_global
  // property list
  {
    using prop_key = host_access_key;
    using prop_value = host_access_key::value_t<host_access_enum::read>;
    using other_props = type_pack<device_image_scope_key>;

    has_property_method::run_test<T, prop_key, prop_value, other_props>(
        log, type_name);
  }
  {
    using prop_key = device_image_scope_key;
    using prop_value = device_image_scope_key::value_t;
    using other_props = type_pack<host_access_key>;

    has_property_method::run_test<T, prop_key, prop_value, other_props>(
        log, type_name);
  }

  // Run test of get_property method with different properties in
  // device_global property list
  {
    using prop_key = host_access_key;
    using prop_value = host_access_key::value_t<host_access_enum::read>;
    get_property_method::run_test<T, prop_key, prop_value>(log, type_name);
  }
  {
    using prop_key = device_image_scope_key;
    using prop_value = device_image_scope_key::value_t;
    get_property_method::run_test<T, prop_key, prop_value>(log, type_name);
  }

  element_type::run_test<T>(log, type_name);
}

template <typename T>
class check_device_global_api_basic_for_type {
 public:
  void operator()(sycl_cts::util::logger& log, const std::string& type_name) {
    run_tests<T>(log, type_name);
    run_tests<T[5]>(log, type_name);
  }
};
#endif

/** test device_global api basic
 */
class TEST_NAME : public sycl_cts::util::test_base {
 public:
  /** return information about this test
   */
  void get_info(test_base::info& out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute the test
   */
  void run(util::logger& log) override {
#if !defined(SYCL_EXT_ONEAPI_PROPERTIES)
    SKIP("SYCL_EXT_ONEAPI_PROPERTIES is not defined");
#elif !defined(SYCL_EXT_ONEAPI_DEVICE_GLOBAL)
    SKIP("SYCL_EXT_ONEAPI_DEVICE_GLOBAL is not defined");
#else
    auto types = device_global_types::get_types();
    for_all_types<check_device_global_api_basic_for_type>(types, log);
#endif
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

}  // namespace TEST_NAMESPACE
