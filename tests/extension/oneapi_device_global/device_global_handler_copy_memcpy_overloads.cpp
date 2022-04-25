/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for the member functions overloads of the sycl::handler class
//  discribed below:
//  1. copy overloads
//  2. memcpy overloads
//
//  The tests submits task in the sycl::queue and invokes .copy() or .memcpy()
//  member functions of the sycl::handler class then checks correctness of the
//  results. If testing type is array then test invokes .copy() or .memcpy()
//  member functions again but with `count` and `startIndex` parameters to copy
//  the first element.
//
*******************************************************************************/

#include "../../common/common.h"
#include "../../common/get_cts_object.h"
#include "device_global_common.h"
#include "type_pack.h"

#define TEST_NAME device_global_handler_functions_overloads

namespace TEST_NAMESPACE {
using namespace sycl_cts;
using namespace sycl_cts::util;
using namespace device_global_common_functions;

#if defined(SYCL_EXT_ONEAPI_PROPERTY_LIST) && \
    defined(SYCL_EXT_ONEAPI_DEVICE_GLOBAL)
namespace oneapi = sycl_stub::ext::oneapi;

namespace copy_to_dg {
template <typename T>
struct kernel1;
template <typename T>
struct kernel2;
// Creating instance with default constructor
template <typename T>
oneapi::device_global<T> dev_global;

/** @brief The function tests that .copy() member function overload correctly
 * copy data from the pointer to the device_global instance
 *  @tparam T Type of underlying device_global value
 */
template <typename T>
void run_test(util::logger& log, const std::string& type_name) {
  using element_type = std::remove_all_extents_t<T>;
  using element_ptr = element_type*;

  // Count size of one element
  constexpr size_t element_size = sizeof(element_type);
  // Count number of elements
  constexpr size_t elements_count = sizeof(T) / element_size;

  element_type init_value;
  init_value = 1;
  element_type changed_value;
  changed_value = 2;

  T src_value;
  value_operations<T>::assign(src_value, init_value);

  element_ptr src = pointer_helper(src_value);

  bool is_copy_correct{false};
  {
    sycl::buffer<bool, 1> is_copy_correct_buf(&is_copy_correct,
                                              sycl::range<1>(1));
    auto queue = util::get_cts_object::queue();
    queue.submit([&](sycl::handler& cgh) {
      // Copy elements from the pointed memory to the device_global instance
      cgh.copy<T>(src, dev_global<T>);
    });
    queue.wait_and_throw();

    queue.submit([&](sycl::handler& cgh) {
      auto is_copy_correct_acc =
          is_copy_correct_buf.template get_access<sycl::access_mode::write>(
              cgh);
      cgh.single_task<kernel1<T>>([=] {
        // dev_global have to be equal to *src after copy
        is_copy_correct_acc[0] =
            value_operations::are_equal<T>(dev_global<T>, *src);
      });
    });
    queue.wait_and_throw();

    // If T is array we can test .copy() member function with count and
    // startIndex parameters
    if constexpr (elements_count > 1) {
      queue.submit([&](sycl::handler& cgh) {
        // Changing value of the first element
        src[0] = changed_value;
        // Copy first element from the array
        cgh.copy<T>(src, dev_global<T>, element_size, 0);
      });
      queue.wait_and_throw();

      queue.submit([&](sycl::handler& cgh) {
        auto is_copy_correct_acc =
            is_copy_correct_buf.template get_access<sycl::access_mode::write>(
                cgh);
        cgh.single_task<kernel2<T>>([=] {
          // dev_global have to be equal to *src after copy
          is_copy_correct_acc[0] &=
              value_operations::are_equal<T>(dev_global<T>, *src);
        });
      });
      queue.wait_and_throw();
    }
  }

  if (!is_copy_correct) {
    FAIL(
        log,
        get_case_description(
            "device_global: sycl::handler .copy() member function overload",
            "Wrong value after copy to the device_global instance", type_name));
  }
}
}  // namespace copy_to_dg

namespace copy_from_dg {
template <typename T>
struct kernel1;
template <typename T>
struct kernel2;
template <typename T>
struct kernel3;
// Creating instance with default constructor
template <typename T>
oneapi::device_global<T> dev_global;

/** @brief The function tests that .copy() member function overload correctly
 * copy data to the pointer from the device_global instance
 *  @tparam T Type of underlying device_global value
 */
template <typename T>
void run_test(util::logger& log, const std::string& type_name) {
  using element_type = std::remove_all_extents_t<T>;
  using element_ptr = element_type*;

  // Count size of one element
  constexpr size_t element_size = sizeof(element_type);
  // Count number of elements
  constexpr size_t elements_count = sizeof(T) / element_size;

  element_type changed_value;
  changed_value = 2;

  T dest_value;
  value_operations<T>::assign(dest_value, changed_value);

  element_ptr dest = pointer_helper(dest_value);

  bool is_copy_correct{false};
  auto queue = util::get_cts_object::queue();
  {
    sycl::buffer<bool, 1> is_copy_correct_buf(&is_copy_correct,
                                              sycl::range<1>(1));
    auto queue = util::get_cts_object::queue();
    queue.submit([&](sycl::handler& cgh) {
      // Copy elements from the device_global instance to the pointed memory
      cgh.copy<T>(dev_global<T>, dest);
    });
    queue.wait_and_throw();

    queue.submit([&](sycl::handler& cgh) {
      auto is_copy_correct_acc =
          is_copy_correct_buf.template get_access<sycl::access_mode::write>(
              cgh);
      cgh.single_task<kernel1<T>>([=] {
        // dev_global have to be equal to *dest after copy
        is_copy_correct_acc[0] =
            value_operations::are_equal<T>(dev_global<T>, *dest);
      });
    });
    queue.wait_and_throw();

    // If T is array we can test .copy() member function with count and
    // startIndex parameters
    if constexpr (elements_count > 1) {
      queue.submit([&](sycl::handler& cgh) {
        // Changing value of the first element
        cgh.single_task<kernel2<T>>([=] { dev_global<T>[0] = changed_value; });
      });
      queue.wait_and_throw();

      queue.submit([&](sycl::handler& cgh) {
        // Copy first element from the array
        cgh.copy<T>(dev_global<T>, dest, element_size, 0);
      });
      queue.wait_and_throw();

      queue.submit([&](sycl::handler& cgh) {
        auto is_copy_correct_acc =
            is_copy_correct_buf.template get_access<sycl::access_mode::write>(
                cgh);
        cgh.single_task<kernel3<T>>([=] {
          // Compare again after copy
          is_copy_correct_acc[0] &=
              value_operations::are_equal<T>(dev_global<T>, *dest);
        });
      });
      queue.wait_and_throw();
    }
  }

  if (!is_copy_correct) {
    FAIL(log,
         get_case_description(
             "device_global: sycl::handler .copy() member function overload",
             "Wrong value after copy from the device_global instance",
             type_name));
  }
}
}  // namespace copy_from_dg

namespace memcpy_to_dg {
template <typename T>
struct kernel1;
template <typename T>
struct kernel2;
// Creating instance with default constructor
template <typename T>
oneapi::device_global<T> dev_global;

/** @brief The function tests that .memcpy() member function overload correctly
 * copy memory from the pointer to the device_global instance
 *  @tparam T Type of underlying device_global value
 */
template <typename T>
void run_test(util::logger& log, const std::string& type_name) {
  using element_type = std::remove_all_extents_t<T>;
  using element_ptr = element_type*;
  using void_ptr = void*;

  // Count size of one element
  constexpr size_t element_size = sizeof(element_type);
  // Count number of elements
  constexpr size_t elements_count = sizeof(T) / element_size;

  element_type init_value;
  init_value = 1;
  element_type changed_value;
  changed_value = 2;

  T src_value;
  value_operations<T>::assign(src_value, init_value);
  void_ptr src = static_cast<void_ptr>(pointer_helper(src_value));

  bool is_copy_correct{false};
  auto queue = util::get_cts_object::queue();
  {
    sycl::buffer<bool, 1> is_copy_correct_buf(&is_copy_correct,
                                              sycl::range<1>(1));
    auto queue = util::get_cts_object::queue();
    queue.submit([&](sycl::handler& cgh) {
      // Copy memory from the pointed memory to the device_global instance
      cgh.memcpy<T>(dev_global<T>, src);
    });
    queue.wait_and_throw();

    queue.submit([&](sycl::handler& cgh) {
      auto is_copy_correct_acc =
          is_copy_correct_buf.template get_access<sycl::access_mode::write>(
              cgh);
      cgh.single_task<kernel1<T>>([=] {
        // dev_global have to be equal to *src after copy
        is_copy_correct_acc[0] = value_operations::are_equal<T>(
            dev_global<T>, *(static_cast<element_ptr>(src)));
      });
    });
    queue.wait_and_throw();

    // If T is array we can test .copy() member function with count and
    // startIndex parameters
    if constexpr (elements_count > 1) {
      queue.submit([&](sycl::handler& cgh) {
        // Changing value of the first element
        element_ptr src_ptr = static_cast<element_ptr>(src);
        src_ptr[0] = changed_value;
        // Copy first element from the array
        cgh.memcpy<T>(dev_global<T>, src, element_size, 0);
      });
      queue.wait_and_throw();

      queue.submit([&](sycl::handler& cgh) {
        auto is_copy_correct_acc =
            is_copy_correct_buf.template get_access<sycl::access_mode::write>(
                cgh);
        cgh.single_task<kernel2<T>>([=] {
          // Compare again after copy
          is_copy_correct_acc[0] &= value_operations::are_equal<T>(
              dev_global<T>, *(static_cast<element_ptr>(src)));
        });
      });
      queue.wait_and_throw();
    }
  }

  if (!is_copy_correct) {
    FAIL(log,
         get_case_description(
             "device_global: sycl::handler .memcpy() member function overload",
             "Wrong value after memcpy to the device_global instance",
             type_name));
  }
}
}  // namespace memcpy_to_dg

namespace memcpy_from_dg {
template <typename T>
struct kernel1;
template <typename T>
struct kernel2;
template <typename T>
struct kernel3;
// Creating instance with default constructor
template <typename T>
oneapi::device_global<T> dev_global;

/** @brief The function tests that .memcpy() member function overload correctly
 * copy memory to the pointer from the device_global instance
 */
template <typename T>
void run_test(util::logger& log, const std::string& type_name) {
  using element_type = std::remove_all_extents_t<T>;
  using element_ptr = element_type*;
  using void_ptr = void*;

  // Count size of one element
  constexpr size_t element_size = sizeof(element_type);
  // Count number of elements
  constexpr size_t elements_count = sizeof(T) / element_size;

  element_type changed_value;
  changed_value = 2;

  T dest_value;
  value_operations<T>::assign(dest_value, changed_value);

  void_ptr dest = pointer_helper(dest_value);

  bool is_copy_correct{false};
  auto queue = util::get_cts_object::queue();
  {
    sycl::buffer<bool, 1> is_copy_correct_buf(&is_copy_correct,
                                              sycl::range<1>(1));
    auto queue = util::get_cts_object::queue();
    queue.submit([&](sycl::handler& cgh) {
      // Copy memory to the pointed memory from the device_global instance
      cgh.memcpy<T>(dest, dev_global<T>);
    });
    queue.wait_and_throw();

    queue.submit([&](sycl::handler& cgh) {
      auto is_copy_correct_acc =
          is_copy_correct_buf.template get_access<sycl::access_mode::write>(
              cgh);
      cgh.single_task<kernel1<T>>([=] {
        // dev_global have to be equal to *dest after copy
        is_copy_correct_acc[0] = value_operations::are_equal<T>(
            dev_global<T>, *(static_cast<element_ptr>(dest)));
      });
    });
    queue.wait_and_throw();

    // If T is array we can test .copy() member function with count and
    // startIndex parameters
    if constexpr (elements_count > 1) {
      queue.submit([&](sycl::handler& cgh) {
        // Changing value of the first element
        cgh.singe_task<kernel2<T>>([=] { dev_global<T>[0] = changed_value; });
      });
      queue.wait_and_throw();

      queue.submit([&](sycl::handler& cgh) {
        // Copy first element from the array
        cgh.memcpy<T>(dest, dev_global<T>, element_size, 0);
      });
      queue.wait_and_throw();

      queue.submit([&](sycl::handler& cgh) {
        auto is_copy_correct_acc =
            is_copy_correct_buf.template get_access<sycl::access_mode::write>(
                cgh);
        cgh.single_task<kernel3<T>>([=] {
          // Compare again after copy
          is_copy_correct_acc[0] = value_operations::are_equal<T>(
              dev_global<T>, *(static_cast<element_type*>(dest)));
        });
      });
      queue.wait_and_throw();
    }
    }

    if (!is_copy_correct) {
      FAIL(
          log,
          get_case_description(
              "device_global: sycl::handler .memcpy() member function overload",
              "Wrong value after memcpy from the device_global instance",
              type_name));
    }
  }
}  // namespace memcpy_from_dg

template <typename T>
class check_device_global_handler_functions_overloads {
 public:
  void operator()(sycl_cts::util::logger& log, const std::string& type_name) {
    copy_from_dg::run_test<T>(log, type_name);
    copy_to_dg::run_test<T>(log, type_name);
    memcpy_from_dg::run_test<T>(log, type_name);
    memcpy_to_dg::run_test<T>(log, type_name);

    copy_from_dg::run_test<T[5]>(log, type_name);
    copy_to_dg::run_test<T[5]>(log, type_name);
    memcpy_from_dg::run_test<T[5]>(log, type_name);
    memcpy_to_dg::run_test<T[5]>(log, type_name);
  }
};
#endif

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
#if !defined(SYCL_EXT_ONEAPI_PROPERTY_LIST)
    WARN("SYCL_EXT_ONEAPI_PROPERTY_LIST is not defined, test is skipped");
#elif !defined(SYCL_EXT_ONEAPI_DEVICE_GLOBAL)
    WARN("SYCL_EXT_ONEAPI_DEVICE_GLOBAL is not defined, test is skipped");
#else
    auto types = device_global_types::get_types();
    for_all_types<check_device_global_handler_functions_overloads>(types, log);
#endif
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;
}  // namespace TEST_NAMESPACE
