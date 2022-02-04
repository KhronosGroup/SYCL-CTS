/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for the exceptions that are thrown by sycl::queue and
//  sycl::handler copy member functions overloads
//
*******************************************************************************/

#include "../../common/common.h"
#include "../../common/get_cts_object.h"
#include "device_global_common.h"
#include "type_pack.h"

#define TEST_NAME device_global_exceptions

namespace TEST_NAMESPACE {
using namespace sycl_cts;
using namespace sycl_cts::util;
using namespace device_global_common_functions;

#if defined(SYCL_EXT_ONEAPI_PROPERTY_LIST) && \
    defined(SYCL_EXT_ONEAPI_DEVICE_GLOBAL)
namespace oneapi = sycl::ext::oneapi;

/**
 * @brief The functions checks if error code of the given exception is equal to
 * sycl::errc::invalid
 * @param e The sycl::exception instance that needs to verify error code
 * @return True if error code of exception is sycl::errc::invalid, false
 * otherwise
 */
inline bool is_errc_invalid(const sycl::exception& e) {
  return e.code() != sycl::errc::invalid;
}

namespace handler_copy_exceptions {
// Creating instance with default constructor
template <typename T>
oneapi::device_global<T> dev_global;

/** @brief The function tests that .copy() member function overload throws an exception
 * with error code equal to errc::invalid if attempt to write beyond the end of
 * the destination variable
 *  @tparam T Type of underlying device_global value
 */
template <typename T, size_t N>
void run_test(util::logger& log, const std::string& type_name) {
  std::shared_ptr<T[N]> src(new T[N]);
  std::shared_ptr<T[N]> dest(new T[N]);

  bool is_exception_thrown{true};
  bool is_exception_correct{true};
  auto queue = util::get_cts_object::queue();

  queue.submit([&](sycl::handler& cgh) {
    // Try to write data beyond the end of the destination variable
    // Have to throw an exception
    try {
      cgh.template copy<T>(src.get(), dev_global<T>, N, N / 2);
      is_exception_thrown &= false;
    } catch (sycl::exception const& e) {
      is_exception_correct &= is_errc_invalid(e);
    }

    try {
      cgh.template copy<T>(dev_global<T>, dest.get(), N, N / 2);
      is_exception_thrown &= false;
    } catch (sycl::exception const& e) {
      is_exception_correct &= is_errc_invalid(e);
    }
  });
  queue.wait_and_throw();

  if (!is_exception_thrown) {
    FAIL(log, get_case_description(
                  "deivice_global: sycl::handler .copy() member function exception",
                  "Exception was not thrown after attempt to "
                  "write beyond the end of the destination variable",
                  type_name));
  } else if (!is_exception_correct) {
    FAIL(log,
         get_case_description(
             "deivice_global: sycl::handler .copy() member function exception",
             "Wrong errc inside the exception. Expected sycl::errc::invalid",
             type_name));
  }
}
}  // namespace handler_copy_exceptions

namespace handler_memcpy_exceptions {
// Creating instance with default constructor
template <typename T>
oneapi::device_global<T> dev_global;

/** @brief The function tests that .memcpy() member function overload throws an exception
 * with error code equal to errc::invalid if attempt to write beyond the end of
 * the destination variable
 *  @tparam T Type of underlying device_global value
 */
template <typename T, size_t N>
void run_test(util::logger& log, const std::string& type_name) {
  std::shared_ptr<void> src(new T[N]);
  std::shared_ptr<void> dest(new T[N]);

  bool is_exception_thrown{true};
  bool is_exception_correct{true};
  auto queue = util::get_cts_object::queue();

  queue.submit([&](sycl::handler& cgh) {
    // Try to write data beyond the end of the destination variable
    // Have to throw an exception
    try {
      cgh.template memcpy<T>(src.get(), dev_global<T>, sizeof(T) * N,
                             (sizeof(T) * N) / 2);
      is_exception_thrown &= false;
    } catch (sycl::exception const& e) {
      is_exception_correct &= is_errc_invalid(e);
    }

    try {
      cgh.template memcpy<T>(dev_global<T>, dest.get(), sizeof(T) * N,
                             (sizeof(T) * N) / 2);
      is_exception_thrown &= false;
    } catch (sycl::exception const& e) {
      is_exception_correct &= is_errc_invalid(e);
    }
  });
  queue.wait_and_throw();

  if (!is_exception_thrown) {
    FAIL(log, get_case_description(
                  "deivice_global: sycl::handler .memcpy() member function exception",
                  "Exception was not thrown after attempt to "
                  "write beyond the end of the destination variable",
                  type_name));
  } else if (!is_exception_correct) {
    FAIL(log,
         get_case_description(
             "deivice_global: sycl::handler .memcpy() member function exception",
             "Wrong errc inside the exception. Expected sycl::errc::invalid",
             type_name));
  }
}
}  // namespace handler_memcpy_exceptions

namespace queue_copy_exceptions {
// Creating instance with default constructor
template <typename T>
oneapi::device_global<T> dev_global;

/** @brief The function tests that .copy() member function overload throws an exception
 * with error code equal to errc::invalid if attempt to write beyond the end of
 * the destination variable
 *  @tparam T Type of underlying device_global value
 */
template <typename T, size_t N>
void run_test(util::logger& log, const std::string& type_name) {
  std::shared_ptr<T[N]> src(new T[N]);
  std::shared_ptr<T[N]> dest(new T[N]);

  bool is_exception_thrown{true};
  bool is_exception_correct{true};
  auto queue = util::get_cts_object::queue();
  // Try to write data beyond the end of the destination variable
  // Have to throw an exception
  try {
    queue.template copy<T>(src.get(), dev_global<T>, N, N / 2);
    queue.wait_and_throw();
    is_exception_thrown &= false;
  } catch (sycl::exception const& e) {
    is_exception_correct &= is_errc_invalid(e);
  }

  try {
    queue.template copy<T>(dev_global<T>, dest.get(), N, N / 2);
    queue.wait_and_throw();
    is_exception_thrown &= false;
  } catch (sycl::exception const& e) {
    is_exception_correct &= is_errc_invalid(e);
  }

  if (!is_exception_thrown) {
    FAIL(log, get_case_description(
                  "deivice_global: sycl::queue .copy() member function exception",
                  "Exception was not thrown after attempt to "
                  "write beyond the end of the destination variable",
                  type_name));
  } else if (!is_exception_correct) {
    FAIL(log,
         get_case_description(
             "deivice_global: sycl::queue .copy() member function exception",
             "Wrong errc inside the exception. Expected sycl::errc::invalid",
             type_name));
  }
}
}  // namespace queue_copy_exceptions

namespace queue_memcpy_exceptions {
// Creating instance with default constructor
template <typename T>
oneapi::device_global<T> dev_global;

/** @brief The function tests that .memcpy() member function overload throws an exception
 * with error code equal to errc::invalid if attempt to write beyond the end of
 * the destination variable
 *  @tparam T Type of underlying device_global value
 */
template <typename T, size_t N>
void run_test(util::logger& log, const std::string& type_name) {
  std::shared_ptr<void> src(new T[N]);
  std::shared_ptr<void> dest(new T[N]);

  bool is_exception_thrown{true};
  bool is_exception_correct{true};
  auto queue = util::get_cts_object::queue();

  try {
    queue.template memcpy<T>(src.get(), dev_global<T>, sizeof(T) * N,
                             (sizeof(T) * N) / 2);
    queue.wait_and_throw();
    is_exception_thrown &= false;
  } catch (sycl::exception const& e) {
    is_exception_correct &= is_errc_invalid(e);
  }

  try {
    queue.template memcpy<T>(dev_global<T>, dest.get(), sizeof(T) * N,
                             (sizeof(T) * N) / 2);
    queue.wait_and_throw();
    is_exception_thrown &= false;
  } catch (sycl::exception const& e) {
    is_exception_correct &= is_errc_invalid(e);
  }

  if (!is_exception_thrown) {
    FAIL(log, get_case_description(
                  "deivice_global: sycl::queue .memcpy() member function exception",
                  "Exception was not thrown after attempt to "
                  "write beyond the end of the destination variable",
                  type_name));
  } else if (!is_exception_correct) {
    FAIL(log,
         get_case_description(
             "deivice_global: sycl::queue .memcpy() member function exception",
             "Wrong errc inside the exception. Expected sycl::errc::invalid",
             type_name));
  }
}
}  // namespace queue_memcpy_exceptions

template <typename T>
class check_device_global_handler_queue_exceptions {
 public:
  void operator()(sycl_cts::util::logger& log, const std::string& type_name) {
    handler_copy_exceptions::run_test<T, 5>(log, type_name);
    handler_memcpy_exceptions::run_test<T, 5>(log, type_name);

    queue_copy_exceptions::run_test<T, 5>(log, type_name);
    queue_memcpy_exceptions::run_test<T, 5>(log, type_name);
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
    for_all_types<check_device_global_handler_queue_exceptions>(types, log);
#endif
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;
}  // namespace TEST_NAMESPACE
