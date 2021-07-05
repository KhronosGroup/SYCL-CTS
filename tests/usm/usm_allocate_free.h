/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Common checks for usm allocation/free functions
//
*******************************************************************************/

#ifndef __SYCLCTS_TESTS_USM_ALLOC_FREE_H
#define __SYCLCTS_TESTS_USM_ALLOC_FREE_H

#include "../common/common.h"
#include "usm.h"
#include "usm_allocations_helper.h"

namespace usm_allocate_free {
using namespace sycl_cts;

struct templated : std::true_type {};
struct non_templated : std::false_type {};
struct by_context : std::true_type {};
struct by_queue : std::false_type {};

template <typename T, typename id, usm_alloc_help::memb_func_index index>
class usm_kernel_name;

enum class usm_op_name : int {
  malloc = 0,
  malloc_device,
  malloc_host,
  malloc_shared
};

enum class usm_op_form : int { general = 0, aligned };

/** @brief Defines an usm allocation function to test
 *  @tparam usm_op_name Name of usm allocation function
 *  @tparam usm_op_form Form of usm allocation function (general/aligned)
 */
template <usm_op_name opName, usm_op_form opForm>
struct usm_operation {
  static constexpr usm_op_name name = opName;
  static constexpr usm_op_form form = opForm;

  static constexpr sycl::usm::alloc extract_kind() {
    if constexpr (name == usm_op_name::malloc_device) {
      return sycl::usm::alloc::device;
    } else if constexpr (name == usm_op_name::malloc_host) {
      return sycl::usm::alloc::host;
    } else if constexpr (name == usm_op_name::malloc_shared) {
      return sycl::usm::alloc::shared;
    } else {
      static_assert(name != name, "Unknown name");
    }
  }
};

/** @brief Defines a case and used to make unique kernel name
 *  @tparam op Defines name and form of usm allocation function
 *  @tparam caseNum Index of test case
 *  @tparam alKind usm::alloc kind of usm allocation
 *  @tparam templatedT Points to use a templated variant of usm function or not
 */
template <typename op, int caseNum, sycl::usm::alloc alKind,
          typename templatedT>
struct case_id {
  using operation = op;
  static constexpr int case_num = caseNum;
  static constexpr sycl::usm::alloc kind = alKind;

  static constexpr bool templated() { return templatedT::value; }

  static std::string get_description() {
    std::string result{"case " + std::to_string(caseNum)};
    if constexpr (op::name == usm_op_name::malloc) {
      result += " with kind:";
      if constexpr (kind == sycl::usm::alloc::device) {
        result += " usm::alloc::device";
      }
      if constexpr (kind == sycl::usm::alloc::host) {
        result += " usm::alloc::host";
      }
      if constexpr (kind == sycl::usm::alloc::shared) {
        result += " usm::alloc::shared";
      }
    }
    if constexpr (templatedT::value) {
      result += " (templated form)";
    }
    return result;
  }
};

/** @brief Defines calls to all overloads of usm allocation functions and
 *         different checks based on usm::alloc kind
 *  @tparam T Testing type
 *  @tparam count Count of elements to allocate
 */
template <typename T, std::size_t count>
class usm_allocations_test_helper {
  sycl::queue &q;
  std::size_t align;

 public:
  usm_allocations_test_helper(sycl::queue &queue, std::size_t al)
      : q(queue), align(al) {}

  /** @brief Overload to call general form of usm allocation functions
   *  @tparam c_id Id of test
   *  @param cnt Count of elements to allocate
   *  @param args Manualy passed parameters depending on the tested overload
   *  @retval Pointer to memory allocated with required usm allocation function
   */
  template <typename c_id, typename... argsT>
  T *allocate(std::size_t cnt, argsT &&... args) {
    using op = typename c_id::operation;
    if constexpr (c_id::templated()) {
      // Templated
      if constexpr (op::name == usm_op_name::malloc) {
        return sycl::malloc<T>(cnt, std::forward<argsT>(args)...);
      } else if constexpr (op::name == usm_op_name::malloc_device) {
        return sycl::malloc_device<T>(cnt, std::forward<argsT>(args)...);
      } else if constexpr (op::name == usm_op_name::malloc_host) {
        return sycl::malloc_host<T>(cnt, std::forward<argsT>(args)...);
      } else if constexpr (op::name == usm_op_name::malloc_shared) {
        return sycl::malloc_shared<T>(cnt, std::forward<argsT>(args)...);
      } else {
        static_assert(op::name != op::name, "Unknown op::name passed");
      }
    } else {
      // Non-templated
      std::size_t n_bytes = sizeof(T) * cnt;
      if constexpr (op::name == usm_op_name::malloc) {
        return (T *)sycl::malloc(n_bytes, std::forward<argsT>(args)...);
      } else if constexpr (op::name == usm_op_name::malloc_device) {
        return (T *)sycl::malloc_device(n_bytes, std::forward<argsT>(args)...);
      } else if constexpr (op::name == usm_op_name::malloc_host) {
        return (T *)sycl::malloc_host(n_bytes, std::forward<argsT>(args)...);
      } else if constexpr (op::name == usm_op_name::malloc_shared) {
        return (T *)sycl::malloc_shared(n_bytes, std::forward<argsT>(args)...);
      } else {
        static_assert(op::name != op::name, "Unknown op::name passed");
      }
    }
  }

  /** @brief Overload to call aligned form of usm allocation functions
   *  @tparam c_id Id of test
   *  @param al Requred alignment
   *  @param cnt Count of elements to allocate
   *  @param args Manualy passed parameters depending on the tested overload
   *  @retval Pointer to memory allocated with required usm allocation function
   */
  template <typename c_id, typename... argsT>
  T *allocate(std::size_t al, std::size_t cnt, argsT &&... args) {
    using op = typename c_id::operation;
    if constexpr (c_id::templated()) {
      // Templated
      if constexpr (op::name == usm_op_name::malloc) {
        return sycl::aligned_alloc<T>(al, cnt, std::forward<argsT>(args)...);
      } else if constexpr (op::name == usm_op_name::malloc_device) {
        return sycl::aligned_alloc_device<T>(al, cnt,
                                             std::forward<argsT>(args)...);
      } else if constexpr (op::name == usm_op_name::malloc_host) {
        return sycl::aligned_alloc_host<T>(al, cnt,
                                           std::forward<argsT>(args)...);
      } else if constexpr (op::name == usm_op_name::malloc_shared) {
        return sycl::aligned_alloc_shared<T>(al, cnt,
                                             std::forward<argsT>(args)...);
      } else {
        static_assert(op::name != op::name, "Unknown op::name passed");
      }
    } else {
      // Non-templated
      std::size_t n_bytes = sizeof(T) * cnt;
      if constexpr (op::name == usm_op_name::malloc) {
        return (T *)sycl::aligned_alloc(al, n_bytes,
                                        std::forward<argsT>(args)...);
      } else if constexpr (op::name == usm_op_name::malloc_device) {
        return (T *)sycl::aligned_alloc_device(al, n_bytes,
                                               std::forward<argsT>(args)...);
      } else if constexpr (op::name == usm_op_name::malloc_host) {
        return (T *)sycl::aligned_alloc_host(al, n_bytes,
                                             std::forward<argsT>(args)...);
      } else if constexpr (op::name == usm_op_name::malloc_shared) {
        return (T *)sycl::aligned_alloc_shared(al, n_bytes,
                                               std::forward<argsT>(args)...);
      } else {
        static_assert(op::name != op::name, "Unknown op::name passed");
      }
    }
  }

  /** @brief Perform required allocations and checks
   */
  template <typename c_id>
  void fill_and_check(util::logger &log, T *ptr) {
    using namespace usm_alloc_help;
    // Create unique kernel names
    using fill_kernel = usm_kernel_name<T, c_id, memb_func_index::fill>;
    using simple_kernel = usm_kernel_name<T, c_id, memb_func_index::simple>;
    using device_kernel = usm_kernel_name<T, c_id, memb_func_index::device>;
    using host_kernel = usm_kernel_name<T, c_id, memb_func_index::host>;

    fill<T, count, fill_kernel>(ptr, q);

    if (!check_simple<T, count, simple_kernel>(log, ptr, q))
      FAIL(log, "Default check " + c_id::get_description());

    // Memory allocated on 'host' or as 'shared' should be accessible to memory
    // allocated on 'device'
    if constexpr (c_id::kind == sycl::usm::alloc::host ||
                  c_id::kind == sycl::usm::alloc::shared) {
      if (allocation_supported<sycl::usm::alloc::device>(log, q)) {
        if (!check_by_index<T, memb_func_index::device, count, device_kernel>(
                log, ptr, q))
          FAIL(log, "Check on device " + c_id::get_description());
      } else {
        log.note("(skipped check on device)");
      }
    }
    // Memory allocated as 'shared' should be accessible to memory allocated
    // on 'host'
    if constexpr (c_id::kind == sycl::usm::alloc::shared) {
      if (allocation_supported<sycl::usm::alloc::host>(log, q)) {
        if (!check_by_index<T, memb_func_index::host, count, host_kernel>(
                log, ptr, q))
          FAIL(log, "Check on host " + c_id::get_description());
      } else {
        log.note("(skipped check on host)");
      }
    }
  }

  /** @brief Run an actual check for given case id
   *  @tparam c_id Id of current test case
   *  @tparam byQueue Defines the sycl::free strategy
   *  @param args Manualy passed parameters depending on the tested overload
   */
  template <typename c_id, typename byQueue, typename... argsT>
  void run(util::logger &log, argsT &&... args) {
    using namespace usm_alloc_help;
    T *ptr = nullptr;
    if (c_id::operation::form == usm_op_form::general) {
      ptr = allocate<c_id>(count, std::forward<argsT>(args)...);
    } else {
      ptr = allocate<c_id>(align, count, std::forward<argsT>(args)...);
    }
    if (ptr == nullptr) {
      FAIL(log, "allocation returned 'nullptr' in " + c_id::get_description());
      return;
    }
    // Custom deleter
    auto deleter = [&](T *ptr) { sycl::free(ptr, q); };
    std::unique_ptr<T, decltype(deleter)> free_ptr(ptr, deleter);

    // Check alignment
    if constexpr (c_id::operation::form == usm_op_form::aligned) {
      if (!check_alignment(ptr, align)) {
        FAIL(log, "wrong aligned pointer in " + c_id::get_description());
      }
    }

    fill_and_check<c_id>(log, ptr);
  }
};

/** @brief Run checks for 'malloc' or 'aligned_alloc' with given usm::alloc
 *         'kind'
 *  @tparam T Testing type
 *  @tparam op Defines name and form of usm allocation function
 *  @tparam count Count of elements to allocate
 *  @tparam kind usm::alloc kind of usm allocation
 *  @tparam templatedT Points to use a templated variant of usm function or not
 */
template <typename T, typename op, std::size_t count, sycl::usm::alloc kind,
          typename templatedT>
static void do_malloc_by_kind(util::logger &log, std::size_t align,
                              sycl::queue &q, const sycl::property_list pl) {
  using namespace usm_alloc_help;
  if (!allocation_supported<kind>(log, q)) {
    log.note("(skipped)");
    return;
  }
  auto dev = q.get_device();
  auto ctx = q.get_context();
  usm_allocations_test_helper<T, count> test(q, align);

  {
    /** Passing 'queue' as parameter
     */
    constexpr int case_num = 1;
    using id = case_id<op, case_num, kind, templatedT>;
    test.template run<id, by_queue>(log, q, kind);
  }
  {
    /** Passing 'context' as parameter
     */
    constexpr int case_num = 2;
    using id = case_id<op, case_num, kind, templatedT>;
    test.template run<id, by_context>(log, dev, ctx, kind);
  }
  // Cases for overloads with sycl::property_list will be added here
}

/** @brief Run checks for 'malloc_host' or 'aligned_alloc_host'
 *  @tparam T Testing type
 *  @tparam op Defines name and form of usm allocation function
 *  @tparam count Count of elements to allocate
 *  @tparam templatedT Points to use a templated variant of usm function or not
 */
template <typename T, typename op, std::size_t count, typename templatedT>
static void do_malloc_host(util::logger &log, std::size_t align, sycl::queue &q,
                           const sycl::property_list pl) {
  using namespace usm_alloc_help;
  constexpr sycl::usm::alloc kind = op::extract_kind();
  if (!allocation_supported<kind>(log, q)) {
    log.note("(skipped)");
    return;
  }
  auto ctx = q.get_context();
  usm_allocations_test_helper<T, count> test(q, align);

  {
    /** Passing 'queue' as parameter
     */
    constexpr int case_num = 1;
    using id = case_id<op, case_num, kind, templatedT>;
    test.template run<id, by_queue>(log, q);
  }
  {
    /** Passing 'context' as parameter
     */
    constexpr int case_num = 2;
    using id = case_id<op, case_num, kind, templatedT>;
    test.template run<id, by_context>(log, ctx);
  }
  // Cases for overloads with sycl::property_list will be added here
}

/** @brief Run checks for 'malloc_...' or 'aligned_alloc_...' (host/device)
 *  @tparam T Testing type
 *  @tparam op Defines name and form of usm allocation function
 *  @tparam count Count of elements to allocate
 *  @tparam templatedT Points to use a templated variant of usm function or not
 */
template <typename T, typename op, std::size_t count, typename templatedT>
static void do_malloc_by_operation(util::logger &log, std::size_t align,
                                   sycl::queue &q,
                                   const sycl::property_list pl) {
  using namespace usm_alloc_help;
  constexpr sycl::usm::alloc kind = op::extract_kind();
  if (!allocation_supported<kind>(log, q)) {
    log.note("(skipped)");
    return;
  }
  auto dev = q.get_device();
  auto ctx = q.get_context();
  usm_allocations_test_helper<T, count> test(q, align);

  {
    /** Passing 'queue' as parameter
     */
    constexpr int case_num = 1;
    using id = case_id<op, case_num, kind, templatedT>;
    test.template run<id, by_queue>(log, q);
  }
  {
    /** Passing 'context' as parameter
     */
    constexpr int case_num = 2;
    using id = case_id<op, case_num, kind, templatedT>;
    test.template run<id, by_context>(log, dev, ctx);
  }
  // Cases for overloads with sycl::property_list will be added here
}

/** @brief Main test-suite
 *  @tparam T Testing type
 *  @tparam op Defines name and form of usm allocation function
 */
template <typename T, typename op>
class check_usm_allocate_free {
 public:
  void operator()(util::logger &log) {
    constexpr auto device = sycl::usm::alloc::device;
    constexpr auto host = sycl::usm::alloc::host;
    constexpr auto shared = sycl::usm::alloc::shared;
    constexpr std::size_t count = 10;
    /** Possible universal values for alignment are powers of two (8, 16, etc)
     *  Behaviour is backend-specific if incorrect alignment passed
     */
    constexpr std::size_t align = 8;

    auto q = util::get_cts_object::queue();
    const sycl::property_list pl{};

    if constexpr (op::name == usm_op_name::malloc) {
      do_malloc_by_kind<T, op, count, device, non_templated>(log, align, q, pl);
      do_malloc_by_kind<T, op, count, host, non_templated>(log, align, q, pl);
      do_malloc_by_kind<T, op, count, shared, non_templated>(log, align, q, pl);

      do_malloc_by_kind<T, op, count, device, templated>(log, align, q, pl);
      do_malloc_by_kind<T, op, count, host, templated>(log, align, q, pl);
      do_malloc_by_kind<T, op, count, shared, templated>(log, align, q, pl);
    } else if constexpr (op::name == usm_op_name::malloc_host) {
      do_malloc_host<T, op, count, non_templated>(log, align, q, pl);
      do_malloc_host<T, op, count, templated>(log, align, q, pl);
    } else {
      do_malloc_by_operation<T, op, count, non_templated>(log, align, q, pl);
      do_malloc_by_operation<T, op, count, templated>(log, align, q, pl);
    }
  }
};

/** @brief Execute test for type defined by operation
 *  @tparam T Testing type
 *  @tparam op Defines name and form of usm allocation function
 */
template <typename T, typename op>
static void run_usm_test(util::logger &log) {
  try {
    check_usm_allocate_free<T, op>{}(log);
  } catch (const sycl::exception &e) {
    log_exception(log, e);
    std::string errorMsg =
        "a SYCL exception was caught: " + std::string(e.what());
    FAIL(log, errorMsg);
  } catch (const std::exception &e) {
    std::string errorMsg = "an exception was caught: " + std::string(e.what());
    FAIL(log, errorMsg);
  }
}

}  // namespace usm_allocate_free

#endif  // __SYCLCTS_TESTS_USM_ALLOC_FREE_H
