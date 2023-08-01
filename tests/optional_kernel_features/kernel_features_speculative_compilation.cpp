/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides test when multiple kernels with different attributes and features
//  are defined in the same translation unit.
//
*******************************************************************************/

#include "../common/disabled_for_test_case.h"
#include "catch2/catch_template_test_macros.hpp"
#include "kernel_features_common.h"

namespace kernel_features_speculative_compilation {
using namespace sycl_cts;
using namespace kernel_features_common;

class empty_functor {
 public:
  void operator()() const {}
  void operator()(sycl::item<1>) const {}
  void operator()(sycl::group<1>) const {}
};

template <size_t N>
class sub_group_decorated_functor {
 public:
  [[sycl::reqd_sub_group_size(N)]] void operator()(sycl::nd_item<1>) const {}
  [[sycl::reqd_sub_group_size(N)]] void operator()(sycl::group<1>) const {}
};

template <size_t N>
class work_group_decorated_functor {
 public:
  [[sycl::reqd_work_group_size(N)]] void operator()(sycl::nd_item<1>) const {}
  [[sycl::reqd_work_group_size(N)]] void operator()(sycl::group<1>) const {}
};

using AtomicRefT =
    sycl::atomic_ref<unsigned long long, sycl::memory_order::relaxed,
                     sycl::memory_scope::device>;
template <int Case>
class kernel_speculative;

DISABLED_FOR_TEST_CASE(hipSYCL)
("Speculative compilation with supported feature", "[kernel_features]")({
  auto queue = util::get_cts_object::queue();
  const sycl::errc errc_expected = sycl::errc::success;
  constexpr bool is_exception_expected = false;

  // Kernels that doesn't use any feature and don't require sub-group or work
  // group size
  {
    {
      const auto separate_lambda_no_arg = []() {};
      const auto separate_lambda_item_arg = [](sycl::item<1>) {};
      const auto separate_lambda_group_arg = [](sycl::group<1>) {};

      run_separate_lambda<kernel_speculative<1>>(
          is_exception_expected, errc_expected, queue, separate_lambda_no_arg,
          separate_lambda_item_arg, separate_lambda_group_arg);
    }

    {
      using FunctorT = empty_functor;
      run_functor<FunctorT>(is_exception_expected, errc_expected, queue);
    }

    {
      RUN_SUBMISSION_CALL(is_exception_expected, errc_expected, queue,
                          NO_ATTRIBUTE, kernel_speculative<1>, NO_KERNEL_BODY);
    }
  }

  if (queue.get_device().has(sycl::aspect::fp16)) {
    {
      const auto separate_lambda_no_arg = []() {
        use_feature_function_non_decorated<sycl::half>();
      };
      const auto separate_lambda_item_arg = [](sycl::item<1>) {
        use_feature_function_non_decorated<sycl::half>();
      };
      const auto separate_lambda_group_arg = [](sycl::group<1>) {
        use_feature_function_non_decorated<sycl::half>();
      };

      run_separate_lambda<kernel_speculative<2>>(
          is_exception_expected, errc_expected, queue, separate_lambda_no_arg,
          separate_lambda_item_arg, separate_lambda_group_arg);
    }

    {
      using FunctorT = non_decorated_call_non_decorated_function<sycl::half>;
      run_functor<FunctorT>(is_exception_expected, errc_expected, queue);
    }

    {
      RUN_SUBMISSION_CALL(is_exception_expected, errc_expected, queue,
                          NO_ATTRIBUTE, kernel_speculative<2>,
                          use_feature_function_non_decorated<sycl::half>());
    }
  }

  if (queue.get_device().has(sycl::aspect::fp64)) {
    {
      const auto separate_lambda_no_arg = []() {
        use_feature_function_non_decorated<double>();
      };
      const auto separate_lambda_item_arg = [](sycl::item<1>) {
        use_feature_function_non_decorated<double>();
      };
      const auto separate_lambda_group_arg = [](sycl::group<1>) {
        use_feature_function_non_decorated<double>();
      };

      run_separate_lambda<kernel_speculative<3>>(
          is_exception_expected, errc_expected, queue, separate_lambda_no_arg,
          separate_lambda_item_arg, separate_lambda_group_arg);
    }

    {
      using FunctorT = non_decorated_call_non_decorated_function<double>;
      run_functor<FunctorT>(is_exception_expected, errc_expected, queue);
    }

    {
      RUN_SUBMISSION_CALL(is_exception_expected, errc_expected, queue,
                          NO_ATTRIBUTE, kernel_speculative<3>,
                          use_feature_function_non_decorated<double>());
    }
  }

  if (queue.get_device().has(sycl::aspect::atomic64)) {
    {
      const auto separate_lambda_no_arg = []() {
        use_feature_function_non_decorated<AtomicRefT>();
      };
      const auto separate_lambda_item_arg = [](sycl::item<1>) {
        use_feature_function_non_decorated<AtomicRefT>();
      };
      const auto separate_lambda_group_arg = [](sycl::group<1>) {
        use_feature_function_non_decorated<AtomicRefT>();
      };

      run_separate_lambda<kernel_speculative<4>>(
          is_exception_expected, errc_expected, queue, separate_lambda_no_arg,
          separate_lambda_item_arg, separate_lambda_group_arg);
    }

    {
      using FunctorT = non_decorated_call_non_decorated_function<AtomicRefT>;
      run_functor<FunctorT>(is_exception_expected, errc_expected, queue);
    }

    {
      RUN_SUBMISSION_CALL(is_exception_expected, errc_expected, queue,
                          NO_ATTRIBUTE, kernel_speculative<4>,
                          use_feature_function_non_decorated<AtomicRefT>());
    }
  }

  static constexpr size_t testing_wg_size[2] = {16, 4294967295};
  auto max_wg_size =
      queue.get_device().get_info<sycl::info::device::max_work_group_size>();

  if (max_wg_size >= testing_wg_size[0]) {
    {
      const auto separate_lambda_item_arg = [](sycl::item<1>)
          [[sycl::reqd_work_group_size(testing_wg_size[0])]]{};
      const auto separate_lambda_group_arg = [](sycl::group<1>)
          [[sycl::reqd_work_group_size(testing_wg_size[0])]]{};

      run_separate_lambda_nd_range<kernel_speculative<5>>(
          is_exception_expected, errc_expected, queue, separate_lambda_item_arg,
          separate_lambda_group_arg);
    }

    {
      using FunctorT = work_group_decorated_functor<testing_wg_size[0]>;
      run_functor_nd_range<FunctorT>(is_exception_expected, errc_expected,
                                     queue);
    }

    {
      RUN_SUBMISSION_CALL_ND_RANGE(
          testing_wg_size[0], 1, is_exception_expected, errc_expected, queue,
          [[sycl::reqd_work_group_size(testing_wg_size[0])]],
          kernel_speculative<5>, NO_KERNEL_BODY);
    }
  }

  if (max_wg_size >= testing_wg_size[1]) {
    {
      const auto separate_lambda_item_arg = [](sycl::item<1>)
          [[sycl::reqd_work_group_size(testing_wg_size[1])]]{};
      const auto separate_lambda_group_arg = [](sycl::group<1>)
          [[sycl::reqd_work_group_size(testing_wg_size[1])]]{};

      run_separate_lambda_nd_range<kernel_speculative<6>>(
          is_exception_expected, errc_expected, queue, separate_lambda_item_arg,
          separate_lambda_group_arg);
    }

    {
      using FunctorT = work_group_decorated_functor<testing_wg_size[1]>;
      run_functor_nd_range<FunctorT>(is_exception_expected, errc_expected,
                                     queue);
    }

    {
      RUN_SUBMISSION_CALL_ND_RANGE(
          testing_wg_size[1], 1, is_exception_expected, errc_expected, queue,
          [[sycl::reqd_work_group_size(testing_wg_size[1])]],
          kernel_speculative<6>, NO_KERNEL_BODY);
    }
  }

  static constexpr size_t testing_sg_size[2] = {16, 4099};

  const auto sg_sizes_vec =
      queue.get_device().get_info<sycl::info::device::sub_group_sizes>();
  auto find_res =
      std::find(sg_sizes_vec.begin(), sg_sizes_vec.end(), testing_sg_size[0]);
  if (find_res != sg_sizes_vec.end()) {
    {
      const auto separate_lambda_item_arg =
          [](sycl::item<1>) [[sycl::reqd_sub_group_size(testing_sg_size[0])]]{};
      const auto separate_lambda_group_arg = [](sycl::group<1>)
          [[sycl::reqd_sub_group_size(testing_sg_size[0])]]{};

      run_separate_lambda_nd_range<kernel_speculative<7>>(
          is_exception_expected, errc_expected, queue, separate_lambda_item_arg,
          separate_lambda_group_arg);
    }

    {
      using FunctorT = sub_group_decorated_functor<testing_sg_size[0]>;
      run_functor_nd_range<FunctorT>(is_exception_expected, errc_expected,
                                     queue);
    }

    {
      RUN_SUBMISSION_CALL_ND_RANGE(
          testing_sg_size[0], 1, is_exception_expected, errc_expected, queue,
          [[sycl::reqd_sub_group_size(testing_sg_size[0])]],
          kernel_speculative<7>, NO_KERNEL_BODY);
    }
  }

  find_res =
      std::find(sg_sizes_vec.begin(), sg_sizes_vec.end(), testing_sg_size[1]);
  if (find_res != sg_sizes_vec.end()) {
    {
      const auto separate_lambda_item_arg =
          [](sycl::item<1>) [[sycl::reqd_sub_group_size(testing_sg_size[1])]]{};
      const auto separate_lambda_group_arg = [](sycl::group<1>)
          [[sycl::reqd_sub_group_size(testing_sg_size[1])]]{};

      run_separate_lambda_nd_range<kernel_speculative<8>>(
          is_exception_expected, errc_expected, queue, separate_lambda_item_arg,
          separate_lambda_group_arg);
    }

    {
      using FunctorT = sub_group_decorated_functor<testing_sg_size[1]>;
      run_functor_nd_range<FunctorT>(is_exception_expected, errc_expected,
                                     queue);
    }

    {
      RUN_SUBMISSION_CALL_ND_RANGE(
          testing_sg_size[1], 1, is_exception_expected, errc_expected, queue,
          [[sycl::reqd_sub_group_size(testing_sg_size[1])]],
          kernel_speculative<8>, NO_KERNEL_BODY);
    }
  }
});
}  // namespace kernel_features_speculative_compilation
