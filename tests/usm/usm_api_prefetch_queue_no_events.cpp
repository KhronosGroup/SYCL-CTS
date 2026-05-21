/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for overload of the queue::prefetch() method with no events
//
*******************************************************************************/

#include "../common/common.h"
#include "../common/disabled_for_test_case.h"
#include "usm_api.h"

namespace usm_api_prefetch_queue_no_events {
using namespace usm_api;

DISABLED_FOR_TEST_CASE(SimSYCL)
("usm_api_prefetch_queue_no_events", "[usm]")({
  sycl_cts::util::logger log;
  run_all_tests<int, tests::prefetch, caller::queue, 0_events>{}(log);
});

}  // namespace usm_api_prefetch_queue_no_events
