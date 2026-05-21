/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for overload of the queue::prefetch() method with multiple
//  events
//
*******************************************************************************/

#include "../common/common.h"
#include "../common/disabled_for_test_case.h"
#include "usm_api.h"

namespace usm_api_prefetch_queue_multiple_events {
using namespace usm_api;

DISABLED_FOR_TEST_CASE(SimSYCL)
("usm_api_prefetch_queue_multiple_events", "[usm]")({
  sycl_cts::util::logger log;
  run_all_tests<int, tests::prefetch, caller::queue, multiple_events>{}(log);
});

}  // namespace usm_api_prefetch_queue_multiple_events
