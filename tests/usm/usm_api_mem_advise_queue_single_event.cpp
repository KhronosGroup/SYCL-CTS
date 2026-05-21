/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for overload of the queue::mem_advise() method with 1 event
//
*******************************************************************************/

#include "../common/common.h"
#include "../common/disabled_for_test_case.h"
#include "usm_api.h"

namespace usm_api_mem_advise_queue_single_event {
using namespace usm_api;

DISABLED_FOR_TEST_CASE(SimSYCL)
("usm_api_mem_advise_queue_single_event", "[usm]")({
  sycl_cts::util::logger log;
  run_all_tests<int, tests::mem_advise, caller::queue, 1_events>{}(log);
});

}  // namespace usm_api_mem_advise_queue_single_event
