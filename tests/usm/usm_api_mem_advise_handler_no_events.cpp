/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for overload of the handler::mem_advise() member function
//  with no events
//
*******************************************************************************/

#include "../common/common.h"
#include "../common/disabled_for_test_case.h"
#include "usm_api.h"

namespace usm_api_mem_advise_handler_no_events {
using namespace usm_api;

DISABLED_FOR_TEST_CASE(SimSYCL)
("usm_api_mem_advise_handler_no_events", "[usm]")({
  sycl_cts::util::logger log;
  run_all_tests<int, tests::mem_advise, caller::handler, 0_events>{}(log);
});

}  // namespace usm_api_mem_advise_handler_no_events
