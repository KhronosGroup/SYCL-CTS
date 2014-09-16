/*************************************************************************
//
//  SYCL Conformance Test Suite
// 
//  Copyright:	(c) 2014 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#pragma once

// this define will enable and disable standard compliant test features.
// since SYCLONE currently does not meet the full standard it is desirable
// to have a method to skip broken tests, so that the test suite will compile.
#define ENABLE_FULL_TEST 0

// test framework specific device selector
#include "../common/cts_selector.h"
#include "../../util/proxy.h"
#include "macros.h"

#include "../../util/test_base.h"
#include "../../util/test_base_opencl.h"
