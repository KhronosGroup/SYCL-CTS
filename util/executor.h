/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#ifndef __SYCLCTS_UTIL_EXECUTOR_H
#define __SYCLCTS_UTIL_EXECUTOR_H

#include "singleton.h"

namespace sycl_cts {
namespace util {

/** this class oversees the execution of tests
 *
 */
class executor : public singleton<executor> {
 public:
  /** execute all tests currently in the collection
   */
  bool run_all();
};

}  // namespace util
}  // namespace sycl_cts

#endif  // __SYCLCTS_UTIL_EXECUTOR_H