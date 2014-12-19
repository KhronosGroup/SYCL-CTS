/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2014 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#pragma once

#include "singleton.h"

namespace sycl_cts
{
namespace util
{

/** this class oversees the execution of tests
 *
 */
class executor : public singleton<executor>
{
public:
    /** execute all tests currently in the collection
     */
    void run_all();
};

}  // namespace util
}  // namespace sycl_cts
