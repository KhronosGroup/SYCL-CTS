/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2015 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME std_classes

namespace std_classes__
{
using namespace sycl_cts;

/** check vector_class
*/
template <typename T, class Alloc>
using vectorClass = cl::sycl::vector_class<T, Alloc>;

/** check string_class
*/
using stringClass = cl::sycl::string_class;

/** check unique_ptr
*/
template <typename T>
using uniquePtr = cl::sycl::unique_ptr<T>;

/** check shared_ptr
*/
template <typename T>
using sharedPtr = cl::sycl::shared_ptr<T>;

/** check weak_ptr
*/
template <typename T>
using weakPtr = cl::sycl::weak_ptr<T>;

/** check mutex_class
*/
using mutexClass = cl::sycl::mutex_class;

/** check function_class
*/
template <typename R, typename... Args>
using functionClass = cl::sycl::function_class<R(Args...)>;


/** tests the availability of std classes
*/
class TEST_NAME : public util::test_base
{
public:
    /** return information about this test
    */
    virtual void get_info( test_base::info &out ) const override
    {
        set_test_info( out, TOSTRING( TEST_NAME ), TEST_FILE );
    }

    /** execute this test
    */
    virtual void run( util::logger &log ) override
    {
    }
};

// register this test with the test_collection
util::test_proxy<TEST_NAME> proxy;

} /* namespace std_classes__ */
