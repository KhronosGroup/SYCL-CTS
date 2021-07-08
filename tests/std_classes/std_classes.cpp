/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME std_classes

namespace std_classes__ {
using namespace sycl_cts;

/** check vector_class
*/
template <class T, class Alloc>
using vectorClass = sycl::vector_class<T, Alloc>;

/** check string_class
*/
using stringClass = sycl::string_class;

/** check function_class
*/
template <class R, class... Args>
using functionClass = sycl::function_class<R(Args...)>;

/** check mutex_class
*/
using mutexClass = sycl::mutex_class;

/** check unique_ptr_class
*/
template <class T, class D>
using uniquePtrClass = sycl::unique_ptr_class<T, D>;

/** check shared_ptr_class
*/
template <class T>
using sharedPtrClass = sycl::shared_ptr_class<T>;

/** check weak_ptr_class
*/
template <class T>
using weakPtrClass = sycl::weak_ptr_class<T>;

/** check hash_class
*/
template <class T>
using hashClass = sycl::hash_class<T>;

/** check exception_ptr_class
*/
using exceptionPtrClass = sycl::exception_ptr_class;

struct custom_deleter {
  void operator()(int *p) const {};
};

/** tests the availability of std classes
*/
class TEST_NAME : public util::test_base {
 public:
  /** return information about this test
  */
  void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute this test
  */
  void run(util::logger &log) override {
    /* Try instantiating these classes */
    {
      /** check vector_class
      */
      sycl::vector_class<int> vector;

      /** check string_class
      */
      stringClass string;

      /** check function_class
      */
      functionClass<void> function;

      /** check mutex_class
      */
      mutexClass mutex;

      /** check unique_ptr_class
      */
      uniquePtrClass<int, custom_deleter> uniquePtr;

      /** check shared_ptr_class
      */
      sharedPtrClass<int> sharedPtr;

      /** check weak_ptr_class
      */
      weakPtrClass<int> weakPtr;

      /** check hash_class
      */
      hashClass<int> hash;

      /** check exception_ptr_class
      */
      exceptionPtrClass exceptionPtr;
    }
  }
};

// register this test with the test_collection
util::test_proxy<TEST_NAME> proxy;

} /* namespace std_classes__ */
