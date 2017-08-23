/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME stream_api

namespace stream_api__ {
using namespace sycl_cts;

/**
 * Function that streams a type using the cl::sycl::stream object.
 */
template <typename T>
void stream_type(cl::sycl::stream &os, T var) {
  os << var;
}

/** test cl::sycl::stream interface
*/
class TEST_NAME : public util::test_base {
 public:
  /** return information about this test
   */
  virtual void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute the test
   */
  virtual void run(util::logger &log) override {
    try {
      auto testQueue = util::get_cts_object::queue();
      testQueue.submit([&](cl::sycl::handler &cgh) {

        cl::sycl::stream os(2048, 80, cgh);

        auto size = os.get_size();

        check_return_type<size_t>(log, size, "cl::sycl::context::get_size()");

        auto maxStatementSize = os.get_max_statement_size();

        check_return_type<size_t>(
            log, maxStatementSize,
            "cl::sycl::context::get_max_statement_size()");
      });

      testQueue.submit([&](cl::sycl::handler &cgh) {

        cl::sycl::stream os;

        cgh.single_task<class test_kernel_1>([=]() {
          /** check stream operator for basic types
          */
          os << cl::sycl::string_class("hello world!");
          os << char('c');
          os << static_cast<unsigned char>('c');
          os << int(5);
          os << static_cast<unsigned int>(5);
          os << short(5);
          os << static_cast<unsigned short>(5);
          os << long(5);
          os << static_cast<unsigned long>(5);
          os << float(5.5f);
          os << double(5.5);
          os << size_t(5);
          os << true;

          /** check stream operator for sycl types
          */
          os << cl::sycl::id<3>(1, 2, 3);
          os << cl::sycl::range<3>(1, 2, 3);

          /** check stream operator for sycl vec types
          */
          cl::sycl::vec<float, 4> f4(1.0f, 2.0f, 3.0f, 4.0f);
          os << f4;
          os << f4.wzyx();

          /** check stream operator for manipulators
          */
          os << cl::sycl::endl;
          os << cl::sycl::precision << float(5.0f);
          os << cl::sycl::scientific << float(5.0f);
          os << cl::sycl::hex << float(5.0f);
          os << cl::sycl::oct << float(5.0f);
          os << cl::sycl::showbase << int(5);
          os << cl::sycl::showpos << int(-5) << int(5);
        });
      });

      testQueue.submit([&](cl::sycl::handler &cgh) {

        cgh.parallel_for<class test_kernel_2>(
            cl::sycl::nd_range<3>(cl::sycl::range<3>(16, 8, 4),
                                  cl::sycl::range<3>(8, 4, 2)),
            [=](cl::sycl::nd_item<3> ndItem) {
              /** check stream operator for nd_item
              */
              cl::sycl::stream os(2048, 80, cgh);
              os << ndItem;
            });
      });

      testQueue.submit([&](cl::sycl::handler &cgh) {

        cgh.parallel_for<class test_kernel_3>(
            cl::sycl::range<3>(16, 8, 4), [=](cl::sycl::item<3> it) {
              /** check stream operator for item
              */
              cl::sycl::stream os(2048, 80, cgh);
              os << it;
            });
      });

      testQueue.submit([&](cl::sycl::handler &cgh) {

        cgh.parallel_for_work_group<class test_kernel_4>(
            cl::sycl::range<3>(16, 8, 4), cl::sycl::range<3>(1, 1, 1),
            [=](cl::sycl::group<3> gp) {
              /** check stream operator for group
              */
              cl::sycl::stream os(2048, 80, cgh);
              os << gp;
            });
      });
    } catch (cl::sycl::exception e) {
      log_exception(log, e);
      cl::sycl::string_class errorMsg =
          "a SYCL exception was caught: " + cl::sycl::string_class(e.what());
      FAIL(log, errorMsg.c_str());
    }
  }
};

// register this test with the test_collection.
util::test_proxy<TEST_NAME> proxy;

} /* namespace stream_api__ */
