/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#define SYCL_SIMPLE_SWIZZLES

#include "../common/common.h"

#define TEST_NAME stream_api

namespace TEST_NAMESPACE {

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
  void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute the test
   */
  void run(util::logger &log) override {
    try {
      /** check cl::sycl::stream_manipulator
      */
      check_enum_class_value(cl::sycl::stream_manipulator::dec);
      check_enum_class_value(cl::sycl::stream_manipulator::hex);
      check_enum_class_value(cl::sycl::stream_manipulator::oct);
      check_enum_class_value(cl::sycl::stream_manipulator::noshowbase);
      check_enum_class_value(cl::sycl::stream_manipulator::showbase);
      check_enum_class_value(cl::sycl::stream_manipulator::noshowpos);
      check_enum_class_value(cl::sycl::stream_manipulator::showpos);
      check_enum_class_value(cl::sycl::stream_manipulator::endl);
      check_enum_class_value(cl::sycl::stream_manipulator::fixed);
      check_enum_class_value(cl::sycl::stream_manipulator::scientific);
      check_enum_class_value(cl::sycl::stream_manipulator::hexfloat);
      check_enum_class_value(cl::sycl::stream_manipulator::defaultfloat);

      /** Check stream interface
      */
      {
        auto testQueue = util::get_cts_object::queue();
        testQueue.submit([&](cl::sycl::handler &cgh) {

          cl::sycl::stream os(2048, 80, cgh);

          /** check get_size()
          */
          {
            auto size = os.get_size();
            check_return_type<size_t>(log, size,
                                      "cl::sycl::context::get_size()");
          }

          /** check get_max_statement_size()
          */
          {
            auto maxStatementSize = os.get_max_statement_size();
            check_return_type<size_t>(
                log, maxStatementSize,
                "cl::sycl::context::get_max_statement_size()");
          }

          cgh.single_task<class test_kernel_0>([=]() {});
        });
      }

      /** check stream operator for supported types
      */
      {
        auto testQueue = util::get_cts_object::queue();
        testQueue.submit([&](cl::sycl::handler &cgh) {

          cl::sycl::stream os(2048, 80, cgh);

          cgh.single_task<class test_kernel_1>([=]() {

            /** check stream operator for basic types
            */
            os << "hello world!";
            os << const_cast<char *>("hello world!");
            os << char('c');
            os << static_cast<signed char>('c');
            os << static_cast<unsigned char>('c');
            os << int(5);
            os << static_cast<unsigned int>(5);
            os << short(5);
            os << static_cast<unsigned short>(5);
            os << long(5);
            os << static_cast<unsigned long>(5);
            os << static_cast<long long>(5);
            os << static_cast<unsigned long long>(5);
            os << float(5.5f);
            os << double(5.5);
            os << true;
            os << size_t(5);

            // check stream operator for pointers
            int a = 5;
            int *aPtr = &a;
            os << aPtr;
            const int *const aConstPtr = &a;
            os << aConstPtr;
            auto multiPtr = cl::sycl::private_ptr<int>(aPtr);
            os << multiPtr;

            /** check stream operator for cl types
            */
            os << "hello world!";
            os << cl_char('c');
            os << static_cast<cl_uchar>('c');
            os << cl_int(5);
            os << static_cast<cl_uint>(5);
            os << cl_short(5);
            os << static_cast<cl_ushort>(5);
            os << cl_long(5);
            os << static_cast<cl_ulong>(5);
            os << cl_float(5.5f);
            os << cl_double(5.5);
            os << static_cast<cl_bool>(true);

            /** check stream operator for sycl types
            */
            os << cl::sycl::byte(72);
            os << cl::sycl::id<3>(1, 2, 3);
            os << cl::sycl::range<3>(1, 2, 3);
            os << cl::sycl::nd_range<3>(cl::sycl::range<3>(2, 4, 1),
                                        cl::sycl::range<3>(1, 2, 1));

            /** check stream operator for sycl vec types
            */
            cl::sycl::vec<float, 4> f4(1.0f, 2.0f, 3.0f, 4.0f);
            os << f4;
            os << f4.wzyx();

            /** check stream operator for manipulators
            */
            os << cl::sycl::endl;
            os << cl::sycl::setprecision(5) << float(5.0f);
            os << cl::sycl::setw(3) << float(5.0f);
            os << cl::sycl::hex << float(5.0f);
            os << cl::sycl::oct << float(5.0f);
            os << cl::sycl::dec << float(5.0f);
            os << cl::sycl::showbase << int(5);
            os << cl::sycl::noshowbase << int(5);
            os << cl::sycl::showpos << int(-5) << int(5);
            os << cl::sycl::noshowpos << int(-5) << int(5);
            os << cl::sycl::fixed << float(5.0f);
            os << cl::sycl::scientific << float(5.0f);
            os << cl::sycl::hexfloat << float(5.0f);
            os << cl::sycl::defaultfloat << float(5.0f);
          });
        });
      }

      /** check stream operator for cl::sycl::nd_item
      */
      {
        auto testQueue = util::get_cts_object::queue();
        testQueue.submit([&](cl::sycl::handler &cgh) {

          cl::sycl::stream os(2048, 80, cgh);

          cgh.parallel_for<class test_kernel_2>(
              cl::sycl::nd_range<3>(cl::sycl::range<3>(2, 4, 1),
                                    cl::sycl::range<3>(1, 2, 1)),
              [=](cl::sycl::nd_item<3> ndItem) {
                /** check stream operator for nd_item
                */
                os << ndItem;

                // check stream operator for nd_range
                os << ndItem.get_nd_range();
              });
        });
      }

      /** check stream operator for cl::sycl::item
      */
      {
        auto testQueue = util::get_cts_object::queue();
        testQueue.submit([&](cl::sycl::handler &cgh) {

          cl::sycl::stream os(2048, 80, cgh);

          cgh.parallel_for<class test_kernel_3>(cl::sycl::range<3>(4, 2, 1),
                                                [=](cl::sycl::item<3> it) {
                                                  /** check stream operator for
                                                   * item
                                                  */
                                                  os << it;
                                                });
        });
      }

      /** check stream operator for cl::sycl::group and cl::sycl::h_item
      */
      {
        auto testQueue = util::get_cts_object::queue();
        testQueue.submit([&](cl::sycl::handler &cgh) {

          cl::sycl::stream os(2048, 80, cgh);

          cgh.parallel_for_work_group<class test_kernel_4>(
              cl::sycl::range<3>(4, 2, 1), cl::sycl::range<3>(1, 1, 1),
              [=](cl::sycl::group<3> gp) {
                /** check stream operator for cl::sycl::group
                */
                os << gp;

                gp.parallel_for_work_item([&](cl::sycl::h_item<3> hit) {
                  /** check stream operator for cl::sycl::h_item
                  */
                  os << hit;
                });
              });
        });
      }

      // Check stream operator for cl::sycl::half
      {
        auto testQueue = util::get_cts_object::queue();

        if (testQueue.get_device().has_extension("cl_khr_fp16")) {
          testQueue.submit([&](cl::sycl::handler &cgh) {

            cl::sycl::stream os(2048, 80, cgh);

            cgh.single_task<class test_kernel_5>([=]() {
              os << cl::sycl::half(0.2f);
              os << cl::sycl::cl_half(0.3f);
            });
          });
        }
      }
    } catch (const cl::sycl::exception &e) {
      log_exception(log, e);
      cl::sycl::string_class errorMsg =
          "a SYCL exception was caught: " + cl::sycl::string_class(e.what());
      FAIL(log, errorMsg.c_str());
    }
  }
};

// register this test with the test_collection.
util::test_proxy<TEST_NAME> proxy;

} /* namespace TEST_NAMESPACE */
