/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2015 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"
#ifdef _WIN32
#include <windows.h>
#endif
#include "../../gl_util/gl_util.h"

#define TEST_NAME opencg_interop_interface

namespace opencg_interop_interface__ {
using namespace sycl_cts;
using namespace cl::sycl;

GLenum targets_1d[] = {GL_TEXTURE_1D, GL_TEXTURE_1D_ARRAY, GL_TEXTURE_BUFFER,
                       0};

GLenum targets_2d[] = {
    GL_TEXTURE_2D, GL_TEXTURE_2D_ARRAY, GL_TEXTURE_CUBE_MAP_POSITIVE_X,
    GL_TEXTURE_CUBE_MAP_POSITIVE_Y, GL_TEXTURE_CUBE_MAP_POSITIVE_Z,
    GL_TEXTURE_CUBE_MAP_NEGATIVE_X, GL_TEXTURE_CUBE_MAP_NEGATIVE_Y,
    GL_TEXTURE_CUBE_MAP_NEGATIVE_Z, GL_TEXTURE_RECTANGLE, 0};

GLenum targets_3d[] = {GL_TEXTURE_3D, 0};

/** Test for the SYCL buffer OpenGL interoperation interface
 */
class TEST_NAME : public sycl_cts::util::test_base_opencl {
 public:
  /** return information about this test
   */
  virtual void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute this test
   */
  virtual void run(util::logger &log) override {
    try {
      context ctx;
      GLuint m_gl_buffer_obj = 2;

      buffer<int, 1> buffer_gl(ctx, m_gl_buffer_obj);
      cl_gl_object_type tmp;
      tmp = get_gl_info(m_gl_buffer_obj);

      image<1> img_1d(ctx, m_gl_buffer_obj);
      image<2> img_2d(ctx, m_gl_buffer_obj);

      // test for access::target
      cts_selector selector;
      cl::sycl::queue queue(selector);

      queue.submit([&](cl::sycl::handler &cgh) {
        auto acc_img = img_1d.get_access<float4, access::cl_gl_image>(cgh);
        auto acc_buf = img_1d.get_access<float4, access::cl_gl_buffer>(cgh);
      });

      GLuint m_texture = 4;
      int m_miplevel = 1;

      // One dim constructors
      for (int i = 0; targets_1d[i] != 0; i++) {
        image<1> img_1d_adv(ctx, targets_1d[i], m_texture, m_miplevel);
        if (img_1d_adv.get_gl_texture_target() != targets_1d[i]) {
          FAIL(log, "Target not as expected.");
        }

        if (img_1d_adv.get_gl_mipmap_level() != m_miplevel) {
          FAIL(log, "Miplevel not as expected.");
        }
      }

      // Two dims constructors
      for (int i = 0; targets_2d[i] != 0; i++) {
        image<2> img_2d_adv(ctx, targets_2d[i], m_texture, m_miplevel);
        if (img_2d_adv.get_gl_texture_target() != targets_2d[i]) {
          FAIL(log, "Target not as expected.");
        }

        if (img_2d_adv.get_gl_mipmap_level() != m_miplevel) {
          FAIL(log, "Miplevel not as expected.");
        }
      }

      // Three dims constructor
      for (int i = 0; targets_3d[i] != 0; i++) {
        image<3> img_3d_adv(ctx, targets_3d[i], m_texture, m_miplevel);
        if (img_3d_adv.get_gl_texture_target() != targets_3d[i]) {
          FAIL(log, "Target not as expected.");
        }

        if (img_3d_adv.get_gl_mipmap_level() != m_miplevel) {
          FAIL(log, "Miplevel not as expected.");
        }
      }

      GLsync sync_obj;
      event evnt(ctx, sync_obj);
      GLsync gl_info = evnt.get_gl_info();

      queue.wait_and_throw();
    } catch (cl::sycl::exception e) {
      log_exception(log, e);
      FAIL(log, "sycl exception caught");
    }
  }
};

// register this test with the test_collection
util::test_proxy<TEST_NAME> proxy;

} /* namespace opencl_interop_buffer__ */
