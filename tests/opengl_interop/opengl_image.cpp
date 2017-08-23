/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2015 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"
#include "../../gl_util/gl_util.h"

//#define GL_GLEXT_PROTOTYPES 1
#ifdef _WIN32
#include <windows.h>
#endif
//#include <GL/gl.h>
//#include "../../gl_util/glext.h"

#define TEST_NAME opengl_image

namespace opencl_image__ {
using namespace sycl_cts;
using namespace cl::sycl;

const float golden = 0.2f;
float vertices[] = {-1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f,  1.0f,  1.0f,
                    1.0f,  -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f,
                    -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f,  -1.0f};

float pixels[] = {0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f,
                  1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f};

GLenum targets_1d[] = {GL_TEXTURE_1D, GL_TEXTURE_1D_ARRAY, GL_TEXTURE_BUFFER,
                       0};

GLenum targets_2d[] = {GL_TEXTURE_2D,
                       GL_TEXTURE_2D_ARRAY,
                       GL_TEXTURE_CUBE_MAP_POSITIVE_X,
                       GL_TEXTURE_CUBE_MAP_POSITIVE_Y,
                       GL_TEXTURE_CUBE_MAP_POSITIVE_Z,
                       GL_TEXTURE_CUBE_MAP_NEGATIVE_X,
                       GL_TEXTURE_CUBE_MAP_NEGATIVE_Y,
                       GL_TEXTURE_CUBE_MAP_NEGATIVE_Z,
                       GL_TEXTURE_RECTANGLE,
                       0};

GLenum targets_3d[] = {GL_TEXTURE_3D, 0};

void perform(util::logger& log, context& sycl_ctx) {
  // buffer
  {
    GLuint vbo_vertices;
    glGenBuffersARB(1, &vbo_vertices);
    glBindBufferARB(GL_ARRAY_BUFFER_ARB, vbo_vertices);
    glBufferDataARB(GL_ARRAY_BUFFER_ARB, 24 * sizeof(float), vertices,
                    GL_STATIC_DRAW_ARB);

    {
      image<1> img(sycl_ctx, vbo_vertices);

      queue queue;
      queue.submit([&](handler& cgh) {
        auto img_acc =
            img.template get_access<float, cl::sycl::access::mode::read_write>(
                cgh);
        auto my_range = nd_range<1>(range<1>(24), range<1>(1));
        auto my_kernel =
            ([=](item<1> item) { img_acc[item.get_global(0)] = golden; });
        cgh.parallel_for<class TEST_NAME>(my_range, my_kernel);
      });

      auto img_acc_host =
          img.template get_access<float, cl::sycl::access::mode::read,
                                  access::host_image>();
      for (int i = 0; i < 24, i++) {
        if (img_acc_host[i] != golden) {
          FAIL(log, "Value not updated.");
        }
      }

      void* gl_ptr = glMapNamedBuffer(vbo_vertices, GL_READ_ONLY);
      float* gl_ptr_float = static_cast<float>(gl_ptr);
      for (int i = 0; i < 24, i++) {
        if (gl_ptr_float[i] != golden) {
          FAIL(log, "GL buffer is not as expected.");
        }
      }
      glUnmapNamedBuffer(vbo_vertices);
    }

  }  // end buffer

  // renderbuffer
  {
    GLuint render_buffer;
    glGenRenderbuffers(1, &render_buffer);
    glBindRenderbuffer(GL_RENDERBUFFER, render_buffer);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_RGB, 1, 8);

    {
      image<2> img(sycl_ctx, render_buffer);

      queue queue;
      queue.submit([&](handler& cgh) {
        auto img_acc =
            img.template get_access<float, cl::sycl::access::mode::read_write>(
                cgh);
        auto my_range = nd_range<2>(range<2>(1, 24), range<2>(1, 1));
        auto my_kernel =
            ([=](item<1> item) { img_acc[item.get_global(0)] = golden; });
        cgh.parallel_for<class TEST_NAME>(my_range, my_kernel);
      });

      auto img_acc_host =
          img.template get_access<float, cl::sycl::access::mode::read,
                                  access::host_image>();
      for (int i = 0; i < 24, i++) {
        if (img_acc_host[i] != golden) {
          FAIL(log, "Value not updated.");
        }
      }

      void* gl_ptr = glMapNamedBuffer(vbo_vertices, GL_READ_ONLY);
      float* gl_ptr_float = static_cast<float>(gl_ptr);
      for (int i = 0; i < 24, i++) {
        if (gl_ptr_float[i] != golden) {
          FAIL(log, "GL buffer is not as expected.");
        }
      }
      glUnmapNamedBuffer(vbo_vertices);
    }

  }  // end renderbuffer

  // texture 1D
  for (int i = 0; targets_1d[i] != 0; i++) {
    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(targets_1d[i], tex);
    glTexImage1D(targets_1d[i], 0, GL_RGB, 4, 0, GL_RGB, GL_FLOAT, pixels);

    {
      image<1> img(sycl_ctx, targets_1d[i], tex, 0);

      queue queue;
      queue.submit([&](handler& cgh) {
        auto img_acc =
            img.template get_access<float, cl::sycl::access::mode::read_write>(
                cgh);
        auto my_range = nd_range<2>(range<2>(4, 3), range<2>(1, 1));
        auto my_kernel =
            ([=](item<1> item) { img_acc[item.get_global(0)] = golden; });
        cgh.parallel_for<class TEST_NAME>(my_range, my_kernel);
      });

      auto img_acc_host =
          img.template get_access<float, cl::sycl::access::mode::read,
                                  access::host_image>();
      for (int i = 0; i < 12, i++) {
        if (img_acc_host[i] != golden) {
          FAIL(log, "Value not updated.");
        }
      }

      void* gl_ptr = glMapNamedBuffer(tex, GL_READ_ONLY);
      float* gl_ptr_float = static_cast<float>(gl_ptr);
      for (int i = 0; i < 12, i++) {
        if (gl_ptr_float[i] != golden) {
          FAIL(log, "GL buffer is not as expected.");
        }
      }
      glUnmapNamedBuffer(tex);
    }
    glDeleteTextures(1, &tex);

  }  // end texture 1D

  // texture 2d
  for (int i = 0; targets_2d[i] != 0; i++) {
    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(targets_2d[i], tex);
    glTexImage2D(targets_2d[i], 0, GL_RGB, 2, 2, 0, GL_RGB, GL_FLOAT, pixels);

    {
      queue queue;
      queue.submit([&](handler& cgh) {
        auto img_acc =
            img.template get_access<float, cl::sycl::access::mode::read_write>(
                cgh);
        auto my_range = nd_range<2>(range<2>(4, 3), range<2>(1, 1));
        auto my_kernel =
            ([=](item<1> item) { img_acc[item.get_global(0)] = golden; });
        cgh.parallel_for<class TEST_NAME>(my_range, my_kernel);
      });

      auto img_acc_host =
          img.template get_access<float, cl::sycl::access::mode::read,
                                  access::host_image>();
      for (int i = 0; i < 12, i++) {
        if (img_acc_host[i] != golden) {
          FAIL(log, "Value not updated.");
        }
      }

      void* gl_ptr = glMapNamedBuffer(tex, GL_READ_ONLY);
      float* gl_ptr_float = static_cast<float>(gl_ptr);
      for (int i = 0; i < 12, i++) {
        if (gl_ptr_float[i] != golden) {
          FAIL(log, "GL buffer is not as expected.");
        }
      }
      glUnmapNamedBuffer(tex);
    }
    glDeleteTextures(1, &tex);
  }  // end texture 2d

  // texture 3d
  for (int i = 0; targets_3d[i] != 0; i++) {
    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(targets_3d[i], tex);
    glTexImage3D(targets_3d[i], 0, GL_RGB, 2, 2, 1, 0, GL_RGB, GL_FLOAT,
                 pixels);

    {
      queue queue;
      queue.submit([&](handler& cgh) {
        auto img_acc =
            img.template get_access<float, cl::sycl::access::mode::read_write>(
                cgh);
        auto my_range = nd_range<2>(range<2>(4, 3), range<2>(1, 1));
        auto my_kernel =
            ([=](item<1> item) { img_acc[item.get_global(0)] = golden; });
        cgh.parallel_for<class TEST_NAME>(my_range, my_kernel);
      });

      auto img_acc_host =
          img.template get_access<float, cl::sycl::access::mode::read,
                                  access::host_image>();
      for (int i = 0; i < 12, i++) {
        if (img_acc_host[i] != golden) {
          FAIL(log, "Value not updated.");
        }
      }

      void* gl_ptr = glMapNamedBuffer(tex, GL_READ_ONLY);
      float* gl_ptr_float = static_cast<float>(gl_ptr);
      for (int i = 0; i < 12, i++) {
        if (gl_ptr_float[i] != golden) {
          FAIL(log, "GL buffer is not as expected.");
        }
      }
      glUnmapNamedBuffer(tex);

      queue.wait_and_throw();
    }
    glDeleteTextures(1, &tex);
  }  // end texture 3d
}

/** Test for the SYCL buffer OpenGL interoperation interface
 */
class TEST_NAME : public sycl_cts::util::test_base_opencl {
 public:
  /** return information about this test
   */
  virtual void get_info(test_base::info& out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute this test
   */
  virtual void run(util::logger& log) override {
    try {
      gl_util::cl_gl_context cl_gl_ctx;
      // set up gl_framework
      if (!gl_util::gl_set_up()) {
        FAIL(log, "gl_set_up failed.");
      }

      if (gl_util::init_gl_context(&cl_gl_ctx)) {
        // creat sycl context of cl_context that contains gl info.
        {
          context sycl_ctx(cl_gl_ctx.get_context());
          perform(log, sycl_ctx);
        }

        // create sycl context of device_selector & cl_context_properties
        {
          cts_selector sel;
          context sycl_ctx(sel, cl_gl_ctx.get_properties());
          perform(log, sycl_ctx);
        }

        // create sycl context of device & cl_context_properties
        {
          device dev;
          context sycl_ctx(dev, cl_gl_ctx.get_properties());
          perform(log, sycl_ctx);
        }

        // create sycl context of device & cl_context_properties
        {
          device dev;
          device dev_ref(dev);
          context sycl_ctx(dev_ref, cl_gl_ctx.get_properties());
          perform(log, sycl_ctx);
        }

        // create sycl context of platform & cl_context_properties
        {
          platform plat;
          context sycl_ctx(plat, cl_gl_ctx.get_properties());
          perform(log, sycl_ctx);
        }

        // create of vector_class<device>
        {
          VECTOR_CLASS<device> dev_vec;
          device a;
          device b;
          dev_vec.push_back(a);
          dev_vec.push_back(b);

          context sycl_ctx(dev_vec, cl_gl_ctx.get_properties());
          perform(log, sycl_ctx);
        }

        function_class<void(cl::sycl::exception_list)> fn =
            [&](exception_list l) {
              if (l.size() > 1)
                FAIL(log, "Exception thrown during execution of kernel");
            };

        // device_selector, cl_context_properties & async_handler
        {
          cts_selector sel;
          context sycl_ctx(sel, cl_gl_ctx.get_properties(), fn);
          perform(log, sycl_ctx);
        }

        // create sycl context of device, cl_context_properties & async_handler
        {
          device dev;
          context sycl_ctx(dev, cl_gl_ctx.get_properties(), fn);
          perform(log, sycl_ctx);
        }

        // create sycl context of platform, cl_context_properties, &
        // async_handler
        {
          platform plat;
          context sycl_ctx(plat, cl_gl_ctx.get_properties(), fn);
          perform(log, sycl_ctx);
        }

        // create of vector_class<device> & async_handler
        {
          VECTOR_CLASS<device> dev_vec;
          device a;
          device b;
          dev_vec.push_back(a);
          dev_vec.push_back(b);

          context sycl_ctx(dev_vec, cl_gl_ctx.get_properties(), fn);
          perform(log, sycl_ctx);
        }
      } else {
        FAIL(log, "cl_context creation failed.");
      }
    } catch (cl::sycl::exception e) {
      log_exception(log, e);
      FAIL(log, "sycl exception caught");
    }
  }
};

// register this test with the test_collection
util::test_proxy<TEST_NAME> proxy;

} /* namespace opencl_interop_buffer__ */
