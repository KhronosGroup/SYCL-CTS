/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2015 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#pragma once

#if defined( _MSC_VER )
#  pragma warning( push )
#  pragma warning( disable : 4267 )
#  pragma warning( disable : 4201 )
#  pragma warning( disable : 4189 )
#  include <cl/sycl.hpp>
#  pragma warning( pop )
#else
#  include <CL/sycl.hpp>
#endif
