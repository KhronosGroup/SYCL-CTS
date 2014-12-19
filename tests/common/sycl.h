/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2014 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#pragma once

#if defined( _MSC_VER )
# pragma warning( push )
// disable any sycl header warnings here
# include <cl/sycl.hpp>
# pragma warning( pop )
#else
# include <CL/sycl.hpp>
#endif
