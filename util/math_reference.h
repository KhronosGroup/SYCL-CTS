/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2014 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#pragma once

namespace reference
{
/* two argument relational reference */
int isequal( float x, float y );
int isnotequal( float x, float y );
int isgreater( float x, float y );
int isgreaterequal( float x, float y );
int isless( float x, float y );
int islessequal( float x, float y );
int islessgreater( float x, float y );
int isordered( float x, float y );
int isunordered( float x, float y );
};
