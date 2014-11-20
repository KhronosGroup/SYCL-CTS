/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2014 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "math_reference.h"

namespace reference
{
int isequal( float x, float y )
{
    return x == y;
}

int isnotequal( float x, float y )
{
    return x != y;
}

int isgreater( float x, float y )
{
    return x > y;
}

int isgreaterequal( float x, float y )
{
    return x >= y;
}

int isless( float x, float y )
{
    return x < y;
}

int islessequal( float x, float y )
{
    return x <= y;
}

int islessgreater( float x, float y )
{
    return ( x < y ) || ( x > y );
}

int isordered( float x, float y )
{
    return ( x == x ) && ( y == y );
}

int isunordered( float x, float y )
{
    return !( ( x == x ) && ( y == y ) );
}
};
