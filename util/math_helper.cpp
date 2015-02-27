/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2015 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "math_helper.h"

namespace sycl_cts
{
/** math utility functions
 */
namespace math
{
using namespace cl::sycl;

/* cast an integer to a float */
float int_to_float( uint32_t x )
{
    static_assert( sizeof( x ) == sizeof( float ), "incompatible type sizes" );
    float f;
    memcpy( &f, &x, sizeof( x ) );
    return f;
}

void fill( float &e, float v )
{
    e = v;
}

void fill( float2 &e, float v )
{
    e[0] = v;
    e[1] = v;
}

void fill( float3 &e, float v )
{
    e[0] = v;
    e[1] = v;
    e[2] = v;
}

void fill( float4 &e, float v )
{
    e[0] = v;
    e[1] = v;
    e[2] = v;
    e[3] = v;
}

void fill( float8 &e, float v )
{
    for ( int i = 0; i < 8; i++ )
        e[i] = v;
}

void fill( float16 &e, float v )
{
    for ( int i = 0; i < 16; i++ )
        e[i] = v;
}

/* return number of elements in a type */
int numElements( const float & )
{
    return 1;
}
int numElements( const float2 & )
{
    return 2;
}
int numElements( const float3 & )
{
    return 3;
}
int numElements( const float4 & )
{
    return 4;
}
int numElements( const float8 & )
{
    return 8;
}
int numElements( const float16 & )
{
    return 16;
}

/* return number of elements in a type */
int numElements( const int & )
{
    return 1;
}
int numElements( const int2 & )
{
    return 2;
}
int numElements( const int3 & )
{
    return 3;
}
int numElements( const int4 & )
{
    return 4;
}
int numElements( const int8 & )
{
    return 8;
}
int numElements( const int16 & )
{
    return 16;
}

/* extract an individual elements of a float type */
float getElement( const float &f, int ix )
{
    return f;
}
float getElement( const float2 &f, int ix )
{
    return f[ix];
}
float getElement( const float3 &f, int ix )
{
    return f[ix];
}
float getElement( const float4 &f, int ix )
{
    return f[ix];
}
float getElement( const float8 &f, int ix )
{
    return f[ix];
}
float getElement( const float16 &f, int ix )
{
    return f[ix];
}

/* extract individual elements of an integer type */
int getElement( const int &f, int ix )
{
    return f;
}
int getElement( const int2 &f, int ix )
{
    return f[ix];
}
int getElement( const int3 &f, int ix )
{
    return f[ix];
}
int getElement( const int4 &f, int ix )
{
    return f[ix];
}
int getElement( const int8 &f, int ix )
{
    return f[ix];
}
int getElement( const int16 &f, int ix )
{
    return f[ix];
}

/* create random floats with full integer range */
void rand( MTdata &rng, float *buf, int num )
{
    for ( int i = 0; i < num; i++ )
        buf[i] = (float)int32_t( genrand_int32( rng ) );
}

void rand( MTdata &rng, float2 *buf, int num )
{
    const int nDim = int( sizeof( float2 ) / sizeof( float ) );
    rand( rng, (float *)buf, num * nDim );
}

void rand( MTdata &rng, float3 *buf, int num )
{
    const int nDim = int( sizeof( float3 ) / sizeof( float ) );
    rand( rng, (float *)buf, num * nDim );
}

void rand( MTdata &rng, float4 *buf, int num )
{
    const int nDim = int( sizeof( float4 ) / sizeof( float ) );
    rand( rng, (float *)buf, num * nDim );
}

void rand( MTdata &rng, float8 *buf, int num )
{
    const int nDim = int( sizeof( float8 ) / sizeof( float ) );
    rand( rng, (float *)buf, num * nDim );
}

void rand( MTdata &rng, float16 *buf, int num )
{
    const int nDim = int( sizeof( float16 ) / sizeof( float ) );
    rand( rng, (float *)buf, num * nDim );
}

/* generate a stream of random integer data */
void rand( MTdata &rng, uint8_t *buf, int size )
{
    uint32_t r = 0;
    for ( int i = 0; i < size; i++ )
    {
        if ( (i % 4) == 0 )
            r = genrand_int32( rng );
        buf[i] = r & 0xff;
        r >>= 8;
    }
}

} /* namespace math     */
} /* namespace sycl_cts */
