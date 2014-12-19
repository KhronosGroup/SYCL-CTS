/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2014 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#pragma once

#include "../tests/common/sycl.h"
#include "./../oclmath/mt19937.h"

namespace sycl_cts
{
/** math utility functions
 */
namespace math
{
using namespace cl::sycl;

/* cast an integer to a float */
float int_to_float( uint32_t x );

void fill( float &e, float v );
void fill( float2 &e, float v );
void fill( float3 &e, float v );
void fill( float4 &e, float v );
void fill( float8 &e, float v );
void fill( float16 &e, float v );

/* return number of elements in a type */
int numElements( const float & );
int numElements( const float2 & );
int numElements( const float3 & );
int numElements( const float4 & );
int numElements( const float8 & );
int numElements( const float16 & );

/* return number of elements in a type */
int numElements( const int & );
int numElements( const int2 & );
int numElements( const int3 & );
int numElements( const int4 & );
int numElements( const int8 & );
int numElements( const int16 & );

/* extract an individual elements */
float getElement( const float &f, int ix );
float getElement( const float2 &f, int ix );
float getElement( const float3 &f, int ix );
float getElement( const float4 &f, int ix );
float getElement( const float8 &f, int ix );
float getElement( const float16 &f, int ix );

/* extract individual elements of an integer type */
int getElement( const int &f, int ix );
int getElement( const int2 &f, int ix );
int getElement( const int3 &f, int ix );
int getElement( const int4 &f, int ix );
int getElement( const int8 &f, int ix );
int getElement( const int16 &f, int ix );

/* create random floats with an integer range [-0x7fffffff to 0x7fffffff]
 */
void rand( MTdata &rng, float *buf, int num );
void rand( MTdata &rng, float2 *buf, int num );
void rand( MTdata &rng, float3 *buf, int num );
void rand( MTdata &rng, float4 *buf, int num );
void rand( MTdata &rng, float8 *buf, int num );
void rand( MTdata &rng, float16 *buf, int num );

/* generate a stream of random integer data
 */
void rand( MTdata &rng, uint8_t *buf, int size );

} /* namespace math     */
} /* namespace sycl_cts */
