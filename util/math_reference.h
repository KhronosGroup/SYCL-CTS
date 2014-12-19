/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2014 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#pragma once

#include "./stl.h"

namespace reference
{
/* two argument relational reference */
int isequal         ( float x, float y );
int isnotequal      ( float x, float y );
int isgreater       ( float x, float y );
int isgreaterequal  ( float x, float y );
int isless          ( float x, float y );
int islessequal     ( float x, float y );
int islessgreater   ( float x, float y );
int isordered       ( float x, float y );
int isunordered     ( float x, float y );

/* one argument relational reference */
int isfinite        ( float x );
int isinf           ( float x );
int isnan           ( float x );
int isnormal        ( float x );
int signbit         ( float x );

/* absolute value */
uint8_t  abs( uint8_t  );
uint16_t abs( uint16_t );
uint32_t abs( uint32_t );
uint64_t abs( uint64_t );
int8_t   abs( int8_t   );
int16_t  abs( int16_t  );
int32_t  abs( int32_t  );
int64_t  abs( int64_t  );

/* asbsolue difference */
uint8_t  abs_diff( uint8_t  a, uint8_t  b );
uint16_t abs_diff( uint16_t a, uint16_t b );
uint32_t abs_diff( uint32_t a, uint32_t b );
uint64_t abs_diff( uint64_t a, uint64_t b );
int8_t   abs_diff( int8_t   a, int8_t   b );
int16_t  abs_diff( int16_t  a, int16_t  b );
int32_t  abs_diff( int32_t  a, int32_t  b );
int64_t  abs_diff( int64_t  a, int64_t  b );

/* add with saturation */
uint8_t  add_sat( uint8_t  a, uint8_t  b );
uint16_t add_sat( uint16_t a, uint16_t b );
uint32_t add_sat( uint32_t a, uint32_t b );
uint64_t add_sat( uint64_t a, uint64_t b );
int8_t   add_sat( int8_t   a, int8_t   b );
int16_t  add_sat( int16_t  a, int16_t  b );
int32_t  add_sat( int32_t  a, int32_t  b );
int64_t  add_sat( int64_t  a, int64_t  b );

/* half add */
uint8_t  hadd( uint8_t  a, uint8_t  b );
uint16_t hadd( uint16_t a, uint16_t b );
uint32_t hadd( uint32_t a, uint32_t b );
uint64_t hadd( uint64_t a, uint64_t b );
int8_t   hadd( int8_t   a, int8_t   b );
int16_t  hadd( int16_t  a, int16_t  b );
int32_t  hadd( int32_t  a, int32_t  b );
int64_t  hadd( int64_t  a, int64_t  b );

/* round up half add */
uint8_t  rhadd( uint8_t  a, uint8_t  b );
uint16_t rhadd( uint16_t a, uint16_t b );
uint32_t rhadd( uint32_t a, uint32_t b );
uint64_t rhadd( uint64_t a, uint64_t b );
int8_t   rhadd( int8_t   a, int8_t   b );
int16_t  rhadd( int16_t  a, int16_t  b );
int32_t  rhadd( int32_t  a, int32_t  b );
int64_t  rhadd( int64_t  a, int64_t  b );

/* clamp */
uint8_t  clamp( uint8_t  a, uint8_t  b, uint8_t c );
uint16_t clamp( uint16_t a, uint16_t b, uint8_t c );
uint32_t clamp( uint32_t a, uint32_t b, uint8_t c );
uint64_t clamp( uint64_t a, uint64_t b, uint8_t c );
int8_t   clamp( int8_t   a, int8_t   b, uint8_t c );
int16_t  clamp( int16_t  a, int16_t  b, uint8_t c );
int32_t  clamp( int32_t  a, int32_t  b, uint8_t c );
int64_t  clamp( int64_t  a, int64_t  b, uint8_t c );

/* count leading zeros */
uint8_t  clz( uint8_t  );
uint16_t clz( uint16_t );
uint32_t clz( uint32_t );
uint64_t clz( uint64_t );
int8_t   clz( int8_t   );
int16_t  clz( int16_t  );
int32_t  clz( int32_t  );
int64_t  clz( int64_t  );

/* multiply add, get high part */
uint8_t  mad_hi( uint8_t  a, uint8_t  b, uint8_t c );
uint16_t mad_hi( uint16_t a, uint16_t b, uint16_t c );
uint32_t mad_hi( uint32_t a, uint32_t b, uint32_t c );
uint64_t mad_hi( uint64_t a, uint64_t b, uint64_t c );
int8_t   mad_hi( int8_t   a, int8_t   b, int8_t c );
int16_t  mad_hi( int16_t  a, int16_t  b, int16_t c );
int32_t  mad_hi( int32_t  a, int32_t  b, int32_t c );
int64_t  mad_hi( int64_t  a, int64_t  b, int64_t c );

/* multiply add saturate */
uint8_t  mad_sat( uint8_t  a, uint8_t  b, uint8_t c );
uint16_t mad_sat( uint16_t a, uint16_t b, uint8_t c );
uint32_t mad_sat( uint32_t a, uint32_t b, uint8_t c );
uint64_t mad_sat( uint64_t a, uint64_t b, uint8_t c );
int8_t   mad_sat( int8_t   a, int8_t   b, uint8_t c );
int16_t  mad_sat( int16_t  a, int16_t  b, uint8_t c );
int32_t  mad_sat( int32_t  a, int32_t  b, uint8_t c );
int64_t  mad_sat( int64_t  a, int64_t  b, uint8_t c );

/* maximum value */
uint8_t  max( uint8_t  a, uint8_t  b );
uint16_t max( uint16_t a, uint16_t b );
uint32_t max( uint32_t a, uint32_t b );
uint64_t max( uint64_t a, uint64_t b );
int8_t   max( int8_t   a, int8_t   b );
int16_t  max( int16_t  a, int16_t  b );
int32_t  max( int32_t  a, int32_t  b );
int64_t  max( int64_t  a, int64_t  b );

/* minimum value */
uint8_t  min( uint8_t  a, uint8_t  b );
uint16_t min( uint16_t a, uint16_t b );
uint32_t min( uint32_t a, uint32_t b );
uint64_t min( uint64_t a, uint64_t b );
int8_t   min( int8_t   a, int8_t   b );
int16_t  min( int16_t  a, int16_t  b );
int32_t  min( int32_t  a, int32_t  b );
int64_t  min( int64_t  a, int64_t  b );

/* multiply and return high part */
uint8_t  mul_hi( uint8_t  a, uint8_t  b );
uint16_t mul_hi( uint16_t a, uint16_t b );
uint32_t mul_hi( uint32_t a, uint32_t b );
uint64_t mul_hi( uint64_t a, uint64_t b );
int8_t   mul_hi( int8_t   a, int8_t   b );
int16_t  mul_hi( int16_t  a, int16_t  b );
int32_t  mul_hi( int32_t  a, int32_t  b );
int64_t  mul_hi( int64_t  a, int64_t  b );

/* bitwise rotate */
uint8_t  rotate( uint8_t  a, uint8_t  b );
uint16_t rotate( uint16_t a, uint16_t b );
uint32_t rotate( uint32_t a, uint32_t b );
uint64_t rotate( uint64_t a, uint64_t b );
int8_t   rotate( int8_t   a, int8_t   b );
int16_t  rotate( int16_t  a, int16_t  b );
int32_t  rotate( int32_t  a, int32_t  b );
int64_t  rotate( int64_t  a, int64_t  b );

/* returns x - y and saturates the result */
uint8_t  sub_sat( uint8_t  a, uint8_t  b );
uint16_t sub_sat( uint16_t a, uint16_t b );
uint32_t sub_sat( uint32_t a, uint32_t b );
uint64_t sub_sat( uint64_t a, uint64_t b );
int8_t   sub_sat( int8_t   a, int8_t   b );
int16_t  sub_sat( int16_t  a, int16_t  b );
int32_t  sub_sat( int32_t  a, int32_t  b );
int64_t  sub_sat( int64_t  a, int64_t  b );

/* return number of non zero bits in x */
uint8_t  popcount( uint8_t  );
uint16_t popcount( uint16_t );
uint32_t popcount( uint32_t );
uint64_t popcount( uint64_t );
int8_t   popcount( int8_t   );
int16_t  popcount( int16_t  );
int32_t  popcount( int32_t  );
int64_t  popcount( int64_t  );

/*  */
uint8_t  mad24( uint8_t  x, uint8_t  y, uint8_t  z );
uint16_t mad24( uint16_t x, uint16_t y, uint16_t z );
uint32_t mad24( uint32_t x, uint32_t y, uint32_t z );
uint64_t mad24( uint64_t x, uint64_t y, uint64_t z );
uint8_t  mad24(  int8_t  x,  int8_t  y,  int8_t  z );
uint16_t mad24(  int16_t x,  int16_t y,  int16_t z );
uint32_t mad24(  int32_t x,  int32_t y,  int32_t z );
uint64_t mad24(  int64_t x,  int64_t y,  int64_t z );

/*  */
uint8_t  mul24( uint8_t  x, uint8_t  y, uint8_t  z );
uint16_t mul24( uint16_t x, uint16_t y, uint16_t z );
uint32_t mul24( uint32_t x, uint32_t y, uint32_t z );
uint64_t mul24( uint64_t x, uint64_t y, uint64_t z );
uint8_t  mul24(  int8_t  x,  int8_t  y,  int8_t  z );
uint16_t mul24(  int16_t x,  int16_t y,  int16_t z );
uint32_t mul24(  int32_t x,  int32_t y,  int32_t z );
uint64_t mul24(  int64_t x,  int64_t y,  int64_t z );

}
