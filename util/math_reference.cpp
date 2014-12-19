/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2014 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "math_reference.h"
#include "stl.h"

namespace
{

template <typename A, typename B>
void type_punn( const A & from, B & to )
{
    static_assert( sizeof( A ) == sizeof( B ), "type punning of incompatible sized types" );
    memcpy( (void*)&to, (void*)&from, sizeof( A ) );
}

template <typename T>
int32_t num_bits( T )
{
    return int32_t( sizeof( T ) * 8u );
}

const uint64_t max_uint64_t = (~0x0ull);
const uint32_t max_uint32_t = 0xffffffff;
const uint16_t max_uint16_t = 0xffff;
const uint8_t  max_uint8_t  = 0xff;

const int64_t  max_int64_t  = ~( 1ull << 63 );
const int32_t  max_int32_t  = 0x7fffffff;
const int32_t  max_int16_t  = 0x7fff;
const int32_t  max_int8_t   = 0x7f;

const int64_t  min_int64_t  = int64_t( 1ull << 63 );
const int32_t  min_int32_t  = int32_t( 0x80000000 );
const int16_t  min_int16_t  = int16_t( 0x8000 );
const int8_t   min_int8_t   = int8_t ( 0x80 );

#define MAX( _a, _b ) ( (_a) > (_b) ? (_a) : (_b) )
#define MIN( _a, _b ) ( (_a) < (_b) ? (_a) : (_b) )

} /* namespace {} */

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

int isfinite( float x )
{
    int i = 0;
    type_punn( x, i );
    return (i & 0x7f800000u) != 0x7f800000u;
}

int isinf( float x )
{
    int i = 0;
    type_punn( x, i );
    return (i & 0x7fffffffu) == 0x7f800000u;
}

int isnan( float x )
{
    int i = 0;
    type_punn( x, i );
    return ((i & 0x7f800000u) == 0x7f800000u) && (i & 0x007fffffu);
}

int isnormal( float x )
{
    return !isnan( x );
}

int signbit( float x )
{
    int i = 0;
    type_punn( x, i );
    return (i & 0x80000000u) != 0;
}

/* ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ABS
 * Returns |x|
 */
uint8_t  abs( uint8_t  x ) { return x; }
uint16_t abs( uint16_t x ) { return x; }
uint32_t abs( uint32_t x ) { return x; }
uint64_t abs( uint64_t x ) { return x; }
int8_t   abs( int8_t   x ) { return x < 0 ? -x : x; }
int16_t  abs( int16_t  x ) { return x < 0 ? -x : x; }
int32_t  abs( int32_t  x ) { return x < 0 ? -x : x; }
int64_t  abs( int64_t  x ) { return x < 0 ? -x : x; }

/* ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ABS_DIFF
 * Returns |x-y| without modulo overflow
 */
uint8_t abs_diff( uint8_t a, uint8_t b )
{
    decltype(a) h = ( a >  b ) ? a : b;
    decltype(a) l = ( a <= b ) ? a : b;
    return h - l;
}

uint16_t abs_diff( uint16_t a, uint16_t b )
{
    decltype(a) h = ( a >  b ) ? a : b;
    decltype(a) l = ( a <= b ) ? a : b;
    return h - l;
}

uint32_t abs_diff( uint32_t a, uint32_t b )
{
    decltype(a) h = ( a >  b ) ? a : b;
    decltype(a) l = ( a <= b ) ? a : b;
    return h - l;
}

uint64_t abs_diff( uint64_t a, uint64_t b )
{
    decltype(a) h = ( a >  b ) ? a : b;
    decltype(a) l = ( a <= b ) ? a : b;
    return h - l;
}

int8_t abs_diff( int8_t   a, int8_t   b )
{
    decltype(a) h = ( a >  b ) ? a : b;
    decltype(a) l = ( a <= b ) ? a : b;
    return h - l;
}

int16_t abs_diff( int16_t  a, int16_t  b )
{
    decltype(a) h = ( a >  b ) ? a : b;
    decltype(a) l = ( a <= b ) ? a : b;
    return h - l;
}

int32_t abs_diff( int32_t  a, int32_t  b )
{
    decltype(a) h = ( a >  b ) ? a : b;
    decltype(a) l = ( a <= b ) ? a : b;
    return h - l;
}

int64_t abs_diff( int64_t  a, int64_t  b )
{
    decltype(a) h = ( a >  b ) ? a : b;
    decltype(a) l = ( a <= b ) ? a : b;
    return h - l;
}

/* ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ADD_SAT
 * Returns x+y and saturates the result
 */
uint8_t add_sat( uint8_t a, uint8_t b )
{
    uint32_t r = uint32_t( a ) + uint32_t( b );
    r = MIN( r, max_uint8_t );
    r = MAX( r, 0 );
    return uint8_t( r );
}

uint16_t add_sat( uint16_t a, uint16_t b )
{
    uint32_t r = uint32_t( a ) + uint32_t( b );
    r = MIN( r, max_uint16_t );
    r = MAX( r, 0 );
    return uint16_t( r );
}

uint32_t add_sat( uint32_t a, uint32_t b )
{
    uint32_t r = a + b;
    if ( r < a ) return max_uint32_t;
    else         return r;
}

uint64_t add_sat( uint64_t a, uint64_t b )
{
    uint64_t r = a + b;
    if ( r < a ) return max_uint64_t;
    else         return r;
}

int8_t add_sat( int8_t a, int8_t b )
{
    int32_t r = int32_t( a ) + int32_t( b );
    r = MIN( r, max_int8_t );
    r = MAX( r, min_int8_t );
    return int8_t( r );
}

int16_t add_sat( int16_t a, int16_t b )
{
    int32_t r = int32_t( a ) + int32_t( b );
    r = MIN( r, max_int16_t );
    r = MAX( r, min_int16_t );
    return int16_t( r );
}

int32_t add_sat( int32_t a, int32_t b )
{
    int32_t r = int32_t( uint32_t( a ) + uint32_t( b ) );
    if ( b > 0 ) { if ( r < a ) return max_int32_t; }
    else         { if ( r > a ) return min_int32_t; }
    return r;
}

int64_t add_sat( int64_t a, int64_t b )
{
    int64_t r = int64_t( uint64_t( a ) + uint64_t( b ) );
    if ( b > 0 ) { if ( r < a ) return max_int64_t; }
    else         { if ( r > a ) return min_int64_t; }
    return r;
}

/* ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- HADD
 * Returns (x + y) >> 1. The intermediate sum does not modulo overflow.
 */
uint8_t hadd( uint8_t a, uint8_t b )
{
    return (a >> 1) + (b >> 1) + ( ( a&b ) & 0x1 );
}

uint16_t hadd( uint16_t a, uint16_t b )
{
    return (a >> 1) + (b >> 1) + ( ( a&b ) & 0x1 );
}

uint32_t hadd( uint32_t a, uint32_t b )
{
    return (a >> 1) + (b >> 1) + ( ( a&b ) & 0x1 );
}

uint64_t hadd( uint64_t a, uint64_t b )
{
    return (a >> 1ull) + (b >> 1ull) + ( ( a&b ) & 0x1ull );
}

int8_t hadd( int8_t a, int8_t b )
{
    return 0; // todo
}

int16_t hadd( int16_t a, int16_t b )
{
    return 0; // todo
}

int32_t hadd( int32_t a, int32_t b )
{
    return 0; // todo
}

int64_t hadd( int64_t a, int64_t b )
{
    return 0; // todo
}

/* ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- RHADD
 * Returns (x + y + 1) >> 1. The intermediate sum does not modulo overflow.
 */
uint8_t rhadd( uint8_t a, uint8_t b )
{
    return (a >> 1) + (b >> 1) + ( ( a&b ) & 0x1 );
}

uint16_t rhadd( uint16_t a, uint16_t b )
{
    return (a >> 1) + (b >> 1) + ( ( a&b ) & 0x1 );
}

uint32_t rhadd( uint32_t a, uint32_t b )
{
    return (a >> 1) + (b >> 1) + ( ( a&b ) & 0x1 );
}

uint64_t rhadd( uint64_t a, uint64_t b )
{
    return (a >> 1ull) + (b >> 1ull) + ( ( a&b ) & 0x1ull );
}

int8_t rhadd( int8_t a, int8_t b )
{
    return 0; // todo
}

int16_t rhadd( int16_t a, int16_t b )
{
    return 0; // todo
}

int32_t rhadd( int32_t a, int32_t b )
{
    return 0; // todo
}

int64_t rrhadd( int64_t a, int64_t b )
{
    return 0; // todo
}

/* ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- CLAMP
 * 
 */
uint8_t  clamp( uint8_t  a, uint8_t  b, uint8_t c )
{
    return 0; // todo
}

uint16_t clamp( uint16_t a, uint16_t b, uint8_t c )
{
    return 0; // todo
}

uint32_t clamp( uint32_t a, uint32_t b, uint8_t c )
{
    return 0; // todo
}

uint64_t clamp( uint64_t a, uint64_t b, uint8_t c )
{
    return 0; // todo
}

int8_t   clamp( int8_t   a, int8_t   b, uint8_t c )
{
    return 0; // todo
}

int16_t  clamp( int16_t  a, int16_t  b, uint8_t c )
{
    return 0; // todo
}

int32_t  clamp( int32_t  a, int32_t  b, uint8_t c )
{
    return 0; // todo
}

int64_t  clamp( int64_t  a, int64_t  b, uint8_t c )
{
    return 0; // todo
}

/* ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- CLZ
 * Returns the number of leading zero bits
 */
uint8_t clz( uint8_t x )
{
    int lz = 0;
    for ( int i = 0; i < num_bits( x ); i++ )
        if ( x & (1<<i) ) lz = 0;
        else              lz++;
    return uint8_t( lz );
}

uint16_t clz( uint16_t x )
{
    int lz = 0;
    for ( int i = 0; i < num_bits( x ); i++ )
        if ( x & (1<<i) ) lz = 0;
        else              lz++;
    return uint16_t( lz );
}

uint32_t clz( uint32_t x )
{
    int lz = 0;
    for ( int i = 0; i < num_bits( x ); i++ )
        if ( x & (1<<i) ) lz = 0;
        else              lz++;
    return uint32_t( lz );
}

uint64_t clz( uint64_t x )
{
    int lz = 0;
    for ( int i = 0; i < num_bits( x ); i++ )
        if ( x & (1ull<<i) ) lz = 0;
        else                 lz++;
    return uint64_t( lz );
}

int8_t clz( int8_t x )
{
    int lz = 0;
    for ( int i = 0; i < num_bits( x ); i++ )
        if ( x & (1<<i) ) lz = 0;
        else              lz++;
    return int8_t( lz );
}

int16_t clz( int16_t x )
{
    int lz = 0;
    for ( int i = 0; i < num_bits( x ); i++ )
        if ( x & (1<<i) ) lz = 0;
        else              lz++;
    return int16_t( lz );
}

int32_t clz( int32_t x )
{
    int lz = 0;
    for ( int i = 0; i < num_bits( x ); i++ )
        if ( x & (1<<i) ) lz = 0;
        else              lz++;
    return int32_t( lz );
}

int64_t clz( int64_t x )
{
    int lz = 0;
    for ( int i = 0; i < num_bits( x ); i++ )
        if ( x & (1ull<<i) ) lz = 0;
        else                 lz++;
    return int64_t( lz );
}

/* ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- MAD_HI
 * 
 */
uint8_t mad_hi( uint8_t x, uint8_t y, uint8_t z )
{
    return 0; // todo
}

uint16_t mad_sat( uint16_t a, uint16_t b, uint8_t c )
{
    return 0; // todo
}

uint32_t mad_sat( uint32_t a, uint32_t b, uint8_t c )
{
    return 0; // todo
}

uint64_t mad_sat( uint64_t a, uint64_t b, uint8_t c )
{
    return 0; // todo
}

int8_t   mad_sat( int8_t   a, int8_t   b, uint8_t c )
{
    return 0; // todo
}

int16_t  mad_sat( int16_t  a, int16_t  b, uint8_t c )
{
    return 0; // todo
}

int32_t  mad_sat( int32_t  a, int32_t  b, uint8_t c )
{
    return 0; // todo
}

int64_t  mad_sat( int64_t  a, int64_t  b, uint8_t c )
{
    return 0; // todo
}

/* ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- MAD_SAT
 * 
 */
uint8_t mad_sat( uint8_t x, uint8_t y, uint8_t c  )
{
    return 0; // todo
}

/* ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- MAX
 * Returns a if a > b otherwise b
 */
uint8_t  max( uint8_t  a, uint8_t  b ) { return ( a > b ) ? a : b; }
uint16_t max( uint16_t a, uint16_t b ) { return ( a > b ) ? a : b; }
uint32_t max( uint32_t a, uint32_t b ) { return ( a > b ) ? a : b; }
uint64_t max( uint64_t a, uint64_t b ) { return ( a > b ) ? a : b; }
int8_t   max( int8_t   a, int8_t   b ) { return ( a > b ) ? a : b; }
int16_t  max( int16_t  a, int16_t  b ) { return ( a > b ) ? a : b; }
int32_t  max( int32_t  a, int32_t  b ) { return ( a > b ) ? a : b; }
int64_t  max( int64_t  a, int64_t  b ) { return ( a > b ) ? a : b; }

/* ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- MIN
 * Returns a if a < b otherwise b
 */
uint8_t  min( uint8_t  a, uint8_t  b ) { return ( a < b ) ? a : b; }
uint16_t min( uint16_t a, uint16_t b ) { return ( a < b ) ? a : b; }
uint32_t min( uint32_t a, uint32_t b ) { return ( a < b ) ? a : b; }
uint64_t min( uint64_t a, uint64_t b ) { return ( a < b ) ? a : b; }
int8_t   min( int8_t   a, int8_t   b ) { return ( a < b ) ? a : b; }
int16_t  min( int16_t  a, int16_t  b ) { return ( a < b ) ? a : b; }
int32_t  min( int32_t  a, int32_t  b ) { return ( a < b ) ? a : b; }
int64_t  min( int64_t  a, int64_t  b ) { return ( a < b ) ? a : b; }

/* ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- MUL_HI
 * 
 */
uint8_t mul_hi( uint8_t x, uint8_t y )
{
    return 0; // todo
}

/* ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ROTATE
 * Rotate integer right by i bits.  Bits shifted off the left
 * left side are shifted back in from the right.
 */
uint8_t rotate( uint8_t v, uint8_t i )
{
    int32_t nBits = num_bits( v ) - int32_t( i );
    return uint8_t( ( v << i ) | ( ( v >> nBits ) ) );
}

uint16_t rotate( uint16_t v, uint16_t i )
{
    int32_t nBits = num_bits( v ) - int32_t( i );
    return uint16_t( (v << i) | ((v >> nBits)) );
}

uint32_t rotate( uint32_t v, uint32_t i )
{
    int32_t nBits = num_bits( v ) - int32_t( i );
    return uint32_t( (v << i) | ((v >> nBits)) );
}

uint64_t rotate( uint64_t v, uint64_t i )
{
    int32_t nBits = num_bits( v ) - int32_t( i );
    return uint64_t( (v << i) | ((v >> nBits)) );
}

int8_t rotate( int8_t v, int8_t i )
{
    int8_t mask = int8_t( ( 1u << i ) - 1u );
    int32_t nBits = num_bits( v ) - int32_t( i );
    return int8_t( (v << i) | ((v >> nBits) & mask) );
}

int16_t rotate( int16_t v, int16_t i )
{
    int16_t mask = int16_t( ( 1u << i ) - 1u );
    int32_t nBits = num_bits( v ) - int32_t( i );
    return int16_t( (v << i) | ((v >> nBits) & mask) );
}

int32_t rotate( int32_t v, int32_t i )
{
    int32_t mask = int32_t( ( 1u << i ) - 1u );
    int32_t nBits = num_bits( v ) - int32_t( i );
    return int32_t( (v << i) | ((v >> nBits) & mask) );
}

int64_t rotate( int64_t v, int64_t i )
{
    int64_t mask = int64_t( ( 1ull << i ) - 1ull );
    int32_t nBits = num_bits( v ) - int32_t( i );
    return int64_t( (v << i) | ((v >> nBits) & mask) );
}

/* ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- SUB_SAT
 * 
 */
uint8_t sub_sat( uint8_t x, uint8_t y )
{
    return 0; // todo
}

/* ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- UPSAMPLE
 * 
 */
uint16_t upsample( uint8_t h, uint8_t l )
{
    return (uint16_t( h ) << 8) | uint16_t( l );
}

uint32_t upsample( uint16_t h, uint16_t l )
{
    return (uint32_t( h ) << 16) | uint32_t( l );
}

uint64_t upsample( uint32_t h, uint32_t l )
{
    return (uint64_t( h ) << 32) | uint64_t( l );
}

int16_t upsample( int8_t h, uint8_t l )
{
    return (int16_t( h ) << 8) | uint16_t( l );
}

int32_t upsample( int16_t h, uint16_t l )
{
    return (int32_t( h ) << 16) | uint32_t( l );
}

int64_t upsample( int32_t h, uint32_t l )
{
    return (int64_t( h ) << 32) | uint64_t( l );
}

/* ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- POPCOUNT
 * Returns the number of non-zero bits in x
 */
uint8_t popcount( uint8_t  x )
{
    int lz = 0;
    for ( int i = 0; i < num_bits( x ); i++ )
        if ( x & (1<<i) ) lz++;
    return lz;
}

uint16_t popcount( uint16_t x )
{
    int lz = 0;
    for ( int i = 0; i < num_bits( x ); i++ )
        if ( x & (1<<i) ) lz++;
    return lz;
}

uint32_t popcount( uint32_t x )
{
    int lz = 0;
    for ( int i = 0; i < num_bits( x ); i++ )
        if ( x & (1<<i) ) lz++;
    return lz;
}

uint64_t popcount( uint64_t x )
{
    int lz = 0;
    for ( int i = 0; i < num_bits( x ); i++ )
        if ( x & (1ull<<i) ) lz++;
    return lz;
}

int8_t popcount( int8_t x )
{
    int lz = 0;
    for ( int i = 0; i < num_bits( x ); i++ )
        if ( x & (1<<i) ) lz++;
    return lz;
}

int16_t popcount( int16_t x )
{
    int lz = 0;
    for ( int i = 0; i < num_bits( x ); i++ )
        if ( x & (1<<i) ) lz++;
    return lz;
}

int32_t popcount( int32_t x )
{
    int lz = 0;
    for ( int i = 0; i < num_bits( x ); i++ )
        if ( x & (1<<i) ) lz++;
    return lz;
}

int64_t popcount( int64_t x )
{
    int lz = 0;
    for ( int i = 0; i < num_bits( x ); i++ )
        if ( x & (1ull<<i) ) lz++;
    return lz;
}

/* ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- MAD24
 * 
 */
uint8_t mad24( uint8_t x, uint8_t y, uint8_t z )
{
    return 0; // todo
}

uint16_t mad24( uint16_t x, uint16_t y, uint16_t z )
{
    return 0; // todo
}

uint32_t mad24( uint32_t x, uint32_t y, uint32_t z )
{
    return 0; // todo
}

uint64_t mad24( uint64_t x, uint64_t y, uint64_t z )
{
    return 0; // todo
}

uint8_t  mad24(  int8_t  x,  int8_t  y,  int8_t  z )
{
    return 0; // todo
}

uint16_t mad24(  int16_t x,  int16_t y,  int16_t z )
{
    return 0; // todo
}

uint32_t mad24(  int32_t x,  int32_t y,  int32_t z )
{
    return 0; // todo
}

uint64_t mad24(  int64_t x,  int64_t y,  int64_t z )
{
    return 0; // todo
}

/* ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- MUL24
 * 
 */
uint8_t mul24( uint8_t x, uint8_t y, uint8_t z )
{
    return 0; // todo
}

uint16_t mul24( uint16_t x, uint16_t y, uint16_t z )
{
    return 0; // todo
}

uint32_t mul24( uint32_t x, uint32_t y, uint32_t z )
{
    return 0; // todo
}

uint64_t mul24( uint64_t x, uint64_t y, uint64_t z )
{
    return 0; // todo
}

uint8_t  mul24(  int8_t  x,  int8_t  y,  int8_t  z )
{
    return 0; // todo
}

uint16_t mul24(  int16_t x,  int16_t y,  int16_t z )
{
    return 0; // todo
}

uint32_t mul24(  int32_t x,  int32_t y,  int32_t z )
{
    return 0; // todo
}

uint64_t mul24(  int64_t x,  int64_t y,  int64_t z )
{
    return 0; // todo
}

} /* namespace reference */
