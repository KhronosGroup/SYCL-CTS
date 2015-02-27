/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2015 by Codeplay Software LTD. All Rights Reserved.
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
uint8_t  abs( const uint8_t  x ) { return x; }
uint16_t abs( const uint16_t x ) { return x; }
uint32_t abs( const uint32_t x ) { return x; }
uint64_t abs( const uint64_t x ) { return x; }
int8_t   abs( const int8_t   x ) { return x < 0 ? -x : x; }
int16_t  abs( const int16_t  x ) { return x < 0 ? -x : x; }
int32_t  abs( const int32_t  x ) { return x < 0 ? -x : x; }
int64_t  abs( const int64_t  x ) { return x < 0 ? -x : x; }

/* ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ABS_DIFF
 * Returns |x-y| without modulo overflow
 */
uint8_t abs_diff( const uint8_t a, const uint8_t b )
{
    decltype(a) h = ( a >  b ) ? a : b;
    decltype(a) l = ( a <= b ) ? a : b;
    return h - l;
}

uint16_t abs_diff( const uint16_t a, const uint16_t b )
{
    decltype(a) h = ( a >  b ) ? a : b;
    decltype(a) l = ( a <= b ) ? a : b;
    return h - l;
}

uint32_t abs_diff( const uint32_t a, const uint32_t b )
{
    decltype(a) h = ( a >  b ) ? a : b;
    decltype(a) l = ( a <= b ) ? a : b;
    return h - l;
}

uint64_t abs_diff( const uint64_t a, const uint64_t b )
{
    decltype(a) h = ( a >  b ) ? a : b;
    decltype(a) l = ( a <= b ) ? a : b;
    return h - l;
}

int8_t abs_diff( const int8_t   a, const int8_t   b )
{
    decltype(a) h = ( a >  b ) ? a : b;
    decltype(a) l = ( a <= b ) ? a : b;
    return h - l;
}

int16_t abs_diff( const int16_t  a, const int16_t  b )
{
    decltype(a) h = ( a >  b ) ? a : b;
    decltype(a) l = ( a <= b ) ? a : b;
    return h - l;
}

int32_t abs_diff( const int32_t  a, const int32_t  b )
{
    decltype(a) h = ( a >  b ) ? a : b;
    decltype(a) l = ( a <= b ) ? a : b;
    return h - l;
}

int64_t abs_diff( const int64_t  a, const int64_t  b )
{
    decltype(a) h = ( a >  b ) ? a : b;
    decltype(a) l = ( a <= b ) ? a : b;
    return h - l;
}

/* ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ADD_SAT
 * Returns x+y and saturates the result
 */
uint8_t add_sat( const uint8_t a, const uint8_t b )
{
    uint32_t r = uint32_t( a ) + uint32_t( b );
    r = MIN( r, max_uint8_t );
    r = MAX( r, 0 );
    return uint8_t( r );
}

uint16_t add_sat( const uint16_t a, const uint16_t b )
{
    uint32_t r = uint32_t( a ) + uint32_t( b );
    r = MIN( r, max_uint16_t );
    r = MAX( r, 0 );
    return uint16_t( r );
}

uint32_t add_sat( const uint32_t a, const uint32_t b )
{
    uint32_t r = a + b;
    if ( r < a ) return max_uint32_t;
    else         return r;
}

uint64_t add_sat( const uint64_t a, const uint64_t b )
{
    uint64_t r = a + b;
    if ( r < a ) return max_uint64_t;
    else         return r;
}

int8_t add_sat( const int8_t a, const int8_t b )
{
    int32_t r = int32_t( a ) + int32_t( b );
    r = MIN( r, max_int8_t );
    r = MAX( r, min_int8_t );
    return int8_t( r );
}

int16_t add_sat( const int16_t a, const int16_t b )
{
    int32_t r = int32_t( a ) + int32_t( b );
    r = MIN( r, max_int16_t );
    r = MAX( r, min_int16_t );
    return int16_t( r );
}

int32_t add_sat( const int32_t a, const int32_t b )
{
    int32_t r = int32_t( uint32_t( a ) + uint32_t( b ) );
    if ( b > 0 ) { if ( r < a ) return max_int32_t; }
    else         { if ( r > a ) return min_int32_t; }
    return r;
}

int64_t add_sat( const int64_t a, const int64_t b )
{
    int64_t r = int64_t( uint64_t( a ) + uint64_t( b ) );
    if ( b > 0 ) { if ( r < a ) return max_int64_t; }
    else         { if ( r > a ) return min_int64_t; }
    return r;
}

/* ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- HADD
 * Returns (x + y) >> 1. The intermediate sum does not modulo overflow.
 */
uint8_t hadd( const uint8_t a, const uint8_t b )
{
    return (a >> 1) + (b >> 1) + ( ( a&b ) & 0x1 );
}

uint16_t hadd( const uint16_t a, const uint16_t b )
{
    return (a >> 1) + (b >> 1) + ( ( a&b ) & 0x1 );
}

uint32_t hadd( const uint32_t a, const uint32_t b )
{
    return (a >> 1) + (b >> 1) + ( ( a&b ) & 0x1 );
}

uint64_t hadd( const uint64_t a, const uint64_t b )
{
    return (a >> 1ull) + (b >> 1ull) + ( ( a&b ) & 0x1ull );
}

int8_t hadd( const int8_t a, const int8_t b )
{
    return (a >> 1) + (b >> 1) + (a & b & 1);
}

int16_t hadd( const int16_t a, const int16_t b )
{
    return (a >> 1) + (b >> 1) + (a & b & 1);
}

int32_t hadd( const int32_t a, const int32_t b )
{
    return (a >> 1) + (b >> 1) + (a & b & 1);
}

int64_t hadd( const int64_t a, const int64_t b )
{
    return (a >> 1) + (b >> 1) + (a & b & 1);
}

/* ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- RHADD
 * Returns (x + y + 1) >> 1. The intermediate sum does not modulo overflow.
 */
uint8_t rhadd( const uint8_t a, const uint8_t b )
{
    return (a >> 1) + (b >> 1) + ((a & 1) | (b & 1));
}

uint16_t rhadd( const uint16_t a, const uint16_t b )
{
    return (a >> 1) + (b >> 1) + ((a & 1) | (b & 1));
}

uint32_t rhadd( const uint32_t a, const uint32_t b )
{
    return (a >> 1) + (b >> 1) + ((a & 1) | (b & 1));
}

uint64_t rhadd( const uint64_t a, const uint64_t b )
{
    return (a >> 1) + (b >> 1) + ((a & 1) | (b & 1));
}

int8_t rhadd( const int8_t a, const int8_t b )
{
    return (a >> 1) + (b >> 1) + ((a & 1) | (b & 1));
}

int16_t rhadd( const int16_t a, const int16_t b )
{
    return (a >> 1) + (b >> 1) + ((a & 1) | (b & 1));
}

int32_t rhadd( const int32_t a, const int32_t b )
{
    return (a >> 1) + (b >> 1) + ((a & 1) | (b & 1));
}

int64_t rhadd( const int64_t a, const int64_t b )
{
    return (a >> 1) + (b >> 1) + ((a & 1) | (b & 1));
}

/* ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- CLAMP
 * 
 */
template <typename T>
T clamp_t( T v, T minv, T maxv )
{
    return ( v < minv ) ? minv : ( ( v > maxv ) ? maxv : v );
}

uint8_t  clamp( const uint8_t  a, const uint8_t  b, const uint8_t  c ) { return clamp_t( a, b, c ); }
uint16_t clamp( const uint16_t a, const uint16_t b, const uint16_t c ) { return clamp_t( a, b, c ); }
uint32_t clamp( const uint32_t a, const uint32_t b, const uint32_t c ) { return clamp_t( a, b, c ); }
uint64_t clamp( const uint64_t a, const uint64_t b, const uint64_t c ) { return clamp_t( a, b, c ); }
int8_t   clamp( const int8_t   a, const int8_t   b, const int8_t   c ) { return clamp_t( a, b, c ); }
int16_t  clamp( const int16_t  a, const int16_t  b, const int16_t  c ) { return clamp_t( a, b, c ); }
int32_t  clamp( const int32_t  a, const int32_t  b, const int32_t  c ) { return clamp_t( a, b, c ); }
int64_t  clamp( const int64_t  a, const int64_t  b, const int64_t  c ) { return clamp_t( a, b, c ); }
double   clamp( const double   a, const double   b, const double   c ) { return clamp_t( a, b, c ); }
float    clamp( const float    a, const float    b, const float    c ) { return clamp_t( a, b, c ); }

/* ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- CLZ
 * Returns the number of leading zero bits
 */
uint8_t clz( const uint8_t x )
{
    int lz = 0;
    for ( int i = 0; i < num_bits( x ); i++ )
        if ( x & (1<<i) ) lz = 0;
        else              lz++;
    return uint8_t( lz );
}

uint16_t clz( const uint16_t x )
{
    int lz = 0;
    for ( int i = 0; i < num_bits( x ); i++ )
        if ( x & (1<<i) ) lz = 0;
        else              lz++;
    return uint16_t( lz );
}

uint32_t clz( const uint32_t x )
{
    int lz = 0;
    for ( int i = 0; i < num_bits( x ); i++ )
        if ( x & (1<<i) ) lz = 0;
        else              lz++;
    return uint32_t( lz );
}

uint64_t clz( const uint64_t x )
{
    int lz = 0;
    for ( int i = 0; i < num_bits( x ); i++ )
        if ( x & (1ull<<i) ) lz = 0;
        else                 lz++;
    return uint64_t( lz );
}

int8_t clz( const int8_t x )
{
    int lz = 0;
    for ( int i = 0; i < num_bits( x ); i++ )
        if ( x & (1<<i) ) lz = 0;
        else              lz++;
    return int8_t( lz );
}

int16_t clz( const int16_t x )
{
    int lz = 0;
    for ( int i = 0; i < num_bits( x ); i++ )
        if ( x & (1<<i) ) lz = 0;
        else              lz++;
    return int16_t( lz );
}

int32_t clz( const int32_t x )
{
    int lz = 0;
    for ( int i = 0; i < num_bits( x ); i++ )
        if ( x & (1<<i) ) lz = 0;
        else              lz++;
    return int32_t( lz );
}

int64_t clz( const int64_t x )
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
    return mul_hi( x, y ) + z;
}

uint16_t mad_hi( uint16_t x, uint16_t y, uint16_t z )
{
    return mul_hi( x, y ) + z;
}

uint32_t mad_hi( uint32_t x, uint32_t y, uint32_t z )
{
    return mul_hi( x, y ) + z;
}

/* ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- MAD_SAT
 * 
 */
uint8_t mad_sat( uint8_t x, uint8_t y, uint8_t z )
{
    uint32_t a = uint32_t( x ) * uint32_t( y ) + uint32_t( z );
    return uint8_t( ( a > 0xffu ) ? 0xffu : a );
}

uint16_t mad_sat( uint16_t x, uint16_t y, uint16_t z )
{
    uint32_t a = uint32_t( x ) * uint32_t( y ) + uint32_t( z );
    return uint8_t( ( a > 0xffffu ) ? 0xffffu : a );
}

/* ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- MAX
 * Returns a if a > b otherwise b
 */
uint8_t  max( const uint8_t  a, const uint8_t  b ) { return ( a > b ) ? a : b; }
uint16_t max( const uint16_t a, const uint16_t b ) { return ( a > b ) ? a : b; }
uint32_t max( const uint32_t a, const uint32_t b ) { return ( a > b ) ? a : b; }
uint64_t max( const uint64_t a, const uint64_t b ) { return ( a > b ) ? a : b; }
int8_t   max( const int8_t   a, const int8_t   b ) { return ( a > b ) ? a : b; }
int16_t  max( const int16_t  a, const int16_t  b ) { return ( a > b ) ? a : b; }
int32_t  max( const int32_t  a, const int32_t  b ) { return ( a > b ) ? a : b; }
int64_t  max( const int64_t  a, const int64_t  b ) { return ( a > b ) ? a : b; }

/* ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- MIN
 * Returns a if a < b otherwise b
 */
uint8_t  min( const uint8_t  a, const uint8_t  b ) { return ( a < b ) ? a : b; }
uint16_t min( const uint16_t a, const uint16_t b ) { return ( a < b ) ? a : b; }
uint32_t min( const uint32_t a, const uint32_t b ) { return ( a < b ) ? a : b; }
uint64_t min( const uint64_t a, const uint64_t b ) { return ( a < b ) ? a : b; }
int8_t   min( const int8_t   a, const int8_t   b ) { return ( a < b ) ? a : b; }
int16_t  min( const int16_t  a, const int16_t  b ) { return ( a < b ) ? a : b; }
int32_t  min( const int32_t  a, const int32_t  b ) { return ( a < b ) ? a : b; }
int64_t  min( const int64_t  a, const int64_t  b ) { return ( a < b ) ? a : b; }

/* ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- MUL_HI
 * 
 */
uint8_t mul_hi( uint8_t x, uint8_t y )
{
    return uint8_t( ((uint16_t( x ) * uint16_t( y )) >> 8u) & 0xffu );
}

uint16_t mul_hi( uint16_t x, uint16_t y )
{
    return uint16_t( ((uint32_t( x ) * uint32_t( y )) >> 16u) & 0xffffu );
}

uint32_t mul_hi( uint32_t x, uint32_t y )
{
    return uint32_t( ((uint64_t( x ) * uint64_t( y )) >> 32u) & 0xffffffffu );
}

uint64_t mul_hi( uint64_t x, uint64_t y )
{
    // All shifts are half the size of uint64_t in bits
    size_t shft = sizeof( uint64_t ) * 4;

    // hi and lo are the upper and lower parts of the result
    // p, q, r and s are the masked and shifted parts of a
    // b, splitting a and b each into two Ts
    // cross 1 and 2 are the crosswise terms of the multiplication
    uint64_t hi, lo, p, q, r, s, cross1, cross2;

    // The mask used to get the lower half of a uint64_t
    uint64_t mask = -1;
    mask >>= shft;

    // Split a and b in two - upper halves in p and q, lower
    // halves in r and s.
    p = x >> shft;
    q = y >> shft;
    r = x & mask;
    s = y & mask;

    lo = r * s;
    hi = p * q;
    cross1 = (p * s);
    cross2 = (q * r);

    lo >>= shft;
    lo += (cross1 & mask) + (cross2 & mask);
    lo >>= shft;
    hi += lo + (cross1 >> shft) + (cross2 >> shft);

    return hi;
}

int64_t mul_hi( int64_t a, int64_t b )
{
    // All shifts are half the size of int64_t in bits
    size_t shft = sizeof( int64_t ) * 4;

    // hi and lo are the upper and lower parts of the result
    // p, q, r and s are the masked and shifted parts of a
    // b, splitting a and b each into two Ts
    // cross 1 and 2 are the crosswise terms of the multiplication
    uint64_t hi, lo, p, q, r, s, cross1, cross2;

    // The mask used to get the lower half of a int64_t
    uint64_t mask = -1;
    mask >>= shft;

    size_t msb = sizeof( int64_t ) * 8 - 1;

    // a and b rendered positive
    auto a_pos = (a & (1ull << msb)) ? (~a + 1) : a;
    auto b_pos = (b & (1ull << msb)) ? (~b + 1) : b;

    p = static_cast<uint64_t>(a_pos) >> shft;
    q = static_cast<uint64_t>(b_pos) >> shft;
    r = static_cast<uint64_t>(a_pos) & mask;
    s = static_cast<uint64_t>(b_pos) & mask;

    lo = r * s;
    hi = p * q;
    cross1 = (p * s);
    cross2 = (q * r);

    lo >>= shft;
    lo += (cross1 & mask) + (cross2 & mask);
    lo >>= shft;
    hi += lo + (cross1 >> shft) + (cross2 >> shft);

    return (a >> msb) ^ (b >> msb) ?
        static_cast<int64_t>(~hi) :
        hi;
}

/* ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ROTATE
 * Rotate integer right by i bits.  Bits shifted off the left
 * left side are shifted back in from the right.
 */
uint8_t rotate( const uint8_t v, const uint8_t i )
{
    int32_t nBits = num_bits( v ) - int32_t( i );
    return uint8_t( ( v << i ) | ( ( v >> nBits ) ) );
}

uint16_t rotate( const uint16_t v, const uint16_t i )
{
    int32_t nBits = num_bits( v ) - int32_t( i );
    return uint16_t( (v << i) | ((v >> nBits)) );
}

uint32_t rotate( const uint32_t v, const uint32_t i )
{
    int32_t nBits = num_bits( v ) - int32_t( i );
    return uint32_t( (v << i) | ((v >> nBits)) );
}

uint64_t rotate( const uint64_t v, const uint64_t i )
{
    int32_t nBits = num_bits( v ) - int32_t( i );
    return uint64_t( (v << i) | ((v >> nBits)) );
}

int8_t rotate( const int8_t v, const int8_t i )
{
    int8_t mask = int8_t( ( 1u << i ) - 1u );
    int32_t nBits = num_bits( v ) - int32_t( i );
    return int8_t( (v << i) | ((v >> nBits) & mask) );
}

int16_t rotate( const int16_t v, const int16_t i )
{
    int16_t mask = int16_t( ( 1u << i ) - 1u );
    int32_t nBits = num_bits( v ) - int32_t( i );
    return int16_t( (v << i) | ((v >> nBits) & mask) );
}

int32_t rotate( const int32_t v, const int32_t i )
{
    int32_t mask = int32_t( ( 1u << i ) - 1u );
    int32_t nBits = num_bits( v ) - int32_t( i );
    return int32_t( (v << i) | ((v >> nBits) & mask) );
}

int64_t rotate( const int64_t v, const int64_t i )
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
    return x <= y ? 0 : x - y;
}

uint16_t sub_sat( uint16_t x, uint16_t y )
{
    return x <= y ? 0 : x - y;
}

uint32_t sub_sat( uint32_t x, uint32_t y )
{
    return x <= y ? 0 : x - y;
}

uint64_t sub_sat( uint64_t x, uint64_t y )
{
    return x <= y ? 0 : x - y;
}

int8_t sub_sat( int8_t x, int8_t y )
{
    // Min not hex constant because of MSVC warning
    const int8_t max_val = 0x7F;
    const int8_t min_val = -128;
    if ( x > 0 )
    {
        if ( y > 0 )
        {
            return x - y;
        }
        else // x > 0, y <= 0
        {
            return (x - max_val) > y ? max_val : x - y;
        }
    }
    else // x <= 0
    {
        if ( y > 0 )
        {
            return (x - min_val) < y ? min_val : x - y;
        }
        else // x <= 0, y <= 0
        {
            return x - y;
        }
    }
}

int16_t sub_sat( int16_t x, int16_t y )
{
    // Min not hex constant because of MSVC warning
    const int16_t max_val = 0x7FFF;
    const int16_t min_val = -32768;
    if ( x > 0 )
    {
        if ( y > 0 )
        {
            return x - y;
        }
        else // x > 0, y <= 0
        {
            return (x - max_val) > y ? max_val : x - y;
        }
    }
    else // x <= 0
    {
        if ( y > 0 )
        {
            return (x - min_val) < y ? min_val : x - y;
        }
        else // x <= 0, y <= 0
        {
            return x - y;
        }
    }
}

int32_t sub_sat( int32_t x, int32_t y )
{
    const int32_t max_val = 0x7FFFFFFF;
    const int32_t min_val = 0x80000000;
    if ( x > 0 )
    {
        if ( y > 0 )
        {
            return x - y;
        }
        else // x > 0, y <= 0
        {
            return (x - max_val) > y ? max_val : x - y;
        }
    }
    else // x <= 0
    {
        if ( y > 0 )
        {
            return (x - min_val) < y ? min_val : x - y;
        }
        else // x <= 0, y <= 0
        {
            return x - y;
        }
    }
}

int64_t sub_sat( int64_t x, int64_t y )
{
    const int64_t max_val = 0x7FFFFFFFFFFFFFFF;
    const int64_t min_val = 0x8000000000000000;
    if ( x > 0 )
    {
        if ( y > 0 )
        {
            return x - y;
        }
        else // x > 0, y <= 0
        {
            return (x - max_val) > y ? max_val : x - y;
        }
    }
    else // x <= 0
    {
        if ( y > 0 )
        {
            return (x - min_val) < y ? min_val : x - y;
        }
        else // x <= 0, y <= 0
        {
            return x - y;
        }
    }
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
uint8_t popcount( const uint8_t  x )
{
    int lz = 0;
    for ( int i = 0; i < num_bits( x ); i++ )
        if ( x & (1<<i) ) lz++;
    return lz;
}

uint16_t popcount( const uint16_t x )
{
    int lz = 0;
    for ( int i = 0; i < num_bits( x ); i++ )
        if ( x & (1<<i) ) lz++;
    return lz;
}

uint32_t popcount( const uint32_t x )
{
    int lz = 0;
    for ( int i = 0; i < num_bits( x ); i++ )
        if ( x & (1<<i) ) lz++;
    return lz;
}

uint64_t popcount( const uint64_t x )
{
    int lz = 0;
    for ( int i = 0; i < num_bits( x ); i++ )
        if ( x & (1ull<<i) ) lz++;
    return lz;
}

int8_t popcount( const int8_t x )
{
    int lz = 0;
    for ( int i = 0; i < num_bits( x ); i++ )
        if ( x & (1<<i) ) lz++;
    return lz;
}

int16_t popcount( const int16_t x )
{
    int lz = 0;
    for ( int i = 0; i < num_bits( x ); i++ )
        if ( x & (1<<i) ) lz++;
    return lz;
}

int32_t popcount( const int32_t x )
{
    int lz = 0;
    for ( int i = 0; i < num_bits( x ); i++ )
        if ( x & (1<<i) ) lz++;
    return lz;
}

int64_t popcount( const int64_t x )
{
    int lz = 0;
    for ( int i = 0; i < num_bits( x ); i++ )
        if ( x & (1ull<<i) ) lz++;
    return lz;
}

/* ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- MAD24
 * 
 */
template <class T, int length>
cl::sycl::vec<T,length> mad24_tmp( cl::sycl::vec<T,length> x, cl::sycl::vec<T,length> y, cl::sycl::vec<T,length> z )
{
    cl::sycl::vec<T, length> t;
    for ( int i = 0; i < length; i++ )
        t.m_data[i] = T( int64_t( x.m_data[i] ) * int64_t( y.m_data[i] ) + int64_t( z.m_data[i] ) );
    return t;
}

int mad24( int x, int y, int z ) { return int( int64_t( x ) * int64_t( y ) + int64_t( z ) ); }
uint32_t mad24( uint32_t x, uint32_t y, uint32_t z ) { return uint32_t( uint64_t( x ) * uint64_t( y ) + uint64_t( z ) ); }

cl::sycl::int2   mad24( cl::sycl::int2   x, cl::sycl::int2   y, cl::sycl::int2   z ) { return mad24_tmp( x, y, z ); }
cl::sycl::int3   mad24( cl::sycl::int3   x, cl::sycl::int3   y, cl::sycl::int3   z ) { return mad24_tmp( x, y, z ); }
cl::sycl::int4   mad24( cl::sycl::int4   x, cl::sycl::int4   y, cl::sycl::int4   z ) { return mad24_tmp( x, y, z ); }
cl::sycl::int8   mad24( cl::sycl::int8   x, cl::sycl::int8   y, cl::sycl::int8   z ) { return mad24_tmp( x, y, z ); }
cl::sycl::int16  mad24( cl::sycl::int16  x, cl::sycl::int16  y, cl::sycl::int16  z ) { return mad24_tmp( x, y, z ); }
cl::sycl::uint2  mad24( cl::sycl::uint2  x, cl::sycl::uint2  y, cl::sycl::uint2  z ) { return mad24_tmp( x, y, z ); }
cl::sycl::uint3  mad24( cl::sycl::uint3  x, cl::sycl::uint3  y, cl::sycl::uint3  z ) { return mad24_tmp( x, y, z ); }
cl::sycl::uint4  mad24( cl::sycl::uint4  x, cl::sycl::uint4  y, cl::sycl::uint4  z ) { return mad24_tmp( x, y, z ); }
cl::sycl::uint8  mad24( cl::sycl::uint8  x, cl::sycl::uint8  y, cl::sycl::uint8  z ) { return mad24_tmp( x, y, z ); }
cl::sycl::uint16 mad24( cl::sycl::uint16 x, cl::sycl::uint16 y, cl::sycl::uint16 z ) { return mad24_tmp( x, y, z ); }

/* ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- MUL24
 * 
 */
template <class T, int length>
cl::sycl::vec<T,length> mul24_tmp( cl::sycl::vec<T,length> x, cl::sycl::vec<T,length> y )
{
    cl::sycl::vec<T, length> t;
    for ( int i = 0; i < length; i++ )
        t[i] = T( int64_t( x[i] ) * int64_t( y[i] ) );
    return t;
}

int32_t          mul24( int32_t          x, int32_t          y ) { return int32_t (  int64_t( x ) *  int64_t( y ) ); }
uint32_t         mul24( uint32_t         x, uint32_t         y ) { return uint32_t( uint64_t( x ) * uint64_t( y ) ); }
cl::sycl::int2   mul24( cl::sycl::int2   x, cl::sycl::int2   y ) { return mul24_tmp( x, y ); }
cl::sycl::int3   mul24( cl::sycl::int3   x, cl::sycl::int3   y ) { return mul24_tmp( x, y ); }
cl::sycl::int4   mul24( cl::sycl::int4   x, cl::sycl::int4   y ) { return mul24_tmp( x, y ); }
cl::sycl::int8   mul24( cl::sycl::int8   x, cl::sycl::int8   y ) { return mul24_tmp( x, y ); }
cl::sycl::int16  mul24( cl::sycl::int16  x, cl::sycl::int16  y ) { return mul24_tmp( x, y ); }
cl::sycl::uint2  mul24( cl::sycl::uint2  x, cl::sycl::uint2  y ) { return mul24_tmp( x, y ); }
cl::sycl::uint3  mul24( cl::sycl::uint3  x, cl::sycl::uint3  y ) { return mul24_tmp( x, y ); }
cl::sycl::uint4  mul24( cl::sycl::uint4  x, cl::sycl::uint4  y ) { return mul24_tmp( x, y ); }
cl::sycl::uint8  mul24( cl::sycl::uint8  x, cl::sycl::uint8  y ) { return mul24_tmp( x, y ); }
cl::sycl::uint16 mul24( cl::sycl::uint16 x, cl::sycl::uint16 y ) { return mul24_tmp( x, y ); }

} /* namespace reference */
