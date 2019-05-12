
/******************************************************************
 //
 //  OpenCL Conformance Tests
 // 
 //  Copyright:	(c) 2008-2013 by Apple Inc. All Rights Reserved.
 //
 ******************************************************************/

#ifndef __ROUNDING_MODE_H__
#define __ROUNDING_MODE_H__

#include "compat.h"

typedef enum
{
    kDefaultRoundingMode = 0,
    kRoundToNearestEven,
    kRoundUp,
    kRoundDown,
    kRoundTowardZero,

    kRoundingModeCount
}RoundingMode;

typedef enum
{
    kuchar = 0,
    kchar = 1,
    kushort = 2,
    kshort = 3,
    kuint = 4,
    kint = 5,
    kfloat = 6,
    kdouble = 7,
    kulong = 8,
    klong = 9,
    
    //This goes last
    kTypeCount
}Type;

#ifdef __cplusplus
extern "C" {
#endif

extern RoundingMode set_round( RoundingMode r, Type outType );
extern RoundingMode get_round( void );
extern void *FlushToZero( void );
extern void UnFlushToZero( void *p);

#ifdef __cplusplus
}
#endif



#endif /* __ROUNDING_MODE_H__ */
