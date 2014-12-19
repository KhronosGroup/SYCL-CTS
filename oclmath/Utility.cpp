
/******************************************************************
 //
 //  OpenCL Conformance Tests
 // 
 //  Copyright:	(c) 2008-2013 by Apple Inc. All Rights Reserved.
 //
 ******************************************************************/

#include "Utility.h"

#if defined( _MSC_VER )
# pragma warning( push )
# pragma warning( disable : 4756 ) /* overflow in constant arithmetic                                       */
#endif

#if defined(__PPC__)
// Global varaiable used to hold the FPU control register state. The FPSCR register can not
// be used because not all Power implementations retain or observed the NI (non-IEEE 
// mode) bit.
__thread fpu_control_t fpu_control = 0;
#endif

int gIsInRTZMode = 0;
int gCheckTininessBeforeRounding = 0;
int gDeviceILogb0 = 0;
int gDeviceILogbNaN = 0;

void MulD(double *rhi, double *rlo, double u, double v)
{
	const double c = 134217729.0; // 1+2^27
	double up, u1, u2, vp, v1, v2;
	
	up = u*c;
	u1 = (u - up) + up;
	u2 = u - u1;
	
	vp = v*c;
	v1 = (v - vp) + vp;
	v2 = v - v1;
	
	double rh = u*v;
	double rl = (((u1*v1 - rh) + (u1*v2)) + (u2*v1)) + (u2*v2);
	
	*rhi = rh;
	*rlo = rl;
}

void AddD(double *rhi, double *rlo, double a, double b)
{
	double zhi, zlo;
	zhi = a + b;
	if(fabs(a) > fabs(b)) {
		zlo = zhi - a;
		zlo = b - zlo;
	}
	else {
		zlo = zhi - b;
		zlo = a - zlo;
	}
	
	*rhi = zhi;
	*rlo = zlo;
}

void MulDD(double *rhi, double *rlo, double xh, double xl, double yh, double yl)
{
	double mh, ml;
	double c = 134217729.0;
	double up, u1, u2, vp, v1, v2;
	
	up = xh*c;
	u1 = (xh - up) + up;
	u2 = xh - u1;
	
	vp = yh*c;
	v1 = (yh - vp) + vp;
	v2 = yh - v1;
	
	mh = xh*yh;
	ml = (((u1*v1 - mh) + (u1*v2)) + (u2*v1)) + (u2*v2);
	ml += xh*yl + xl*yh;
	
	*rhi = mh + ml;
	*rlo = (mh - (*rhi)) + ml;
}

void AddDD(double *rhi, double *rlo, double xh, double xl, double yh, double yl)
{
	double r, s;
	r = xh + yh;
	s = (fabs(xh) > fabs(yh)) ? (xh - r + yh + yl + xl) : (yh - r + xh + xl + yl);
	*rhi = r + s;
	*rlo = (r - (*rhi)) + s;
}

void DivideDD(double *chi, double *clo, double a, double b)
{
	*chi = a / b;
	double rhi, rlo;
	MulD(&rhi, &rlo, *chi, b);
	AddDD(&rhi, &rlo, -rhi, -rlo, a, 0.0);
	*clo = rhi / b;
}

// These functions comapre two floats/doubles. Since some platforms may choose to
// flush denormals to zeros before comparison, comparison like a < b may give wrong
// result in "certain cases" where we do need correct compasion result when operands
// are denormals .... these functions comapre floats/doubles using signed integer/long int 
// rep. In other cases, when flushing to zeros is fine, these should not be used.
// Also these doesn't check for nans and assume nans are handled separately as special edge case
// by the caller which calls these functions
// return 0 if both are equal, 1 if x > y and -1 if x < y. 

inline
int compareFloats(float x, float y)
{
	int32f_t a, b;
	
	a.f = x;
	b.f = y;
	
	if( a.i & 0x80000000 )
		a.i = 0x80000000 - a.i;
	if( b.i & 0x80000000 )
		b.i = 0x80000000 - b.i;
		
	if( a.i == b.i )
		return 0;
		
	return a.i < b.i ? -1 : 1;	
}

inline
int compareDoubles(double x, double y)
{
	int64d_t a, b;
	
	a.d = x;
	b.d = y;
	
	if( a.l & 0x8000000000000000LL )
		a.l = 0x8000000000000000LL - a.l;
	if( b.l & 0x8000000000000000LL )
		b.l = 0x8000000000000000LL - b.l;
		
	if( a.l == b.l )
		return 0;
		
	return a.l < b.l ? -1 : 1;	
}

#if 0
static int IsInRTZMode( void )
{
    int error;
    const char *kernel = 
    "__kernel void GetRoundingMode( __global int *out )\n"
    "{\n"
    "   volatile float a = 0x1.0p23f;\n"
    "   volatile float b = -0x1.0p23f;\n"
    "   out[0] = (a + 0x1.fffffep-1f == a) && (b - 0x1.fffffep-1f == b);\n"
    "}\n";
    
    cl_program query = clCreateProgramWithSource(gContext, 1, &kernel, NULL, &error);
    if( NULL == query || error)
    {
        vlog_error( "Error: Unable to create program to detect RTZ mode for the device. (%d)", error );
        return error;
    }
    if(( error = clBuildProgram( query, 1, &gDevice, NULL, NULL, NULL ) ))
    {
        vlog_error( "Error: Unable to build program to detect RTZ mode for the device. Err = %d\n", error );
        char log_msg[2048] = "";
        clGetProgramBuildInfo(query, gDevice, CL_PROGRAM_BUILD_LOG, sizeof( log_msg), log_msg, NULL);
        vlog_error( "Log:\n%s\n", log_msg );
        return error;
    }
    
    cl_kernel k = clCreateKernel( query, "GetRoundingMode", &error );
    if( NULL == k || error)
    {
      vlog_error( "Error: Unable to create kernel to gdetect RTZ mode for the device. Err = %d", error );
        return error;
    }
    
    if((error = clSetKernelArg(k, 0, sizeof( gOutBuffer[gMinVectorSizeIndex]), &gOutBuffer[gMinVectorSizeIndex])))
    {
        vlog_error( "Error: Unable to set kernel arg to detect RTZ mode for the device. Err = %d", error );
        return error;
    }
    
    size_t dim = 1;
    if((error = clEnqueueNDRangeKernel(gQueue, k, 1, NULL, &dim, NULL, 0, NULL, NULL) ))
    {
        vlog_error( "Error: Unable to execute kernel to detect RTZ mode for the device. Err = %d", error );
        return error;
    }
  
    struct{ cl_int isRTZ; }data;
    if(( error = clEnqueueReadBuffer( gQueue, gOutBuffer[gMinVectorSizeIndex], CL_TRUE, 0, sizeof( data ), &data, 0, NULL, NULL)))
    {
        vlog_error( "Error: unable to read RTZ mode data from the device. Err = %d", error );
        return error;
    }

    clReleaseKernel(k);
    clReleaseProgram(query);

    return data.isRTZ;
}
#endif

float Ulp_Error_Double( double test, long double reference )
{
//Check for Non-power-of-two and NaN

  // Note: This function presumes that someone has already tested whether the result is correctly,
  // rounded before calling this function.  That test:
  //
  //    if( (float) reference == test )
  //        return 0.0f;
  //
  // would ensure that cases like fabs(reference) > FLT_MAX are weeded out before we get here. 
  // Otherwise, we'll return inf ulp error here, for what are otherwise correctly rounded 
  // results. 

  // Deal with long double = double
  // On most systems long double is a higher precision type than double. They provide either
  // a 80-bit or greater floating point type, or they provide a head-tail double double format.
  // That is sufficient to represent the accuracy of a floating point result to many more bits
  // than double and we can calculate sub-ulp errors. This is the standard system for which this
  // test suite is designed. 
  //
  // On some systems double and long double are the same thing. Then we run into a problem, 
  // because our representation of the infinitely precise result (passed in as reference above) 
  // can be off by as much as a half double precision ulp itself.  In this case, we inflate the
  // reported error by half an ulp to take this into account.  A more correct and permanent fix
  // would be to undertake refactoring the reference code to return results in this format:
  //
  //    typedef struct DoubleReference
  //    { // true value = correctlyRoundedResult + ulps * ulp(correctlyRoundedResult)        (infinitely precise)
  //        double  correctlyRoundedResult;     // as best we can
  //        double  ulps;                       // plus a fractional amount to account for the difference 
  //    }DoubleReference;                       //     between infinitely precise result and correctlyRoundedResult, in units of ulps.
  //
  // This would provide a useful higher-than-double precision format for everyone that we can use, 
  // and would solve a few problems with representing absolute errors below DBL_MIN and over DBL_MAX for systems
  // that use a head to tail double double for long double. 

    int x;
    long double testVal = test;

    // First, handle special reference values
    if (isinf(reference))
    {
	if (reference == testVal)
	    return 0.0f;

	return INFINITY;
    }

    if (isnan(reference))
    {
	if (isnan(testVal))
	    return 0.0f;

	return INFINITY;
    }

    if ( 0.0L != reference && 0.5L != frexpl(reference, &x) )
    { // Non-zero and Non-power of two

       // allow correctly rounded results to pass through unmolested. (We might add error to it below.) 
       // There is something of a performance optimization here. 
        if( testVal == reference )
            return 0.0f;
    
        // The unbiased exponent of the ulp unit place
        int ulp_exp = DBL_MANT_DIG - 1 - MAX( ilogbl( reference), DBL_MIN_EXP-1 );
        
        // Scale the exponent of the error
        float result = (float) scalbnl( testVal - reference, ulp_exp );
        
        // account for rounding error in reference result on systems that do not have a higher precision floating point type (see above)
        if( sizeof(long double) == sizeof( double ) )
            result += copysignf( 0.5f, result);

        return result;
    }
    
    // reference is a normal power of two or a zero
    // The unbiased exponent of the ulp unit place
    int ulp_exp =  DBL_MANT_DIG - 1 - MAX( ilogbl( reference) - 1, DBL_MIN_EXP-1 );

   // allow correctly rounded results to pass through unmolested. (We might add error to it below.) 
   // There is something of a performance optimization here too. 
    if( testVal == reference )
        return 0.0f;
    
    // Scale the exponent of the error
    float result = (float) scalbnl( testVal - reference, ulp_exp );
    
    // account for rounding error in reference result on systems that do not have a higher precision floating point type (see above)
    if( sizeof(long double) == sizeof( double ) )
        result += copysignf( 0.5f, result);

    return result;
}

/*  */
float Ulp_Error( float test, double reference )
{
    union { double d; uint64_t u; } u; u.d = reference;
    double testVal = test;

  // Note: This function presumes that someone has already tested whether the result is correctly,
  // rounded before calling this function.  That test:
  //
  //    if( (float) reference == test )
  //        return 0.0f;
  //
  // would ensure that cases like fabs(reference) > FLT_MAX are weeded out before we get here. 
  // Otherwise, we'll return inf ulp error here, for what are otherwise correctly rounded 
  // results. 


    if( isinf( reference ) )
    {
        if( testVal == reference )
            return 0.0f;
        
        return (float) (testVal - reference );
    }
    
    if( isinf( testVal) )
    { // infinite test value, but finite (but possibly overflowing in float) reference.
      //
      // The function probably overflowed prematurely here. Formally, the spec says this is
      // an infinite ulp error and should not be tolerated. Unfortunately, this would mean 
      // that the internal precision of some half_pow implementations would have to be 29+ bits 
      // at half_powr( 0x1.fffffep+31, 4) to correctly determine that 4*log2( 0x1.fffffep+31 ) 
      // is not exactly 128.0. You might represent this for example as 4*(32 - ~2**-24), which
      // after rounding to single is 4*32 = 128, which will ultimately result in premature 
      // overflow, even though a good faith representation would be correct to within 2**-29 
      // interally. 
      
      // In the interest of not requiring the implementation go to extraordinary lengths to 
      // deliver a half precision function, we allow premature overflow within the limit 
      // of the allowed ulp error. Towards, that end, we "pretend" the test value is actually
      // 2**128, the next value that would appear in the number line if float had sufficient range.
        testVal = copysign( MAKE_HEX_DOUBLE(0x1.0p128, 0x1LL, 128), testVal );
                
      // Note that the same hack may not work in long double, which is not guaranteed to have
      // more range than double.  It is not clear that premature overflow should be tolerated for 
      // double.
    }

    if( u.u & 0x000fffffffffffffULL )
    { // Non-power of two and NaN
        if( isnan( reference ) && isnan( test ) )
            return 0.0f;    // if we are expecting a NaN, any NaN is fine
    
        // The unbiased exponent of the ulp unit place
        int ulp_exp = FLT_MANT_DIG - 1 - MAX( ilogb( reference), FLT_MIN_EXP-1 );
        
        // Scale the exponent of the error
        return (float) scalbn( testVal - reference, ulp_exp );
    }

    // reference is a normal power of two or a zero
    // The unbiased exponent of the ulp unit place
    int ulp_exp =  FLT_MANT_DIG - 1 - MAX( ilogb( reference) - 1, FLT_MIN_EXP-1 );
    
    // Scale the exponent of the error
    return (float) scalbn( testVal - reference, ulp_exp );
}

#if defined( _MSC_VER )
# pragma warning( pop )
#endif
