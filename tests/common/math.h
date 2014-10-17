#pragma once

#include <CL/sycl.hpp>

/** math helper functions
 */
namespace math
{
    using namespace cl::sycl;
    
    static inline 
    void fill( float & e, float v ) {
        e = v;
    }
    
    static inline 
    void fill( float2 & e, float v ) {
        e.x = v; e.y = v;
    }
    
    static inline 
    void fill( float3 & e, float v ) {
        e.x = v; e.y = v; e.z = v;
    }
    
    static inline 
    void fill( float4 & e, float v ) {
        e.x = v; e.y = v; e.z = v; e.w = v;
    }
    
    static inline 
    void fill( float8 & e, float v ) {
        for ( int i = 0; i < 8; i++ )
            e.data[i] = v;
    }
        
    static inline 
    void fill( float16 & e, float v ) {
        for ( int i = 0; i < 16; i++ )
            e.data[i] = v;
    }
    
    static inline
    float abs_max_ulp_error( const float & in, const double ref )
    {
        return fabsf( Ulp_Error( in, ref ) );
    }

    static inline
    float abs_max_ulp_error( const float2 & in, const double ref )
    {
        float l_max = 0.f;
        l_max = fmax( l_max, fabsf( Ulp_Error( in.x, ref ) ) );
        l_max = fmax( l_max, fabsf( Ulp_Error( in.y, ref ) ) );
        return l_max;
    }

    static inline
    float abs_max_ulp_error( const float3 & in, const double ref )
    {
        float l_max = 0.f;
        l_max = fmax( l_max, fabsf( Ulp_Error( in.x, ref ) ) );
        l_max = fmax( l_max, fabsf( Ulp_Error( in.y, ref ) ) );
        l_max = fmax( l_max, fabsf( Ulp_Error( in.z, ref ) ) );
        return l_max;
    }
    
    static inline
    float abs_max_ulp_error( const float4 & in, const double ref )
    {
        float l_max = 0.f;
        l_max = fmax( l_max, fabsf( Ulp_Error( in.x, ref ) ) );
        l_max = fmax( l_max, fabsf( Ulp_Error( in.y, ref ) ) );
        l_max = fmax( l_max, fabsf( Ulp_Error( in.z, ref ) ) );
        l_max = fmax( l_max, fabsf( Ulp_Error( in.w, ref ) ) );
        return l_max;
    }

    static inline
    float abs_max_ulp_error( const float8 & in, const double ref )
    {
        float l_max = 0.f;
        for ( int i = 0; i < 8; i++ )
            l_max = fmax( l_max, fabsf( Ulp_Error( in.data[0], ref ) ) );
        return l_max;
    }
    
    static inline
    float abs_max_ulp_error( const float16 & in, const double ref )
    {
        float l_max = 0.f;
        for ( int i = 0; i < 16; i++ )
            l_max = fmax( l_max, fabsf( Ulp_Error( in.data[0], ref ) ) );
        return l_max;
    }

};
