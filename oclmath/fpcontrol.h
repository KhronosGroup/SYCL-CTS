
#ifndef _fpcontrol_h
#define _fpcontrol_h

// In order to get tests for correctly rounded operations (e.g. multiply) to work properly we need to be able to set the reference hardware 
// to FTZ mode if the device hardware is running in that mode.  We have explored all other options short of writing correctly rounded operations 
// in integer code, and have found this is the only way to correctly verify operation.
//
// Non-Apple implementations will need to provide their own implentation for these features.  If the reference hardware and device are both 
// running in the same state (either FTZ or IEEE compliant modes) then these functions may be empty.  If the device is running in non-default 
// rounding mode (e.g. round toward zero), then these functions should also set the reference device into that rounding mode.
#if defined( __APPLE__ ) || defined( _MSC_VER ) || defined( __linux__ ) || defined (__MINGW32__)
    typedef int     FPU_mode_type;
#if defined( __i386__ ) || defined( __x86_64__ ) || defined( _MSC_VER ) || defined( __MINGW32__ )
    #include <xmmintrin.h>
#elif defined( __PPC__ ) 
    #include <fpu_control.h>
    extern __thread fpu_control_t fpu_control;
#endif    
    // Set the reference hardware floating point unit to FTZ mode
    static inline void ForceFTZ( FPU_mode_type *mode )
    {
#if defined( __i386__ ) || defined( __x86_64__ ) || defined( _MSC_VER ) || defined (__MINGW32__)
        *mode = _mm_getcsr();
        _mm_setcsr( *mode | 0x8040);
#elif defined( __PPC__ ) 
        *mode = fpu_control;
        fpu_control |= _FPU_MASK_NI;
#elif defined ( __arm__ )
        unsigned fpscr;
        __asm__ volatile ("fmrx %0, fpscr" : "=r"(fpscr));
        *mode = fpscr;
        __asm__ volatile ("fmxr fpscr, %0" :: "r"(fpscr | (1U << 24)));
#else
        #error ForceFTZ needs an implentation
#endif
    }
    
    // Disable the denorm flush to zero
    static inline void DisableFTZ( FPU_mode_type *mode )
    {
#if defined( __i386__ ) || defined( __x86_64__ ) || defined( _MSC_VER ) || defined (__MINGW32__)
        *mode = _mm_getcsr();
        _mm_setcsr( *mode & ~0x8040);
#elif defined( __PPC__ ) 
        *mode = fpu_control;
        fpu_control &= ~_FPU_MASK_NI;
#elif defined ( __arm__ )
        unsigned fpscr;
        __asm__ volatile ("fmrx %0, fpscr" : "=r"(fpscr));
        *mode = fpscr;
        __asm__ volatile ("fmxr fpscr, %0" :: "r"(fpscr & ~(1U << 24)));
#else
#error DisableFTZ needs an implentation
#endif  
    }

    // Restore the reference hardware to floating point state indicated by *mode
    static inline void RestoreFPState( FPU_mode_type *mode )
    {
#if defined( __i386__ ) || defined( __x86_64__ ) || defined( _MSC_VER ) || defined (__MINGW32__)
        _mm_setcsr( *mode );
#elif defined( __PPC__)
        fpu_control = *mode;
#elif defined (__arm__)
        __asm__ volatile ("fmxr fpscr, %0" :: "r"(*mode));
#else
        #error RestoreFPState needs an implementation
#endif
    }
#else
        #error ForceFTZ and RestoreFPState need implentations
#endif

#endif