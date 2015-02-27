/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2015 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "selector.h"
#include "cmdarg.h"

namespace sycl_cts
{
namespace util
{

/** constructor
 */
selector::selector()
    : m_device( ctsdevice::unknown )
{
}

/** set the default device to use for the SYCL CTS
 *  @param name, the name of the device to use.
 *  valid options are:
 *      'host'
 *      'opencl_cpu'
 *      'opencl_gpu'
 */
void selector::set_default( const STRING &name )
{
    if ( name == "host" )
        m_device = ctsdevice::host;

    if ( name == "opencl_cpu" )
        m_device = ctsdevice::opencl_cpu;

    if ( name == "opencl_gpu" )
        m_device = ctsdevice::opencl_gpu;
}

/** set the default device type via enum
 */
void selector::set_default( ctsdevice deviceType )
{
    m_device = deviceType;
}

/** return the default device of choice for this cts run
 */
selector::ctsdevice selector::get_default()
{
    // default to host device if a valid one was not specified
    if ( m_device == ctsdevice::unknown )
        m_device = ctsdevice::host;

    // return the cached device type
    return m_device;
}

}  // namespace util
}  // namespace sycl_cts
