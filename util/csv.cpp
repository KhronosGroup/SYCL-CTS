/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2014 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include <stdio.h>

#include "csv.h"

namespace sycl_cts
{
namespace util
{

/** load a CSV file from disk
 */
bool csv::load( const std::string & path )
{


    return false;
}

/** return the number of csv columns (line)
 */
int csv::size( )
{
    return 0;
}

/** return the last error message set by a csv object
 */
bool csv::get_last_error( std::string & out )
{
    out = m_error;
    return ! m_error.empty( );
}

}; // namespace util
}; // namespace sycl_cts