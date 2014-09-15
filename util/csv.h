/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2014 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#pragma once

#include <string>

namespace sycl_cts
{
namespace util
{
    
    /** category, testname, timeout, regression y/n
     */

    /** comma separated values file parser
     */
    class csv
    {
    public:

        /** load a CSV file from disk
         */
        bool load( const std::string & path );
        
        /** return the number of csv columns (line)
         */
        int size( );

        /** return the last error message set
         */
        bool get_last_error( std::string & out );

    protected:

        // the last error message set
        std::string m_error;

    };

}; // namespace util
}; // namespace sycl_cts