/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2014 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#pragma once

#include "stl.h"
#include "singleton.h"

namespace sycl_cts
{
namespace util
{

/** command line parser
 */
class cmdarg
    : public singleton<cmdarg>
{
public:

    /** parse a set of given command line arguments
     *  @return, false if there was an error parsing
     *           true if the cmd line was parsed
     */
    bool parse( const int argc, const char **args );

    /** search for a specific key
     */
    bool find_key( const STRING & key ) const;

    /** find a value from a given key
     *  @param, key, the key to try and locate
     *  @param, value, string to receive the value that was associated
     *                 with the given key
     */
    bool get_value( const STRING & key, STRING & value ) const;

    /** return the last error message given
     *  @param, string to receive the last error message
     */
    bool get_last_error( STRING & out ) const;

protected:

    /** a simple key value pair container
     */
    struct pair
    {
        STRING key;
        STRING value;
    };
    
    /** the options list
     */
    VECTOR<pair> m_pairs;

    /** add a pair to the list
     */
    void push_pair( const pair & opt );

    // the last error message
    STRING m_error;

};

}; // namespace util
}; // namespace sycl_cts