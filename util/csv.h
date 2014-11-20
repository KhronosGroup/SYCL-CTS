/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2014 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#pragma once

#include "stl.h"

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
    /** constructor destructor
     */
    csv();
    ~csv();

    /** load a CSV file from disk
     */
    bool load_file( const STRING &path );

    /** release all stored csv information
     */
    void release();

    /** return the number of csv rows (lines)
     */
    int size();

    /** return the last error message set
     */
    bool get_last_error( STRING &out );

    /** extract a csv value
     */
    bool get_item( int row, int column, STRING &out );

protected:
    // the last error message set
    STRING m_error;

    // raw items from the csv file
    VECTOR<STRING> m_items;

    // indices for the start of a row
    VECTOR<int> m_rowIndex;
};

};  // namespace util
};  // namespace sycl_cts
