/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2014 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "csv.h"

namespace sycl_cts
{
namespace util
{

/** constructor
 */
csv::csv()
    : m_error()
    , m_items()
    , m_rowIndex()
{
}

/** destructor
 */
csv::~csv()
{
    release();
}

/** load a CSV file from disk
 */
bool csv::load_file( const STRING &path )
{
    // load the raw file
    IFSTREAM stream( path, std::ios::in | std::ios::binary );
    if ( !stream.is_open() )
    {
        m_error = "unable to open file";
        return false;
    }

    // start parsing a new row
    m_rowIndex.push_back( 0 );

    // make a buffer to pull together each entry
    int index = 0;
    char buffer[64];

    // parse all characters in the input file
    char ch = '\0';
    while ( true )
    {
        // read a character from the stream
        stream.read( &ch, sizeof( ch ) );
        if ( !stream.good() )
            break;

        switch ( ch )
        {
        // skip over these characters
        case ( ' ' ):
        case ( '\r' ):
            continue;

        // new line start of new row
        case ( '\n' ):
        {
            // null terminate the buffer string
            buffer[index] = '\0';
            // add string to items list
            m_items.push_back( STRING( buffer ) );
            index = 0;

            // index of next item marks start of new row
            int items = m_items.size();
            m_rowIndex.push_back( items );
        }
        break;

        // marks next entry in a column
        case ( ',' ):
        {
            // null terminate the buffer string
            buffer[index] = '\0';
            // add string to items list
            m_items.push_back( STRING( buffer ) );
            index = 0;
        }
        break;

        // add new character to the buffer
        default:
            // check for buffer overflow
            if ( index >= ( sizeof( buffer ) - 1 ) )
            {
                m_error = "value exceeds buffer length";
                return false;
            }
            else
                buffer[index++] = ch;

        };  // switch
    };      // while

    // push any remaining item
    if ( index > 0 )
    {
        // null terminate the buffer string
        buffer[index] = '\0';
        // add string to items list
        m_items.push_back( STRING( buffer ) );
    }

    return true;
}

/**
 */
void csv::release()
{
    m_items.clear();
    m_rowIndex.clear();
}

/**
 */
bool csv::get_item( int row, int column, STRING &out )
{
    out = STRING();

    // test if row is valid
    if ( row < 0 || row >= m_rowIndex.size() )
    {
        m_error = "row index out of bounds";
        return false;
    }

    // find the location of the requested element
    int index = m_rowIndex[row] + column;

    // find the index of the end of this row
    int limit = m_items.size() - 1;
    if ( ( row + 1 ) < m_rowIndex.size() )
        limit = ( m_rowIndex[row + 1] - 1 );

    // test the index doesn't pass end of the row
    if ( index < 0 || index > limit )
    {
        m_error = "column index out of bounds";
        return false;
    }

    // output the item asked for
    assert( index < m_items.size() );
    out = m_items[index];

    // success
    return true;
}

/** return the number of csv rows (line)
 */
int csv::size()
{
    return m_rowIndex.size();
}

/** return the last error message set by a csv object
 */
bool csv::get_last_error( STRING &out )
{
    out = m_error;
    return !m_error.empty();
}

};  // namespace util
};  // namespace sycl_cts