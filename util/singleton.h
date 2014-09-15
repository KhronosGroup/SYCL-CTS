/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2014 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#pragma once

#include <memory>
#include <assert.h>

namespace sycl_cts
{
namespace util
{

/** implement a singleton interface to all derived class.
 *  template argument T must be the derived class.
 */
template <class T>
class singleton
{
    // must be a friend of class T to access the constructor
    friend T;
    
    // singleton instance
    static std::unique_ptr<T> m_instance;

public:

    /** destructor
     *  ensure that we release the singleton instance
     */
    virtual ~singleton( )
    {
        release( );
    }

    /** get singleton instance
     */
    static T & instance( )
    {
        // if the instance has not be created
        if (m_instance.get() == nullptr) 
        {
            m_instance.reset( new T( ) );
        }
        assert( m_instance.get( ) != nullptr );

        // return the singleton instance
        return *( m_instance.get( ) );
    }

    /** release this instance
     */
    static void release( )
    {
        m_instance.release();
    }
    
};

// instance of the singleton
// due to #pragma once this will only be implemented once
template <class T>
std::unique_ptr<T> singleton<T>::m_instance;

}; // namespace util
}; // namespace sycl_cts