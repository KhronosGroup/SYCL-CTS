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

/** implement a singleton interface to all derived class.
 *  template argument T must be the derived class.
 */
template <class T>
class singleton
{
    // must be a friend of class T to access the constructor
    friend T;

    // singleton instance
    static UNIQUE_PTR<T> m_instance;

public:
    /** destructor
     *  ensure that we release the singleton instance
     */
    virtual ~singleton()
    {
        release();
    }

    /** get singleton instance
     */
    static T &instance()
    {
        // if the instance has not be created
        if ( m_instance.get() == nullptr )
        {
            m_instance.reset( new T() );
        }
        assert( m_instance.get() != nullptr );

        // return the singleton instance
        return *( m_instance.get() );
    }

    /** release this instance
     */
    static void release()
    {
        m_instance.release();
    }
};

/** instance of the singleton
 */
template <class T>
std::unique_ptr<T> singleton<T>::m_instance;

/** easy singleton accessors
 */
template <class T>
static inline T &get()
{
    return T::instance();
}

}  // namespace util
}  // namespace sycl_cts
