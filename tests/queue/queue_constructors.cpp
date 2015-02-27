/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2015 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME queue_constructors

namespace queue_constructors__
{
using namespace sycl_cts;

/** tests the constructors for cl::sycl::queue
 */
class TEST_NAME : public util::test_base
{
public:
    /** return information about this test
     */
    virtual void get_info( test_base::info &out ) const override
    {
        set_test_info( out, TOSTRING( TEST_NAME ), TEST_FILE );
    }

    /** execute this test
     */
    virtual void run( util::logger &log ) override
    {
        try
        {
            cl::sycl::function_class<void( cl::sycl::exception_list )> asyncHandler = [&]( cl::sycl::exception_list l )
            {
            };

            /** check default constructor and destructor
            */
            {
                cl::sycl::queue queue;

                if ( queue.is_host() )
                {
                    FAIL( log, "queue was not constructed correctly" );
                }

                if ( queue.get() == nullptr )
                {
                    FAIL( log, "queue was not constructed correctly" );
                }
            }

            /** check (functor_class) constructor
            */
            {
                cl::sycl::queue queue( asyncHandler );

                if ( queue.is_host() )
                {
                    FAIL( log, "queue was not constructed correctly" );
                }

                if ( queue.get() == nullptr )
                {
                    FAIL( log, "queue was not constructed correctly" );
                }
            }

            /** check (host_selector) constructor
            */
            {
                cl::sycl::host_selector selector;
                cl::sycl::queue queue( selector );

                if ( !queue.is_host() )
                {
                    FAIL( log, "queue was not constructed correctly" );
                }

                if ( queue.get() != nullptr )
                {
                    FAIL( log, "queue was not constructed correctly" );
                }
            }

            /** check (device_selector, bool = false) constructor
            */
            {
                cts_selector selector;
                cl::sycl::queue queue( selector );

                if ( queue.is_host() )
                {
                    FAIL( log, "queue was not constructed correctly" );
                }

                if ( queue.get() == nullptr )
                {
                    FAIL( log, "queue was not constructed correctly" );
                }
            }

            /** check (device_selector, bool) constructor
            */
            {
                cts_selector selector;
                cl::sycl::queue queue( selector, true );

                if ( queue.is_host() )
                {
                    FAIL( log, "queue was not constructed correctly" );
                }

                if ( queue.get() == nullptr )
                {
                    FAIL( log, "queue was not constructed correctly" );
                }
            }

            /** check (context, device_selector, bool = false) constructor
            */
            {
                cts_selector selector;
                cl::sycl::context context( selector );
                cl::sycl::queue queue( context, selector );

                if ( queue.is_host() )
                {
                    FAIL( log, "queue was not constructed correctly" );
                }

                if ( queue.get() == nullptr )
                {
                    FAIL( log, "queue was not constructed correctly" );
                }
            }

            /** check (context, device_selector, bool) constructor
            */
            {
                cts_selector selector;
                cl::sycl::context context( selector );
                cl::sycl::queue queue( context, selector, true );

                if ( queue.is_host() )
                {
                    FAIL( log, "queue was not constructed correctly" );
                }

                if ( queue.get() == nullptr )
                {
                    FAIL( log, "queue was not constructed correctly" );
                }
            }

            /** check (context, device_selector, functor_class, bool = false) constructor
            */
            {
                cts_selector selector;
                cl::sycl::context context( selector );
                cl::sycl::queue queue( context, selector, asyncHandler );

                if ( queue.is_host() )
                {
                    FAIL( log, "queue was not constructed correctly" );
                }

                if ( queue.get() == nullptr )
                {
                    FAIL( log, "queue was not constructed correctly" );
                }
            }

            /** check (context, device_selector, functor_class, bool) constructor
            */
            {
                cts_selector selector;
                cl::sycl::context context( selector );
                cl::sycl::queue queue( context, selector, asyncHandler, true );

                if ( queue.is_host() )
                {
                    FAIL( log, "queue was not constructed correctly" );
                }

                if ( queue.get() == nullptr )
                {
                    FAIL( log, "queue was not constructed correctly" );
                }
            }

            /** check (device, bool = false) constructor
            */
            {
                cts_selector selector;
                cl::sycl::device device( selector );
                cl::sycl::queue queue( device );

                if ( queue.is_host() )
                {
                    FAIL( log, "queue was not constructed correctly" );
                }

                if ( queue.get() == nullptr )
                {
                    FAIL( log, "queue was not constructed correctly" );
                }
            }

            /** check (device_selector, bool) constructor
            */
            {
                cts_selector selector;
                cl::sycl::device device( selector );
                cl::sycl::queue queue( device, true );

                if ( queue.is_host() )
                {
                    FAIL( log, "queue was not constructed correctly" );
                }

                if ( queue.get() == nullptr )
                {
                    FAIL( log, "queue was not constructed correctly" );
                }
            }

            /** check (context, device, bool = false) constructor
            */
            {
                cts_selector selector;
                cl::sycl::device device( selector );
                cl::sycl::context context( device );
                cl::sycl::queue queue( context, device );

                if ( queue.is_host() )
                {
                    FAIL( log, "queue was not constructed correctly" );
                }

                if ( queue.get() == nullptr )
                {
                    FAIL( log, "queue was not constructed correctly" );
                }
            }

            /** check (context, device, bool) constructor
            */
            {
                cts_selector selector;
                cl::sycl::device device( selector );
                cl::sycl::context context( device );
                cl::sycl::queue queue( context, device, bool );

                if ( queue.is_host() )
                {
                    FAIL( log, "queue was not constructed correctly" );
                }

                if ( queue.get() == nullptr )
                {
                    FAIL( log, "queue was not constructed correctly" );
                }
            }

            /** check (context, device, functor_class, bool = false) constructor
            */
            {
                cts_selector selector;
                cl::sycl::device device( selector );
                cl::sycl::context context( device );
                cl::sycl::queue queue( context, device, asyncHandler );

                if ( queue.is_host() )
                {
                    FAIL( log, "queue was not constructed correctly" );
                }

                if ( queue.get() == nullptr )
                {
                    FAIL( log, "queue was not constructed correctly" );
                }
            }

            /** check (context, device_selector, functor_class, bool) constructor
            */
            {
                cts_selector selector;
                cl::sycl::device device( selector );
                cl::sycl::context context( device );
                cl::sycl::queue queue( context, device, asyncHandler, true );

                if ( queue.is_host() )
                {
                    FAIL( log, "queue was not constructed correctly" );
                }

                if ( queue.get() == nullptr )
                {
                    FAIL( log, "queue was not constructed correctly" );
                }
            }

            /** check copy constructor
            */
            {
                cts_selector selector;
                cl::sycl::queue queueA( selector );
                cl::sycl::queue queueB( queueA );

                if ( queueA.get() != queueB.get() )
                {
                    FAIL( log, "queue was not copied correctly" );
                }
            }

            /** check assignment operator
            */
            {
                cts_selector selector;
                cl::sycl::queue queueA( selector );
                cl::sycl::queue queueB = queueA;

                if ( queueA.get() != queueB.get() )
                {
                    FAIL( log, "queue was not assigned correctly" );
                }
            }
        }
        catch ( cl::sycl::exception e )
        {
            log_exception( log, e );
            FAIL( log, "a sycl exception was caught" );
        }
    }
};

// register this test with the test_collection
util::test_proxy<TEST_NAME> proxy;

} /* namespace queue_constructors__ */
