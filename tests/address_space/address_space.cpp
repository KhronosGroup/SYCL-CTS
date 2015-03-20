/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2015 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#include <cassert>
#include <type_traits>

#define EXPECT_EQUALS( lhs, rhs ) \
if ( (lhs) != (rhs) ) \
    return false;

#define EXPECT_ADDRSPACE_EQUALS( expr, as ) \
EXPECT_EQUALS( (expr).get_value(), as )

#define AS_ATTR( n ) \
__attribute__(( address_space( static_cast<int>(( n )) ) ))

#define TEST_NAME address_space

namespace address_space__
{
using namespace sycl_cts;
using namespace cl::sycl::access;

// values based on SPIR 1.2 specification, paragraph 2.2
enum class AS
{
    priv     = 0, // private, but that's a C++ keyword...
    global   = 1,
    constant = 2,
    local    = 3
};


#if defined( __CL_SYCL_DEVICE__ ) || \
    __SYCL_DEVICE_ONLY__ || \
    __SYCL_SINGLE_SOURCE__

template<AS kAs>
struct AddrSpace
{
    constexpr AS get_value() const
    {
        return kAs;
    }
};

// global functions
AddrSpace<AS::priv>     f( AS_ATTR( AS::priv     ) void * )
{
    return { };
}
AddrSpace<AS::global>   f( AS_ATTR( AS::global   ) void * )
{
    return { };
}
AddrSpace<AS::constant> f( AS_ATTR( AS::constant ) void * )
{
    return { };
}
AddrSpace<AS::local>    f( AS_ATTR( AS::local    ) void * )
{
    return { };
}

struct C
{
    /*implicit*/
    C() = default;

    // constructors
    AS m_addrspace;

    explicit C( AS_ATTR( AS::priv     ) void * )
        : m_addrspace  ( AS::priv     ) { }
    explicit C( AS_ATTR( AS::global   ) void * )
        : m_addrspace  ( AS::global   ) { }
    explicit C( AS_ATTR( AS::constant ) void * )
        : m_addrspace  ( AS::constant ) { }
    explicit C( AS_ATTR( AS::local    ) void * )
        : m_addrspace  ( AS::local    ) { }

    // methods
    AddrSpace<AS::priv>     g( AS_ATTR( AS::priv     ) void* )
    {
        return { };
    }

    AddrSpace<AS::global>   g( AS_ATTR( AS::global   ) void* )
    {
        return { };
    }

    AddrSpace<AS::constant> g( AS_ATTR( AS::constant ) void* )
    {
        return { };
    }

    AddrSpace<AS::local>    g( AS_ATTR( AS::local    ) void* )
    {
        return { };
    }

    // operators
    AddrSpace<AS::priv>     operator+(AS_ATTR( AS::priv     ) void * )
    {
        return { };
    }

    AddrSpace<AS::global>   operator+(AS_ATTR( AS::global   ) void * )
    {
        return { };
    }

    AddrSpace<AS::constant> operator+(AS_ATTR( AS::constant ) void * )
    {
        return { };
    }

    AddrSpace<AS::local>    operator+(AS_ATTR( AS::local    ) void * )
    {
        return { };
    }
};

bool test_overload_resolution(
    AS_ATTR( AS::priv     ) void * privPtr,
    AS_ATTR( AS::global   ) void * globalPtr,
    AS_ATTR( AS::constant ) void * constantPtr,
    AS_ATTR( AS::local    ) void * localPtr )
{
    // functions
    EXPECT_ADDRSPACE_EQUALS( f( privPtr     ), AS::priv     );
    EXPECT_ADDRSPACE_EQUALS( f( globalPtr   ), AS::global   );
    EXPECT_ADDRSPACE_EQUALS( f( constantPtr ), AS::constant );
    EXPECT_ADDRSPACE_EQUALS( f( localPtr    ), AS::local    );

    // methods
    C c;
    EXPECT_ADDRSPACE_EQUALS( c.g( privPtr     ), AS::priv     );
    EXPECT_ADDRSPACE_EQUALS( c.g( globalPtr   ), AS::global   );
    EXPECT_ADDRSPACE_EQUALS( c.g( constantPtr ), AS::constant );
    EXPECT_ADDRSPACE_EQUALS( c.g( localPtr    ), AS::local    );

    // constructors
    EXPECT_EQUALS( C { privPtr     }.m_addrspace, AS::priv     );
    EXPECT_EQUALS( C { globalPtr   }.m_addrspace, AS::global   );
    EXPECT_EQUALS( C { constantPtr }.m_addrspace, AS::constant );
    EXPECT_EQUALS( C { localPtr    }.m_addrspace, AS::local    );

    // operators
    EXPECT_ADDRSPACE_EQUALS( c + privPtr,     AS::priv     );
    EXPECT_ADDRSPACE_EQUALS( c + globalPtr,   AS::global   );
    EXPECT_ADDRSPACE_EQUALS( c + constantPtr, AS::constant );
    EXPECT_ADDRSPACE_EQUALS( c + localPtr,    AS::local    );

    return true;
}

// the parameter has no address space specification, so it'll
// get duplication based on the argument's address space
// it should call the appropiate overload of f(), and that will
// tell us the address space the parameter was in
template <typename T,
    typename std::enable_if<std::is_pointer<T>::value, void> * = nullptr>
AS getAddrSpace( T p )
{
    return f( p ).get_value();
}

struct D
{
    template<typename T,
        typename std::enable_if<std::is_pointer<T>::value, void> * = nullptr>
    AS getAddrSpace( T p )
    {
        return address_space__::getAddrSpace( p );
    }

    template<typename T,
        typename std::enable_if<std::is_pointer<T>::value, void> * = nullptr>
    AS operator[]( T p )
    {
        return address_space__::getAddrSpace( p );
    }
};

bool test_duplication(
    AS_ATTR( AS::priv     ) void* privPtr,
    AS_ATTR( AS::global   ) void* globalPtr,
    AS_ATTR( AS::constant ) void* constantPtr,
    AS_ATTR( AS::local    ) void* localPtr )
{
    EXPECT_EQUALS( getAddrSpace( privPtr     ), AS::priv     );
    EXPECT_EQUALS( getAddrSpace( globalPtr   ), AS::global   );
    EXPECT_EQUALS( getAddrSpace( constantPtr ), AS::constant );
    EXPECT_EQUALS( getAddrSpace( localPtr    ), AS::local    );

    D d;
    EXPECT_EQUALS( d.getAddrSpace( privPtr     ), AS::priv     );
    EXPECT_EQUALS( d.getAddrSpace( globalPtr   ), AS::global   );
    EXPECT_EQUALS( d.getAddrSpace( constantPtr ), AS::constant );
    EXPECT_EQUALS( d.getAddrSpace( localPtr    ), AS::local    );

    EXPECT_EQUALS( d[privPtr],     AS::priv     );
    EXPECT_EQUALS( d[globalPtr],   AS::global   );
    EXPECT_EQUALS( d[constantPtr], AS::constant );
    EXPECT_EQUALS( d[localPtr],    AS::local    );

    return true;
}

template<typename T,
    typename std::enable_if<std::is_pointer<T>::value, void> * = nullptr>
T id( T p )
{
    return p;
}

bool test_return_type_deduction(
    AS_ATTR( AS::priv     ) void* privPtr,
    AS_ATTR( AS::global   ) void* globalPtr,
    AS_ATTR( AS::constant ) void* constantPtr,
    AS_ATTR( AS::local    ) void* localPtr )
{
    // return type deduction
    EXPECT_EQUALS( getAddrSpace( id( privPtr     ) ), AS::priv     );
    EXPECT_EQUALS( getAddrSpace( id( globalPtr   ) ), AS::global   );
    EXPECT_EQUALS( getAddrSpace( id( constantPtr ) ), AS::constant );
    EXPECT_EQUALS( getAddrSpace( id( localPtr    ) ), AS::local    );

    return true;
}

bool test_initialization(
    AS_ATTR( AS::priv     ) void* privPtr,
    AS_ATTR( AS::global   ) void* globalPtr,
    AS_ATTR( AS::constant ) void* constantPtr,
    AS_ATTR( AS::local    ) void* localPtr )
{
    auto p1 = privPtr;
    auto p2 = globalPtr;
    auto p3 = constantPtr;
    auto p4 = localPtr;

    EXPECT_EQUALS( getAddrSpace( p1 ), AS::priv     );
    EXPECT_EQUALS( getAddrSpace( p2 ), AS::global   );
    EXPECT_EQUALS( getAddrSpace( p3 ), AS::constant );
    EXPECT_EQUALS( getAddrSpace( p4 ), AS::local    );

    return true;
}

void test( bool & pass )
{
    pass = test_overload_resolution  ( nullptr, nullptr, nullptr, nullptr );
    if ( !pass )
        return;
    pass = test_duplication          ( nullptr, nullptr, nullptr, nullptr );
    if ( !pass )
        return;
    pass = test_return_type_deduction( nullptr, nullptr, nullptr, nullptr );
    if ( !pass )
        return;
    pass = test_initialization       ( nullptr, nullptr, nullptr, nullptr );
    if ( !pass )
        return;
}

#endif

class TEST_NAME : public sycl_cts::util::test_base
{
public:
    virtual void get_info( test_base::info & out ) const override
    {
        set_test_info( out, TOSTRING( TEST_NAME ), TEST_FILE );
    }

    virtual void run( util::logger & log ) override
    {
        bool pass = false;
        cl::sycl::range<1> r( 1 );
        try
        {
            cl::sycl::buffer<bool, 1> buf( &pass, r );

            cl::sycl::cpu_selector sel;

            cl::sycl::queue q( sel );
            q.submit( [&]( cl::sycl::handler & cgh )
            {
                auto acc = buf.get_access<cl::sycl::access::mode::read_write>( cgh );

                cgh.single_task<TEST_NAME>( [=]()
                {
                    bool b = acc[0];
#if defined(__CL_SYCL_DEVICE__) || __SYCL_DEVICE_ONLY__ || __SYCL_SINGLE_SOURCE__
                    test( b );
#endif
                    acc[0] = b;
                } );
            } );

            q.wait_and_throw();

        }
        catch ( cl::sycl::exception e )
        {
            log_exception( log, e );
            FAIL( log, "sycl exception caught" );
        }

        if ( !pass )
            FAIL( log, "Device compiler failed address space tests" );

        return;
    }
};

util::test_proxy<TEST_NAME> proxy;

} /* namespace address_space */
