/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2014 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME group_api

namespace group_api__
{
using namespace sycl_cts;

/** test cl::sycl::range::get(int index) return size_t
 */
#define EXPECTED 1

class TEST_NAME : public util::test_base
{
public:
    /** return information about this test
     *  @param info, test_base::info structure as output
     */
    virtual void get_info( test_base::info &out ) const
    {
        set_test_info( out, TOSTRING( TEST_NAME ), TEST_FILE );
    }

    /** execute the test
     *  @param log, test transcript logging class
     */
    virtual void run( util::logger &log )
    {

        try
        {
            using namespace cl::sycl;
            default_selector sel;
            queue queue(sel);

            int f[8];
            {
                cl::sycl::buffer<int,1> buf(&f[0], cl::sycl::range<1>(8));
                command_group(queue, [&] () {
                    auto a_dev = buf.get_access<access::read_write>();

                    parallel_for_workgroup<class TEST_NAME>(
                        nd_range<3>(range<3>(1, 1, 1), range<3>(1, 1, 1)),
                        [=] (group<3> my_group)
                    {
                        //get_group()
                        id<3> m_get_group = my_group.get_group();
                        if(m_get_group.get(0) > EXPECTED ||
                           m_get_group.get(1) > EXPECTED ||
                           m_get_group.get(2) > EXPECTED  )
                        {
                            a_dev[0] = 0;
                        }else {
                            a_dev[0] = 1;
                        }

                        //get_group_no()
                        if(my_group.get_group_no() > EXPECTED)
                        {
                            a_dev[1] = 0;
                        } else {
                            a_dev[1] = 1;
                        }

                        //get_local_range()
                        range<3> m_get_local_range = my_group.get_local_range();
                        if(     m_get_local_range.get(0) > EXPECTED ||
                                m_get_local_range.get(1) > EXPECTED ||
                                m_get_local_range.get(2) > EXPECTED )
                        {
                            a_dev[2] = 0;
                        }else {
                            a_dev[2] = 1;
                        }

                        //get_global_range()
                        range<3> m_get_global_range = my_group.get_global_range();
                        if(     m_get_global_range.get(0) > EXPECTED ||
                                m_get_global_range.get(1) > EXPECTED ||
                                m_get_global_range.get(2) > EXPECTED )
                        {
                            a_dev[3] = 0;
                        }else {
                            a_dev[3] = 1;
                        }

                        //get_offset()
                        id<3> m_get_offset = my_group.get_offset();
                        if(     m_get_offset.get(0) != 0 ||
                                m_get_offset.get(1) != 0 ||
                                m_get_offset.get(2) != 0 )
                        {
                            a_dev[4] = 0;
                        }else {
                            a_dev[4] = 1;
                        }

                        //get_nd_range()
                        auto m_get_nd_range = my_group.get_nd_range();
                        UNUSED(m_get_nd_range);

                        if ( typeid( m_get_nd_range ) != typeid(nd_range<3>))
                        {
                            a_dev[5] = 0;
                        }else {
                            a_dev[5] = 1;
                        }

                        //get(int dimention)
                        size_t m_get_x = my_group.get(0);
                        size_t m_get_y = my_group.get(1);
                        size_t m_get_z = my_group.get(2);
                        if(     m_get_x.get(0) > EXPECTED ||
                                m_get_y.get(1) > EXPECTED ||
                                m_get_z.get(2) > EXPECTED )
                        {
                            a_dev[6] = 0;
                        }else {
                            a_dev[6] = 1;
                        }

                        //[]
                        size_t m_get_x_op = my_group[0];
                        size_t m_get_y_op = my_group[1];
                        size_t m_get_z_op = my_group[2];
                        if(     m_get_x_op.get(0) > EXPECTED ||
                                m_get_y_op.get(1) > EXPECTED ||
                                m_get_z_op.get(2) > EXPECTED )
                        {
                            a_dev[7] = 0;
                        }else {
                            a_dev[7] = 1;
                        }
                    });
                });
            }

            for(int i = 0; i < 8; i++)
            {
                CHECK_VALUE( log, f[i], 1, i );
            }
        }
        catch ( cl::sycl::exception e )
        {
            log_exception( log, e );
            FAIL( log, "sycl exception caught" );
        }

    }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

} /* namespace group_api__ */
