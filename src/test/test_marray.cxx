#define BOOST_TEST_MODULE NiftyTestMarray

#include <boost/test/unit_test.hpp>

#include <iostream> 

#include "nifty/tools/runtime_check.hxx"
#include "nifty/marray/marray.hxx"

#include "nifty/tools/timer.hxx"

BOOST_AUTO_TEST_CASE(StridesTest)
{
    {
        std::vector<size_t> shape({10,20});
        nifty::marray::Marray<int> a(shape.begin(), shape.end(),0, nifty::marray::FirstMajorOrder);
        NIFTY_TEST_OP(a.strides(0),==,20);
        NIFTY_TEST_OP(a.strides(1),==,1);

        NIFTY_TEST(a.coordinateOrder() == nifty::marray::FirstMajorOrder);

        const auto da = &a(0,1) - &a(0,0);
        NIFTY_TEST_OP(da,==,1);

        const auto db = &a(1,0) - &a(0,0);
        NIFTY_TEST_OP(db,!=,1);



        
    }
    {
        std::vector<size_t> shape({10,20});
        nifty::marray::Marray<int> a(shape.begin(), shape.end(),0, nifty::marray::LastMajorOrder);
        
        NIFTY_TEST(a.coordinateOrder() == nifty::marray::LastMajorOrder);
        
        const auto da = &a(0,1) - &a(0,0);
        NIFTY_TEST_OP(da,!=,1);

        const auto db = &a(1,0) - &a(0,0);
        NIFTY_TEST_OP(db,==,1);

        NIFTY_TEST_OP(a.strides(0),==,1);
        NIFTY_TEST_OP(a.strides(1),==,10);

        
    }
}

