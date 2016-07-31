#define BOOST_TEST_MODULE NiftyTestMarray

#include <boost/test/unit_test.hpp>

#include <iostream> 

#include "nifty/tools/runtime_check.hxx"
#include "nifty/marray/marray.hxx"


BOOST_AUTO_TEST_CASE(StridesTest)
{
    std::vector<size_t> shape({10,20});
    nifty::marray::Marray<int> a(shape.begin(), shape.end());
    NIFTY_TEST_OP(a.strides(1),==,1);
    NIFTY_TEST_OP(a.strides(0),==,20);
}

