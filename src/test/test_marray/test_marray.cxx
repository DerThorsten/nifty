#define BOOST_TEST_MODULE NiftyMarrayHdf5Test

#include <boost/test/unit_test.hpp>

#include <iostream> 
#include <nifty/marray/marray.hxx>
#include "nifty/tools/runtime_check.hxx"



BOOST_AUTO_TEST_CASE(TestMarrayHdf5)
{

    nifty::marray::Marray<float, std::allocator<size_t> > array({10,10});

}
