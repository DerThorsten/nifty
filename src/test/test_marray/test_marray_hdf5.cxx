#define BOOST_TEST_MODULE NiftyMarrayHdf5Test

#include <boost/test/unit_test.hpp>

#include <iostream> 
#include <nifty/marray/marray.hxx>
#include <nifty/marray/marray_hdf5.hxx>
#include "nifty/tools/runtime_check.hxx"









BOOST_AUTO_TEST_CASE(TestMarrayHdf5)
{
    
    nifty::marray::Marray<float > array({10,10});
    auto f = nifty::marray::hdf5::createFile("testFile.h5");
    array = 2.0;
    nifty::marray::hdf5::save(f, "data", array);



}
