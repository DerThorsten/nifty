#define BOOST_TEST_MODULE TestNiftyCompute

#include <boost/test/unit_test.hpp>
#include <iostream>


#include <boost/compute/core.hpp>


//#include "cl-1.2.hpp"

BOOST_AUTO_TEST_CASE(NiftyComputeTest)
{



    namespace compute = boost::compute;
    compute::device device = compute::system::default_device();
    std::cout << "hello from " << device.name() << std::endl;

}
