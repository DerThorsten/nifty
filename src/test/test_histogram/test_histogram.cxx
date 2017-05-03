#define BOOST_TEST_MODULE NiftyBreadthFirstSearchTest

#include <boost/test/unit_test.hpp>

#include <iostream> 

#include "nifty/tools/runtime_check.hxx"
#include "nifty/histogram/histogram.hxx"

static const float tol = 0.000001;

BOOST_AUTO_TEST_CASE(HistogramTest1)
{

    nifty::histogram::Histogram<float> hist(0,6,6);
    std::vector<float> values({0,1,2,3,4,5,6});

    for(const auto & value : values)
        hist.insert(value);

    // test basic insert
    // NIFTY_TEST_EQ_TOL(hist[0],1.0, tol);
    // NIFTY_TEST_EQ_TOL(hist[1],1.0, tol);
    // NIFTY_TEST_EQ_TOL(hist[2],1.0, tol);
    // NIFTY_TEST_EQ_TOL(hist[3],1.0, tol);
    // NIFTY_TEST_EQ_TOL(hist[4],1.0, tol);
    // NIFTY_TEST_EQ_TOL(hist[5],1.0, tol);
    // NIFTY_TEST_EQ_TOL(hist[6],1.0, tol);

    // test sum
    NIFTY_TEST_EQ_TOL(hist.sum(), 7.0, tol);
    
    // test fbinToValue
    NIFTY_TEST_EQ_TOL(hist.binToValue(0.0),0.0, tol);

    // NIFTY_TEST_EQ_TOL(hist.binToValue(1.0),1.5, tol);
    // NIFTY_TEST_EQ_TOL(hist.binToValue(2.0),2.5, tol);
    // NIFTY_TEST_EQ_TOL(hist.binToValue(3.0),3.5, tol);
    // NIFTY_TEST_EQ_TOL(hist.binToValue(4.0),4.5, tol);
    // NIFTY_TEST_EQ_TOL(hist.binToValue(5.0),5.5, tol);
    // NIFTY_TEST_EQ_TOL(hist.binToValue(6.0),6.5, tol);


    // test median
    const float rank =  0.5;
    float median;
    nifty::histogram::quantiles(hist, &rank, &rank+1, &median);
    NIFTY_TEST_EQ_TOL(median,3.0, tol);

}
