#define BOOST_TEST_MODULE NiftyRagTest

#include <boost/test/unit_test.hpp>
#include "xtensor/xarray.hpp"

#include <iostream>
#include <random>

#include "nifty/tools/runtime_check.hxx"
#include "nifty/graph/rag/grid_rag_stacked_2d.hxx"
#include "nifty/graph/rag/grid_rag_stacked_2d_hdf5.hxx"


void getStackedSegmentation(xt::xarray<uint32_t> & seg,
        const std::vector<std::size_t> & shape) {
    // random generator
    std::default_random_engine gen;
    std::uniform_int_distribution<int> distr(0,9);
    auto draw = std::bind(distr, gen);
    // segmentation
    uint32_t label = 0;
    for(auto z = 0; z < shape[0]; ++z) {
        if(z > 0) {
            ++label;
        }
        for(auto y = 0; y < shape[1]; ++y) {
            for(auto x = 0; x < shape[2]; ++x) {
                seg(z,y,x) = label;
                if( draw() > 8  && (y != shape[1] - 1 && x != shape[2] - 1) )
                    ++label;
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(StackedRagHdf5Test)
{
    typedef nifty::graph::Hdf5Labels<3, uint32_t> LabelsProxy;

    std::vector<std::size_t> shape({20,100,100});
    xt::xarray<uint32_t> seg({20L, 100L, 100L});
    getStackedSegmentation(seg, shape);
    uint32_t maxLabel = *(std::max_element(seg.begin(), seg.end()));
    std::cout << "MaxLabel: " << maxLabel << std::endl;
    auto segFile = nifty::hdf5::createFile("./seg_tmp.h5");
    std::vector<std::size_t> chunks({10,50,50});
    nifty::hdf5::Hdf5Array<uint32_t> labels(segFile,
            "data",
            shape.begin(),
            shape.end(),
            chunks.begin());
    std::vector<std::size_t> start({0,0,0});
    labels.writeSubarray(start.begin(), seg);
    LabelsProxy labelsProxy(labels, maxLabel);
    nifty::graph::GridRagStacked2D<LabelsProxy> rag(labelsProxy);
}

/*
BOOST_AUTO_TEST_CASE(StackedRagTest)
{

}
*/
