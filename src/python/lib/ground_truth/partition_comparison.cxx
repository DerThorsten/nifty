#include <pybind11/pybind11.h>
#include <iostream>
#include <sstream>
#include <pybind11/numpy.h>

#include "nifty/python/converter.hxx"

#include "nifty/ground_truth/overlap.hxx"
#include "nifty/ground_truth/partition_comparison.hxx"

namespace py = pybind11;


namespace nifty{
namespace ground_truth{




    void exportPartitionComparison(py::module & groundTruthModule){

        typedef Overlap<> OverlapType;


        groundTruthModule.def("variationOfInformation",
            [](

                nifty::marray::PyView<uint32_t> labelA,
                nifty::marray::PyView<uint32_t> labelB,
                const bool ignoreDefaultLabel = false
            ){

                {
                    auto  startPtr = &labelA(0);
                    auto  lastElement = &labelA(labelA.size()-1);
                    auto d = lastElement - startPtr + 1;

                    NIFTY_CHECK_OP(d,==,labelA.size(),"labelA must be contiguous")
                }

                {
                    auto  startPtr = &labelB(0);
                    auto  lastElement = &labelB(labelB.size()-1);
                    auto d = lastElement - startPtr + 1;

                    NIFTY_CHECK_OP(d,==,labelB.size(),"labelB must be contiguous")
                }


                VariationOfInformation<> vInfo(
                    &labelA(0),
                    &labelA(0)+labelA.size(),
                    &labelB(0)
                );

                return std::tuple<double,double,double>(
                    vInfo.value(),
                    vInfo.valueFalseCut(),
                    vInfo.valueFalseJoin()
                );

            }
        )
       
        ;
        
    }
}
}