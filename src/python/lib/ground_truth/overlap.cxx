#include <pybind11/pybind11.h>
#include <iostream>
#include <sstream>
#include <pybind11/numpy.h>

#include "nifty/python/converter.hxx"

#include "nifty/ground_truth/overlap.hxx"

namespace py = pybind11;


namespace nifty{
namespace ground_truth{




    void exportOverlap(py::module & groundTruthModule){

        typedef Overlap<> OverlapType;

        py::class_<OverlapType>(groundTruthModule, "Overlap")

            .def("__init__",
                [](
                    OverlapType &instance,
                    const uint32_t maxLabelA,
                    nifty::marray::PyView<uint32_t> labelA,
                    nifty::marray::PyView<uint32_t> labelB
                ) {
                    new (&instance) OverlapType(maxLabelA, labelA, labelB);
                }
            )
            .def("differentOverlaps",[](
                const OverlapType & self,
                nifty::marray::PyView<uint32_t> uv
            ){
                nifty::marray::PyView<float> out({uv.shape(0)});

                for(auto i=0; i<uv.shape(0); ++i){
                    out(i) = self.differentOverlap(uv(i,0),uv(i,1));
                }

                return out;
            })

            .def("bleeding",[](
                const OverlapType & self,
                nifty::marray::PyView<uint32_t> ids
            ){
                nifty::marray::PyView<float> out({ids.shape(0)});

                for(auto i=0; i<ids.shape(0); ++i){
                    out(i) = self.bleeding(ids(i));
                }
                return out;
            })
        ;
        
    }
}
}