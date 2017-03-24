#include <pybind11/pybind11.h>
#include <iostream>
#include <sstream>
#include <pybind11/numpy.h>

#include "nifty/python/converter.hxx"

#include "nifty/ground_truth/seg_to_edges.hxx"

namespace py = pybind11;


namespace nifty{
namespace ground_truth{


    void exportSegToEdges(py::module & groundTruthModule){

        groundTruthModule.def("segToEdges2D",
        [](
            marray::PyView<uint32_t, 2 >   seg
        ){
            marray::PyView<uint8_t> out({
                size_t(seg.shape(0)),
                size_t(seg.shape(1)),
            }); 
        
            segToEdges2D(seg, out);

            return out;
        },
            py::arg("segmentation")
        );
    }


}
}