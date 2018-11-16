#include <iostream>
#include <sstream>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "xtensor-python/pytensor.hpp"

#include "nifty/ground_truth/seg_to_edges.hxx"

namespace py = pybind11;


namespace nifty{
namespace ground_truth{

    void exportSegToEdges(py::module & groundTruthModule){

        groundTruthModule.def("segToEdges2D",
        [](
            xt::pytensor<uint32_t, 2>   seg
        ){
            xt::pytensor<uint8_t, 1> out = xt::zeros<uint8_t>(seg.shape());
            {
                py::gil_scoped_release liftGil;
                segToEdges2D(seg, out);
            }
            return out;
        },
            py::arg("segmentation")
        );
    }

}
}
