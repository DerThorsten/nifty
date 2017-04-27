#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>



#include "nifty/python/converter.hxx"
#include "nifty/cgp/features/geometric_features.hxx"

namespace py = pybind11;



namespace nifty{
namespace cgp{



    




    void exportFeatures(py::module & module) {


        typedef Cell1CurvatureFeatures2D Op;
        const auto clsName = std::string("Cell1CurvatureFeatures2D");
        auto pyCls = py::class_<Op>(module, clsName.c_str());

        pyCls
        .def(
            py::init< const std::vector<float> & >(),
            py::arg("sigmas") = std::vector<float>({1.0f, 2.0f, 4.0f})
        )
        .def("__call__",
            [](
                const Op & op,
                const CellGeometryVector<2,1> & cellsGeometry
            ){
                const auto nFeatures = size_t(op.numberOfFeatures());
                const auto nCells1   = size_t(cellsGeometry.size());

                nifty::marray::PyView<float> out({nCells1, nFeatures});

                op(cellsGeometry, out);

                return  out;
            }
        )
        ;

    }

}
}
