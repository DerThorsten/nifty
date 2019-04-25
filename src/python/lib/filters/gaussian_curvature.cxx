#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <iostream>

#include "xtensor-python/pytensor.hpp"

#include "nifty/filters/gaussian_curvature.hxx"

namespace py = pybind11;

namespace nifty{
namespace filters{


    void exportGaussianCurvature(py::module & module) {

        typedef GaussianCurvature2D<> ClsType;
        typedef typename ClsType::ValueType ValueType;
        auto pyCls = py::class_< ClsType >(module, "GaussianCurvature2D");
        pyCls
        .def(py::init<
            const ValueType, int, const ValueType
        >(),
            py::arg("sigma"),
            py::arg("radius") = -1,
            py::arg("windowRatio") = 2.0
        )
        .def_property_readonly("radius",&ClsType::radius)
        .def("__call__",[](
            const ClsType & self,
            xt::pytensor<float, 2> coords,
            const bool loop
        ){
            typedef typename xt::pytensor<float, 1>::shape_type ShapeType;
            ShapeType shape = {coords.shape()[0]};
            xt::pytensor<float, 1> out = xt::zeros<float>(shape);
            self(coords, out, loop);
            return out;
        })
        ;

    }

}
}
