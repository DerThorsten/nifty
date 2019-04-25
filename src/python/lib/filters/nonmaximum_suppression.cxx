#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <iostream>

#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"
#include "nifty/filters/nonmaximum_suppression.hxx"

namespace py = pybind11;

namespace nifty{
namespace filters{


    void exportNonMaximumSuppression(py::module & module) {

        module.def("nonMaximumDistanceSuppression", [](const xt::pyarray<float> & distanceMap,
                                                       const xt::pytensor<uint64_t, 2> & points){
            std::set<uint64_t> pointsTmp;
            {
                py::gil_scoped_release allowThreads;
                nonMaximumDistanceSuppression(distanceMap, points, pointsTmp);

            }

            const int64_t nPoints = pointsTmp.size();
            const int64_t ndim = points.shape()[1];
            xt::pytensor<uint64_t, 2> pointsOut = xt::zeros<uint64_t>({nPoints, ndim});
            {
                py::gil_scoped_release allowThreads;
                std::size_t i = 0;
                for(const uint64_t pointId: pointsTmp) {
                    for(int d = 0; d < ndim; ++d) {
                        pointsOut(i, d) = points(pointId, d);
                    }
                    ++i;
                }
            }
            return pointsOut;

        }, py::arg("distanceMap"), py::arg("points"));
    }

}
}
