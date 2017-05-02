#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>



#include "nifty/python/converter.hxx"
#include "nifty/cgp/geometry.hxx"
#include "nifty/cgp/bounds.hxx"
#include "nifty/cgp/features/geometric_features.hxx"

namespace py = pybind11;



namespace nifty{
namespace cgp{



    




    void exportFeatures(py::module & module) {


        {
            typedef Cell1CurvatureFeatures2D Op;
            const auto clsName = std::string("Cell1CurvatureFeatures2D");
            auto pyCls = py::class_<Op>(module, clsName.c_str());

            pyCls
            .def(
                py::init< const std::vector<float> &,const std::vector<float> & >(),
                py::arg("sigmas")    = std::vector<float>({1.0f, 2.0f, 4.0f}),
                py::arg("quantiles") = std::vector<float>({0.1f, 0.25f, 0.50f, 0.75f, 0.9f})
            )
            .def("__call__",
                [](
                    const Op & op,
                    const CellGeometryVector<2,1> &  cell1GeometryVector,
                    const CellBoundedByVector<2,1> & cell1BoundedByVector
                ){
                    const auto nFeatures = size_t(op.numberOfFeatures());
                    const auto nCells1   = size_t(cell1GeometryVector.size());

                    nifty::marray::PyView<float> out({nCells1, nFeatures});

                    op(cell1GeometryVector, cell1BoundedByVector, out);

                    return  out;
                },
                py::arg("cell1GeometryVector"),
                py::arg("cell1BoundedByVector")
            )
            ;
        }
        {
            typedef Cell1LineSegmentDist2D Op;
            const auto clsName = std::string("Cell1LineSegmentDist2D");
            auto pyCls = py::class_<Op>(module, clsName.c_str());

            pyCls
            .def(
                py::init< const std::vector<size_t> &>(),
                py::arg("dists")  =  std::vector<size_t>({size_t(3),size_t(5),size_t(7)})
            )
            .def("__call__",
                [](
                    const Op & op,
                    const CellGeometryVector<2,1> &  cell1GeometryVector
                ){
                    const auto nFeatures = size_t(op.numberOfFeatures());
                    const auto nCells1   = size_t(cell1GeometryVector.size());

                    nifty::marray::PyView<float> out({nCells1, nFeatures});

                    op(cell1GeometryVector, out);

                    return  out;
                },
                py::arg("cell1GeometryVector")
            )
            ;
        }
        {
            typedef Cell1BasicGeometricFeatures2D Op;
            const auto clsName = std::string("Cell1BasicGeometricFeatures2D");
            auto pyCls = py::class_<Op>(module, clsName.c_str());

            pyCls
            .def(
                py::init< const std::vector<size_t> &>(),
                py::arg("dists")  =  std::vector<size_t>({size_t(3),size_t(5),size_t(7)})
            )
            .def("__call__",
                [](
                    const Op & op,
                    const CellGeometryVector<2,0>   & cell0GeometryVector,
                    const CellGeometryVector<2,1>   & cell1GeometryVector,
                    const CellGeometryVector<2,2>   & cell2GeometryVector,
                    const CellBoundsVector<2,0>     & cell0BoundsVector,
                    const CellBoundsVector<2,1>     & cell1BoundsVector,
                    const CellBoundedByVector<2,1>  & cell1BoundedByVector,
                    const CellBoundedByVector<2,2>  & cell2BoundedByVector
                ){
                    const auto nFeatures = size_t(op.numberOfFeatures());
                    const auto nCells1   = size_t(cell1GeometryVector.size());

                    nifty::marray::PyView<float> out({nCells1, nFeatures});

                    op( cell0GeometryVector,
                        cell1GeometryVector,
                        cell2GeometryVector,
                        cell0BoundsVector,
                        cell1BoundsVector,
                        cell1BoundedByVector,
                        cell2BoundedByVector,
                        out
                    );

                    return  out;
                },
                py::arg("cell0GeometryVector"),
                py::arg("cell1GeometryVector"),
                py::arg("cell2GeometryVector"),
                py::arg("cell0BoundsVector"),
                py::arg("cell1BoundsVector"),
                py::arg("cell1BoundedByVector"),
                py::arg("cell2BoundedByVector")

            )
            ;
        }

    }

}
}
