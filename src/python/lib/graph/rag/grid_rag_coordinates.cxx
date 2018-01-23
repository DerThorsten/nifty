#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "nifty/graph/rag/grid_rag.hxx"
#include "nifty/graph/rag/grid_rag_coordinates.hxx"

#include "xtensor-python/pytensor.hpp"

namespace py = pybind11;

namespace nifty{
namespace graph{

    using namespace py;

    template<size_t DIM, class RAG_TYPE>
    void exportGridRagCoordinatesT(py::module & module, const std::string & name) {

        typedef RagCoordinates<DIM, RAG_TYPE> CoordinatesType;
        typedef typename CoordinatesType::RagType RagType;
        typedef typename CoordinatesType::Coord Coord;

        std::string className = "RagCoordinates" + name;
        std::string factoryName = "coordinatesFactory" + name;

        py::class_<CoordinatesType>(module, className.c_str())

            .def("topologicalEdgeCoordinates", [](const CoordinatesType & self, const int64_t edgeId){
                const auto & coords = self.edgeCoordinates(edgeId);
                size_t nCoordinates = coords.size() / DIM;
                xt::pytensor<int32_t, DIM> out({(int64_t) nCoordinates, DIM});
                size_t jj = 0;
                for(size_t ii = 0; ii < nCoordinates; ++ii) {
                    for(size_t d = 0; d < DIM; ++d) {
                        out(ii,d) = coords[jj];
                        ++jj;
                    }
                }
                return out;
            })

            .def("edgeCoordinates", [](const CoordinatesType & self, const int64_t edgeId){
                const auto & coords = self.edgeCoordinates(edgeId);
                size_t nCoordinates = 2 * coords.size() / DIM;
                xt::pytensor<int32_t, DIM> out({(int64_t) nCoordinates, DIM});
                size_t jj = 0;
                for(size_t ii = 0; ii < nCoordinates / 2; ++ii) {
                    for(size_t d = 0; d < DIM; ++d) {
                        out(2*ii, d)   = std::floor(coords[jj] / 2);
                        out(2*ii+1, d) = std::ceil( coords[jj] / 2);
                        ++jj;
                    }
                }
                return out;
            })

            .def("edgesToVolume", [](
                    const CoordinatesType & self,
                    const std::vector<uint32_t> & edgeValues,
                    const int edgeDirection,
                    const uint32_t ignoreValue,
                    const int numberOfThreads) {

                typedef typename xt::pytensor<uint32_t, DIM>::shape_type ShapeType;
                ShapeType shape;
                std::copy(self.rag().shape().begin(), self.rag().shape().end(), shape.begin());
                xt::pytensor<uint32_t, DIM> out = xt::zeros<uint32_t>(shape);

                self.edgesToVolume(edgeValues, out, edgeDirection, ignoreValue, numberOfThreads);
                return out;
            }, py::arg("edgeValues"), py::arg("edgeDirection") = 0, py::arg("ignoreValue") = 0, py::arg("numberOfThreads") = -1)

            .def("edgesToSubVolume", [](
                    const CoordinatesType & self,
                    const std::vector<uint32_t> & edgeValues,
                    const std::vector<int64_t> & begin,
                    const std::vector<int64_t> & end,
                    const int edgeDirection,
                    const uint32_t ignoreValue,
                    const int numberOfThreads) {

                typedef typename xt::pytensor<uint32_t, DIM>::shape_type ShapeType;
                ShapeType shape;
                for(int d = 0; d < DIM; ++d) {
                    shape[d] = end[d] - begin[d];
                }
                xt::pytensor<uint32_t, DIM> out(shape);

                self.edgesToSubVolume(edgeValues, out, begin, end, edgeDirection, ignoreValue, numberOfThreads);
                return out;
            }, py::arg("edgeValues"), py::arg("begin"), py::arg("end"), py::arg("edgeDirection") = 0, py::arg("ignoreValue") = 0, py::arg("numberOfThreads") = -1)

            .def("storageLengths", &CoordinatesType::storageLengths)
        ;

        module.def(factoryName.c_str(),
        [](
           const RagType & rag,
           const int numberOfThreads
        ){
            auto ptr = new CoordinatesType(rag, numberOfThreads);
            return ptr;
        },
        py::return_value_policy::take_ownership,
        py::keep_alive<0, 1>(),
        py::arg("rag"), py::arg("numberOfThreads") = -1
        );

    }

    void exportGridRagCoordinates(py::module & module) {
        typedef xt::pytensor<uint32_t, 2> ExplicitPyLabels2D;
        typedef GridRag<2, ExplicitPyLabels2D> Rag2d;

        typedef xt::pytensor<uint32_t, 3> ExplicitPyLabels3D;
        typedef GridRag<3, ExplicitPyLabels3D> Rag3d;

        exportGridRagCoordinatesT<2, Rag2d>(module, "Explicit2d");
        exportGridRagCoordinatesT<3, Rag3d>(module, "Explicit3d");

        // hdf5
        #ifdef WITH_HDF5
        {
            typedef nifty::hdf5::Hdf5Array<uint32_t> LabelsUInt32;
            typedef GridRagStacked2D<LabelsUInt32> StackedRagUInt32;
            exportGridRagCoordinatesT<3, StackedRagUInt32>(module, "StackedRag3d");
        }
        #endif
    }

}
}
