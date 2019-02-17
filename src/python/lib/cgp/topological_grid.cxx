#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "xtensor-python/pytensor.hpp"
#include "xtensor-python/pyarray.hpp"

#include "nifty/python/converter.hxx"
#include "nifty/cgp/topological_grid.hxx"
#include "nifty/cgp/filled_topological_grid.hxx"

namespace py = pybind11;


namespace nifty{
namespace cgp{

    template<std::size_t DIM>
    void exportTopologicalGridT(py::module & m) {

        typedef TopologicalGrid<DIM> TopologicalGrid;
        const std::string clsName = std::string("TopologicalGrid")+std::to_string(DIM)+std::string("D");
        py::class_<TopologicalGrid>(m, clsName.c_str())
            .def(py::init([](xt::pytensor<uint32_t, DIM> labels){
                return new TopologicalGrid(labels);
            }))
            .def_property_readonly("shape", &TopologicalGrid::shape)
            .def_property_readonly("topologicalGridShape", &TopologicalGrid::topologicalGridShape)
            .def_property_readonly("numberOfCells", &TopologicalGrid::numberOfCells)

            // TODO for this we need to export a reference to the
            // topologicalGridArray (which is a xt::xtensor<uint32_t, 2> type) as pytensor ref
            /*
            .def("_gridView",[](const TopologicalGrid & self){
               const auto & topologicalGridArray = self.array();
               xt::pytensor<uint32_t, 2> numpyArray;
               numpyArray.createViewFrom(topologicalGridArray);
               return numpyArray;
            },py::keep_alive<0, 1>());
            */
        ;
    }

    template<std::size_t DIM>
    void exportFilledTopologicalGridT(py::module & m) {

        typedef TopologicalGrid<DIM> TopologicalGrid;
        typedef FilledTopologicalGrid<DIM> FilledTopologicalGrid;
        const std::string clsName = std::string("FilledTopologicalGrid")+std::to_string(DIM)+std::string("D");
        py::class_<FilledTopologicalGrid>(m, clsName.c_str())
            .def(py::init([](const TopologicalGrid & tGrid){
                return new FilledTopologicalGrid(tGrid);
            }))

            .def_property_readonly("numberOfCells",[](const FilledTopologicalGrid & self){
                return self.numberOfCells();
            })
            .def_property_readonly("cellTypeOffset", &FilledTopologicalGrid::cellTypeOffset)

            // TODO for this we need to export a reference to the
            // topologicalGridArray (which is a xt::xtensor<uint32_t, 2> type) as pytensor ref
            /*
            .def("_gridView",[](const FilledTopologicalGrid & self){
               const auto & topologicalGridArray = self.array();
               xt::pyarray<uint32_t> numpyArray;
               numpyArray.createViewFrom(topologicalGridArray);
               return numpyArray;
            }, py::keep_alive<0, 1>());
            */
        ;
    }

    void exportTopologicalGrid(py::module & m) {

        exportTopologicalGridT<2>(m);
        exportFilledTopologicalGridT<2>(m);
    }

}
}
