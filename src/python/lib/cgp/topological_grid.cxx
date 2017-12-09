#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
//#include <pybind11/stl.h>

#include "nifty/python/converter.hxx"
#include "nifty/cgp/topological_grid.hxx"
#include "nifty/cgp/filled_topological_grid.hxx"

namespace py = pybind11;



namespace nifty{
namespace cgp{

    template<size_t DIM>
    void exportTopologicalGridT(py::module & m) {

        typedef TopologicalGrid<DIM> TopologicalGrid;
        const std::string clsName = std::string("TopologicalGrid")+std::to_string(DIM)+std::string("D");
        py::class_<TopologicalGrid>(m, clsName.c_str())
            .def(py::init([](nifty::marray::PyView<uint32_t, DIM> labels){
                return new TopologicalGrid(labels);
            }))
            .def_property_readonly("shape", &TopologicalGrid::shape)
            .def_property_readonly("topologicalGridShape", &TopologicalGrid::topologicalGridShape)
            .def_property_readonly("numberOfCells", &TopologicalGrid::numberOfCells)

            .def("_gridView",[](const TopologicalGrid & self){
               const auto & topologicalGridArray = self.array();
               //const nifty::marray::View<uint32_t> & view = static_cast<const nifty::marray::View<uint32_t> & >(topologicalGridArray);
               nifty::marray::PyView<uint32_t> numpyArray;
               numpyArray.createViewFrom(topologicalGridArray);
               return numpyArray;
            }
            ,
            py::keep_alive<0, 1>()
            );
        ;
    }

    template<size_t DIM>
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
            .def("_gridView",[](const FilledTopologicalGrid & self){
               const auto & topologicalGridArray = self.array();
               //const nifty::marray::View<uint32_t> & view = static_cast<const nifty::marray::View<uint32_t> & >(topologicalGridArray);
               nifty::marray::PyView<uint32_t> numpyArray;
               numpyArray.createViewFrom(topologicalGridArray);
               return numpyArray;
            }
            ,
            py::keep_alive<0, 1>()
            );
        ;
    }

    void exportTopologicalGrid(py::module & m) {

        exportTopologicalGridT<2>(m);
        exportFilledTopologicalGridT<2>(m);
    }

}
}
