#include <pybind11/pybind11.h>
#include "nifty/graph/multicut/multicut_objective.hxx"
#include "nifty/graph/simple_graph.hxx"

#include "../../converter.hxx"
#include "py_multicut_factory.hxx"

namespace py = pybind11;


namespace nifty{
namespace graph{



    void exportMulticutFactory(py::module & multicutModule) {

        typedef UndirectedGraph<> Graph;
        typedef MulticutObjective<Graph, double> McObjective;
        typedef PyMulticutFactoryBase<McObjective> PyMcFactoryBase;
        typedef MulticutFactoryBase<McObjective> McFactoryBase;

        py::class_<
            McFactoryBase, 
            std::unique_ptr<McFactoryBase>, 
            PyMcFactoryBase 
        > mcFactoryBase(multicutModule, "MulticutFactoryBase");
        
        mcFactoryBase
            .def(py::init<>())
            //.def("go", &Animal::go);
        ;



    }

}
}
