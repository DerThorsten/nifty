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
        typedef MulticutObjective<Graph, double> Objective;
        typedef PyMulticutFactoryBase<Objective> PyMcFactoryBase;
        typedef MulticutFactoryBase<Objective> McFactoryBase;


        // base factory
        py::class_<
            McFactoryBase, 
            std::unique_ptr<McFactoryBase>, 
            PyMcFactoryBase 
        > mcFactoryBase(multicutModule, "MulticutFactoryBaseUndirectedGraph");
        
        mcFactoryBase
            .def(py::init<>())

            .def("create", 
                //&McFactoryBase::create,
                [](McFactoryBase * self, const Objective & obj){
                    return self->createRawPtr(obj);
                },
                //,
                py::return_value_policy::take_ownership,
                py::keep_alive<0,2>()
                )
        ;

    }

}
}
