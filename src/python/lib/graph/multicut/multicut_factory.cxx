#include <pybind11/pybind11.h>


#include "nifty/python/graph/undirected_list_graph.hxx"
#include "nifty/python/graph/edge_contraction_graph.hxx"
#include "nifty/python/graph/multicut/multicut_objective.hxx"


#include "nifty/python/converter.hxx"
#include "nifty/python/graph/multicut/py_multicut_factory.hxx"





namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(McBase, std::shared_ptr<McBase>);

namespace nifty{
namespace graph{



    template<class OBJECTIVE>
    void exportMulticutFactoryT(py::module & multicutModule) {

        typedef OBJECTIVE ObjectiveType;
        typedef PyMulticutFactoryBase<ObjectiveType> PyMcFactoryBase;
        typedef MulticutFactoryBase<ObjectiveType> McFactoryBase;


        const auto objName = MulticutObjectiveName<ObjectiveType>::name();
        const auto clsName = std::string("MulticutFactoryBase") + objName;

        // base factory
        py::class_<
            McFactoryBase, 
            std::shared_ptr<McFactoryBase>, 
            PyMcFactoryBase 
        > mcFactoryBase(multicutModule, clsName.c_str());
        
        mcFactoryBase
            .def(py::init<>())

            .def("create", 
                //&McFactoryBase::create,
                [](McFactoryBase * self, const ObjectiveType & obj){
                    return self->createRawPtr(obj);
                },
                //,
                py::return_value_policy::take_ownership,
                py::keep_alive<0,2>()
                )
        ;

    }

    void exportMulticutFactory(py::module & multicutModule) {
        {
            typedef PyUndirectedGraph GraphType;
            typedef MulticutObjective<GraphType, double> ObjectiveType;
            exportMulticutFactoryT<ObjectiveType>(multicutModule);
        }
        {
            typedef PyContractionGraph<PyUndirectedGraph> GraphType;
            typedef MulticutObjective<GraphType, double> ObjectiveType;
            exportMulticutFactoryT<ObjectiveType>(multicutModule);
        }
    }

}
}
