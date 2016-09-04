#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>


#include "nifty/graph/optimization/multicut/multicut_base.hxx"
#include "nifty/graph/optimization/multicut/multicut_visitor_base.hxx"

#include "nifty/python/graph/undirected_list_graph.hxx"
#include "nifty/python/graph/edge_contraction_graph.hxx"
#include "nifty/python/graph/optimization/multicut/multicut_objective.hxx"
#include "nifty/python/graph/optimization/multicut/py_multicut_visitor_base.hxx"

#include "nifty/python/converter.hxx"





namespace py = pybind11;


namespace nifty{
namespace graph{



    using namespace py;
    //PYBIND11_DECLARE_HOLDER_TYPE(McBase, std::shared_ptr<McBase>);

    template<class OBJECTIVE>
    void exportMulticutVisitorBaseT(py::module & multicutModule) {

        typedef OBJECTIVE ObjectiveType;
        typedef PyMulticutVisitorBase<ObjectiveType> PyMcVisitorBase;
        typedef MulticutVisitorBase<ObjectiveType> McVisitorBase;

        
        const auto objName = MulticutObjectiveName<ObjectiveType>::name();
        const auto mcVisitorBaseClsName = std::string("MulticutVisitorBase") + objName;
        const auto mcVerboseVisitorClsName = std::string("MulticutVerboseVisitor") + objName;

        // base factory
        py::class_<
            McVisitorBase, 
            std::unique_ptr<McVisitorBase>, 
            PyMcVisitorBase 
        > mcVisitorBase(multicutModule, mcVisitorBaseClsName.c_str());
        
        mcVisitorBase
        ;


        // concrete visitors
        

        typedef MulticutVerboseVisitor<ObjectiveType> McVerboseVisitor; 
        
        py::class_<McVerboseVisitor, std::unique_ptr<McVerboseVisitor> >(multicutModule, mcVerboseVisitorClsName.c_str(),  mcVisitorBase)
            .def(py::init<const int >(),
                py::arg_t<int>("printNth",1)
            )
            .def("stopOptimize",&McVerboseVisitor::stopOptimize)
        ;
    }

    void exportMulticutVisitorBase(py::module & multicutModule) {

        {
            typedef PyUndirectedGraph GraphType;
            typedef MulticutObjective<GraphType, double> ObjectiveType;
            exportMulticutVisitorBaseT<ObjectiveType>(multicutModule);
        }
        {
            typedef PyContractionGraph<PyUndirectedGraph> GraphType;
            typedef MulticutObjective<GraphType, double> ObjectiveType;
            exportMulticutVisitorBaseT<ObjectiveType>(multicutModule);
        }
    }      


}
}
    
