#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "nifty/graph/multicut/multicut_objective.hxx"
#include "nifty/graph/simple_graph.hxx"
#include "nifty/graph/multicut/multicut_base.hxx"

#include "../../converter.hxx"
#include "py_multicut_visitor_base.hxx"




namespace py = pybind11;


namespace nifty{
namespace graph{

    typedef UndirectedGraph<> Graph;
    typedef MulticutObjective<Graph, double> Objective;

    using namespace py;
    //PYBIND11_DECLARE_HOLDER_TYPE(McBase, std::shared_ptr<McBase>);

    void exportMulticutVisitorBase(py::module & multicutModule) {

        typedef UndirectedGraph<> Graph;
        typedef MulticutObjective<Graph, double> Objective;
        typedef PyMulticutVisitorBase<Objective> PyMcVisitorBase;
        typedef MulticutVisitorBase<Objective> McVisitorBase;

        //PYBIND11_DECLARE_HOLDER_TYPE(McBase, std::shared_ptr<McBase>);

        // base factory
        py::class_<
            McVisitorBase, 
            std::unique_ptr<McVisitorBase>, 
            PyMcVisitorBase 
        > mcVisitorBase(multicutModule, "MulticutVisitorBaseUndirectedGraph");
        
        mcVisitorBase
        ;


        // concrete visitors
        

        typedef MulticutVerboseVisitor<Objective> McVerboseVisitor; 
        
        py::class_<McVerboseVisitor, std::unique_ptr<McVerboseVisitor> >(multicutModule, "MulticutVerboseVisitorUndirectedGraph",  mcVisitorBase)
            .def(py::init<const int >(),
                py::arg_t<int>("printNth",1)
            )
            .def("stopOptimize",&McVerboseVisitor::stopOptimize)
        ;

    }

}
}
    
