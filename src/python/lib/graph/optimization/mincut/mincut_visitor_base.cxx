#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>


#include "nifty/graph/optimization/mincut/mincut_base.hxx"
#include "nifty/graph/optimization/mincut/mincut_visitor_base.hxx"

#include "nifty/python/graph/undirected_list_graph.hxx"
#include "nifty/python/graph/edge_contraction_graph.hxx"
#include "nifty/python/graph/optimization/mincut/mincut_objective.hxx"
#include "nifty/python/graph/optimization/mincut/py_mincut_visitor_base.hxx"

#include "nifty/python/converter.hxx"





namespace py = pybind11;


namespace nifty{
namespace graph{



    using namespace py;
    //PYBIND11_DECLARE_HOLDER_TYPE(McBase, std::shared_ptr<McBase>);

    template<class OBJECTIVE>
    void exportMincutVisitorBaseT(py::module & mincutModule) {

        typedef OBJECTIVE ObjectiveType;
        typedef PyMincutVisitorBase<ObjectiveType> PyMcVisitorBase;
        typedef MincutVisitorBase<ObjectiveType> McVisitorBase;

        
        const auto objName = MincutObjectiveName<ObjectiveType>::name();
        const auto mcVisitorBaseClsName = std::string("MincutVisitorBase") + objName;
        const auto mcVerboseVisitorClsName = std::string("MincutVerboseVisitor") + objName;

        // base factory
        py::class_<
            McVisitorBase, 
            std::unique_ptr<McVisitorBase>, 
            PyMcVisitorBase 
        > mcVisitorBase(mincutModule, mcVisitorBaseClsName.c_str());
        
        //mcVisitorBase
        //;


        // concrete visitors
        

        typedef MincutVerboseVisitor<ObjectiveType> McVerboseVisitor; 
        
        py::class_<McVerboseVisitor, std::unique_ptr<McVerboseVisitor> >(mincutModule, mcVerboseVisitorClsName.c_str(),  mcVisitorBase)
            .def(py::init<const int, const double>(),
                py::arg_t<int>("printNth",1),
                py::arg_t<double>("timeLimit",std::numeric_limits<double>::infinity())
            )
            .def("stopOptimize",&McVerboseVisitor::stopOptimize)
        ;
    }

    void exportMincutVisitorBase(py::module & mincutModule) {

        {
            typedef PyUndirectedGraph GraphType;
            typedef MincutObjective<GraphType, double> ObjectiveType;
            exportMincutVisitorBaseT<ObjectiveType>(mincutModule);
        }
        {
            typedef PyContractionGraph<PyUndirectedGraph> GraphType;
            typedef MincutObjective<GraphType, double> ObjectiveType;
            exportMincutVisitorBaseT<ObjectiveType>(mincutModule);
        }
    }      


}
}
    
