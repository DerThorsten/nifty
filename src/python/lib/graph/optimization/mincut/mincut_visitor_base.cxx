#include <memory>

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
namespace optimization{
namespace mincut{


    using namespace py;
    //PYBIND11_DECLARE_HOLDER_TYPE(McBase, std::shared_ptr<McBase>);

    template<class OBJECTIVE>
    void exportMincutVisitorBaseT(py::module & module) {

        typedef OBJECTIVE ObjectiveType;
        typedef PyMincutVisitorBase<ObjectiveType> PyVisitorBase;
        typedef MincutVisitorBase<ObjectiveType> VisitorBase;

        
        const auto objName = MincutObjectiveName<ObjectiveType>::name();
        const auto visitorBaseClsName = std::string("MincutVisitorBase") + objName;
        const auto mcVerboseVisitorClsName = std::string("MincutVerboseVisitor") + objName;

        // base factory
        py::class_<
            VisitorBase, 
            std::unique_ptr<VisitorBase>, 
            PyVisitorBase 
        > visitorBase(module, visitorBaseClsName.c_str());
        
        //visitorBase
        //;


        // concrete visitors
        

        {
            const auto visitorClsName = std::string("VerboseVisitor") + objName;
            typedef MincutVerboseVisitor<ObjectiveType> VisitorType; 
            py::class_<VisitorType, std::unique_ptr<VisitorType> >(module, visitorClsName.c_str(),  visitorBase)
                .def(py::init<const int, const double , const double>(),
                    py::arg_t<int>("visitNth",1),
                    py::arg_t<double>("timeLimitSolver",std::numeric_limits<double>::infinity()),
                    py::arg_t<double>("timeLimitTotal",std::numeric_limits<double>::infinity())
                )
                .def("stopOptimize",&VisitorType::stopOptimize)
                .def_property_readonly("timeLimitSolver", &VisitorType::timeLimitSolver)
                .def_property_readonly("timeLimitTotal", &VisitorType::timeLimitTotal)
                .def_property_readonly("runtimeSolver", &VisitorType::runtimeSolver)
                .def_property_readonly("runtimeTotal", &VisitorType::runtimeTotal)
            ;
        }



    }

    void exportMincutVisitorBase(py::module & module) {

        {
            typedef PyUndirectedGraph GraphType;
            typedef MincutObjective<GraphType, double> ObjectiveType;
            exportMincutVisitorBaseT<ObjectiveType>(module);
        }
        {
            typedef PyContractionGraph<PyUndirectedGraph> GraphType;
            typedef MincutObjective<GraphType, double> ObjectiveType;
            exportMincutVisitorBaseT<ObjectiveType>(module);
        }
    }      
} // namespace nifty::graph::optimization::mincut
} // namespace nifty::graph::optimization
}
}
    
