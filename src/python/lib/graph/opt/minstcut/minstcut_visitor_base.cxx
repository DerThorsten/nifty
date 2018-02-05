#include <memory>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>


#include "nifty/graph/opt/minstcut/minstcut_base.hxx"
#include "nifty/graph/opt/minstcut/minstcut_visitor_base.hxx"

#include "nifty/python/graph/undirected_list_graph.hxx"
#include "nifty/python/graph/edge_contraction_graph.hxx"
#include "nifty/python/graph/opt/minstcut/minstcut_objective.hxx"
#include "nifty/python/graph/opt/minstcut/py_minstcut_visitor_base.hxx"

#include "nifty/python/converter.hxx"





namespace py = pybind11;


namespace nifty{
namespace graph{
namespace opt{
namespace minstcut{


    using namespace py;
    //PYBIND11_DECLARE_HOLDER_TYPE(McBase, std::shared_ptr<McBase>);

    template<class OBJECTIVE>
    void exportMinstcutVisitorBaseT(py::module & module) {

        typedef OBJECTIVE ObjectiveType;
        typedef PyMinstcutVisitorBase<ObjectiveType> PyVisitorBase;
        typedef MinstcutVisitorBase<ObjectiveType> VisitorBase;

        
        const auto objName = MinstcutObjectiveName<ObjectiveType>::name();
        const auto visitorBaseClsName = std::string("MinstcutVisitorBase") + objName;
        const auto mcVerboseVisitorClsName = std::string("MinstcutVerboseVisitor") + objName;

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
            typedef MinstcutVerboseVisitor<ObjectiveType> VisitorType; 
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

    void exportMinstcutVisitorBase(py::module & module) {

        {
            typedef PyUndirectedGraph GraphType;
            typedef MinstcutObjective<GraphType, double> ObjectiveType;
            exportMinstcutVisitorBaseT<ObjectiveType>(module);
        }
        {
            typedef PyContractionGraph<PyUndirectedGraph> GraphType;
            typedef MinstcutObjective<GraphType, double> ObjectiveType;
            exportMinstcutVisitorBaseT<ObjectiveType>(module);
        }
    }      
} // namespace nifty::graph::opt::minstcut
} // namespace nifty::graph::opt
}
}
    
