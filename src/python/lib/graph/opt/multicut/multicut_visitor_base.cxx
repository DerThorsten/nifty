#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "xtensor-python/pytensor.hpp"


#include "nifty/graph/opt/multicut/multicut_base.hxx"
#include "nifty/graph/opt/multicut/multicut_visitor_base.hxx"

#include "nifty/graph/opt/multicut/multicut_visitor_base.hxx"


#include "nifty/python/graph/undirected_list_graph.hxx"
#include "nifty/python/graph/edge_contraction_graph.hxx"
#include "nifty/python/graph/opt/multicut/multicut_objective.hxx"
#include "nifty/python/graph/opt/multicut/py_multicut_visitor_base.hxx"
#include "nifty/graph/opt/common/logging_visitor.hxx"


namespace py = pybind11;


namespace nifty{
namespace graph{
namespace opt{
namespace multicut{


    using namespace py;
    //PYBIND11_DECLARE_HOLDER_TYPE(McBase, std::shared_ptr<McBase>);

    template<class OBJECTIVE>
    void exportMulticutVisitorBaseT(py::module & module) {

        typedef OBJECTIVE                               ObjectiveType;
        typedef MulticutBase<ObjectiveType>             SolverBaseType;
        typedef PyMulticutVisitorBase<ObjectiveType>    PyVisitorBaseType;
        typedef MulticutVisitorBase<ObjectiveType>      VisitorBaseType;

        const auto objName = MulticutObjectiveName<ObjectiveType>::name();
        const auto visitorBaseClsName = std::string("VisitorBase") + objName;

        // base factory
        py::class_<
            VisitorBaseType,
            std::unique_ptr<VisitorBaseType>,
            PyVisitorBaseType
        > visitorBase(module, visitorBaseClsName.c_str());

        // concrete visitors
        {
            const auto visitorClsName = std::string("VerboseVisitor") + objName;
            typedef MulticutVerboseVisitor<ObjectiveType> VisitorType;
            py::class_<VisitorType, std::unique_ptr<VisitorType> >(module, visitorClsName.c_str(),  visitorBase)
                .def(py::init<const int, const double , const double, const nifty::logging::LogLevel>(),
                    py::arg("visitNth")=1,
                    py::arg("timeLimitSolver") = std::numeric_limits<double>::infinity(),
                    py::arg("timeLimitTotal") = std::numeric_limits<double>::infinity(),
                    py::arg("logLevel") = nifty::logging::LogLevel::WARN
                )
                .def("stopOptimize",&VisitorType::stopOptimize)
                .def_property_readonly("timeLimitSolver", &VisitorType::timeLimitSolver)
                .def_property_readonly("timeLimitTotal", &VisitorType::timeLimitTotal)
                .def_property_readonly("runtimeSolver", &VisitorType::runtimeSolver)
                .def_property_readonly("runtimeTotal", &VisitorType::runtimeTotal)
            ;
        }


        {
            const auto visitorName = std::string("LoggingVisitor") + objName;
            typedef nifty::graph::opt::common::LoggingVisitor<SolverBaseType> VisitorType;

            py::class_<VisitorType, std::unique_ptr<VisitorType> >(module, visitorName.c_str(),  visitorBase)
                .def(py::init<const int, const double, const double,const nifty::logging::LogLevel>(),
                    py::arg_t<int>("visitNth",1),
                    py::arg_t<double>("timeLimitSolver",std::numeric_limits<double>::infinity()),
                    py::arg_t<double>("timeLimitTotal",std::numeric_limits<double>::infinity()),
                    py::arg("logLevel") = nifty::logging::LogLevel::WARN
                )
                .def("stopOptimize",&VisitorType::stopOptimize)

                // logging
                .def("iterations",[](const VisitorType & visitor){
                    const auto vec = visitor.iterations();
                    typedef typename xt::pytensor<uint32_t, 1>::shape_type ShapeType;
                    ShapeType shape = {static_cast<int64_t>(vec.size())};
                    xt::pytensor<uint32_t, 1> ret = xt::zeros<uint32_t>(shape);
                    for(auto i=0; i<vec.size(); ++i)
                        ret[i] = vec[i];
                    return ret;
                })
                .def("energies",[](const VisitorType & visitor){
                    const auto vec = visitor.energies();
                    typedef typename xt::pytensor<double, 1>::shape_type ShapeType;
                    ShapeType shape = {static_cast<int64_t>(vec.size())};
                    xt::pytensor<double, 1> ret = xt::zeros<double>(shape);
                    for(auto i=0; i<vec.size(); ++i)
                        ret[i] = vec[i];
                    return ret;
                })
                .def("runtimes",[](const VisitorType & visitor){
                    const auto vec = visitor.runtimes();
                    typedef typename xt::pytensor<double, 1>::shape_type ShapeType;
                    ShapeType shape = {static_cast<int64_t>(vec.size())};
                    xt::pytensor<double, 1> ret = xt::zeros<double>(shape);
                    for(auto i=0; i<vec.size(); ++i)
                        ret[i] = vec[i];
                    return ret;
                })
            ;

        }


    }

    void exportMulticutVisitorBase(py::module & module) {

        {
            typedef PyUndirectedGraph GraphType;
            typedef MulticutObjective<GraphType, double> ObjectiveType;
            exportMulticutVisitorBaseT<ObjectiveType>(module);
        }
        {
            typedef PyContractionGraph<PyUndirectedGraph> GraphType;
            typedef MulticutObjective<GraphType, double> ObjectiveType;
            exportMulticutVisitorBaseT<ObjectiveType>(module);
        }
    }      
} // namespace nifty::graph::opt::multicut
} // namespace nifty::graph::opt
}
}
    
