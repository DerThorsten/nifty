#include <pybind11/pybind11.h>

#include <boost/algorithm/string.hpp>


// concrete solvers for concrete factories
#include "nifty/graph/opt/ho_multicut/ho_multicut_dual_decomposition.hxx"

#ifdef WITH_GUROBI
#include "nifty/ilp_backend/gurobi.hxx"
#endif

#ifdef WITH_CPLEX
#include "nifty/ilp_backend/cplex.hxx"
#endif

#ifdef WITH_GLPK
#include "nifty/ilp_backend/glpk.hxx"
#endif

#include "nifty/python/graph/undirected_list_graph.hxx"
#include "nifty/python/graph/edge_contraction_graph.hxx"
#include "nifty/python/graph/opt/ho_multicut/ho_multicut_objective.hxx"
#include "nifty/python/converter.hxx"
#include "nifty/python/graph/opt/solver_docstring.hxx"
#include "nifty/python/graph/opt/ho_multicut/export_ho_multicut_solver.hxx"

namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

namespace nifty{
namespace graph{
namespace opt{
namespace ho_multicut{
    
    template<class OBJECTIVE>
    void exportHoMulticutDualDecompositionT(py::module & hoMulticutModule){






        ///////////////////////////////////////////////////////////////
        // DOCSTRING HELPER
        ///////////////////////////////////////////////////////////////
        nifty::graph::opt::SolverDocstringHelper docHelper;
        docHelper.objectiveName = "ho multicut objective"; 
        docHelper.objectiveClsName = HoMulticutObjectiveName<OBJECTIVE>::name();
        docHelper.name = "ho multicut dual decomposition"; 
        docHelper.mainText =  
            "todo";
        // docHelper.cites.emplace_back("Kappes-2011");
        // docHelper.cites.emplace_back("andres_2011_probabilistic");
        docHelper.note = "This might take very long for large models.";
        // docHelper.requirements.emplace_back(
        //     std::string("WITH_") + boost::to_upper_copy<std::string>(backendName)
        // );


        
        typedef OBJECTIVE ObjectiveType;
        typedef HoMulticutDualDecomposition<ObjectiveType> Solver;
        
        typedef typename Solver::SettingsType SettingsType;        
        const auto solverName = std::string("HoMulticutDualDecomposition");
        // todo exportHoMulticutSolver should be in the correct namespace

        auto settingsCls = nifty::graph::opt::ho_multicut::exportHoMulticutSolver<Solver>(hoMulticutModule, solverName.c_str(), docHelper)

            .def(py::init<>())
            .def_readwrite("numberOfIterations", &SettingsType::numberOfIterations)
            .def_readwrite("stepSize", &SettingsType::stepSize)
            .def_readwrite("submodelMcFactory", &SettingsType::submodelMcFactory)
            .def_readwrite("crfSolver", &SettingsType::crf_solver)
            .def_readwrite("absoluteGap", &SettingsType::absoluteGap)
            .def_readwrite("fusionMoveSettings", &SettingsType::fusionMoveSetting)
        ; 

        py::enum_<typename SettingsType::crf_solver_type>(settingsCls, "crf_solver_type")
            .value("graphcut",  SettingsType::crf_solver_type::graphcut)
            .value("qpbo",      SettingsType::crf_solver_type::qpbo)
            .export_values()
        ;
    }

    void exportHoMulticutDualDecomposition(py::module & hoMulticutModule){

        
        py::options options;
        options.disable_function_signatures();
                


        {
            typedef PyUndirectedGraph GraphType;
            typedef HoMulticutObjective<GraphType, double> ObjectiveType;
            exportHoMulticutDualDecompositionT<ObjectiveType>(hoMulticutModule);
        }  
    }
} // namespace nifty::graph::opt::multicut
} // namespace nifty::graph::opt
} // namespace nifty::graph
} // namespace nifty