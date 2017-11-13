#include <pybind11/pybind11.h>


#include "nifty/python/graph/opt/lifted_multicut/export_lifted_multicut_solver.hxx"
#include "nifty/python/converter.hxx"


#ifdef WITH_GUROBI
#include "nifty/ilp_backend/gurobi.hxx"
#endif

#ifdef WITH_CPLEX
#include "nifty/ilp_backend/cplex.hxx"
#endif

#ifdef WITH_GLPK
#include "nifty/ilp_backend/glpk.hxx"
#endif

#include "nifty/python/graph/undirected_grid_graph.hxx"
#include "nifty/python/graph/undirected_list_graph.hxx"
#include "nifty/python/graph/edge_contraction_graph.hxx"
#include "nifty/python/graph/opt/lifted_multicut/lifted_multicut_objective.hxx"
#include "nifty/graph/opt/lifted_multicut/lifted_multicut_ilp.hxx"

namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

namespace nifty{
namespace graph{
namespace opt{
namespace lifted_multicut{


    template<class OBJECTIVE, class BACKEND>
    void exportLiftedMulticutIlpWithBackendT(py::module & liftedMulticutModule, const std::string & backendName){
        typedef OBJECTIVE ObjectiveType;
        typedef BACKEND IlpSolver;
        typedef LiftedMulticutIlp<ObjectiveType, IlpSolver> Solver;
        typedef typename Solver::SettingsType SettingsType;
        const auto solverName = std::string("LiftedMulticutIlp") + backendName;
        exportLiftedMulticutSolver<Solver>(liftedMulticutModule, solverName.c_str())
            .def(py::init<>())
            .def_readwrite("numberOfIterations", &SettingsType::numberOfIterations)
            .def_readwrite("verbose", &SettingsType::verbose)
            .def_readwrite("verboseIlp", &SettingsType::verboseIlp)
            .def_readwrite("addThreeCyclesConstraints", &SettingsType::addThreeCyclesConstraints)
            .def_readwrite("addOnlyViolatedThreeCyclesConstraints", &SettingsType::addOnlyViolatedThreeCyclesConstraints)
            .def_readwrite("ilpSettings",&SettingsType::ilpSettings)
        ; 
    }

    template<class OBJECTIVE>
    void exportLiftedMulticutIlpT(py::module & liftedMulticutModule) {
        typedef OBJECTIVE ObjectiveType;
        #ifdef WITH_CPLEX
            exportLiftedMulticutIlpWithBackendT<ObjectiveType, ilp_backend::Cplex>(liftedMulticutModule, "Cplex");
        #endif
        #ifdef WITH_GUROBI
            exportLiftedMulticutIlpWithBackendT<ObjectiveType, ilp_backend::Gurobi>(liftedMulticutModule, "Gurobi");
        #endif
        #ifdef WITH_GLPK
            exportLiftedMulticutIlpWithBackendT<ObjectiveType, ilp_backend::Glpk>(liftedMulticutModule, "Glpk");
        #endif   
        
    }
    
    void exportLiftedMulticutIlp(py::module & liftedMulticutModule){
        {
            typedef PyUndirectedGraph GraphType;
            typedef LiftedMulticutObjective<GraphType, double> ObjectiveType;
            exportLiftedMulticutIlpT<ObjectiveType>(liftedMulticutModule);
        }
        {
            typedef nifty::graph::UndirectedGridGraph<2,true> GraphType;
            typedef LiftedMulticutObjective<GraphType, double> ObjectiveType;
            exportLiftedMulticutIlpT<ObjectiveType>(liftedMulticutModule);
        }
        {
            typedef nifty::graph::UndirectedGridGraph<3,true> GraphType;
            typedef LiftedMulticutObjective<GraphType, double> ObjectiveType;
            exportLiftedMulticutIlpT<ObjectiveType>(liftedMulticutModule);
        }
        //{
        //    typedef PyContractionGraph<PyUndirectedGraph> GraphType;
        //    typedef LiftedMulticutObjective<GraphType, double> ObjectiveType;
        //    exportLiftedMulticutIlpT<ObjectiveType>(liftedMulticutModule);
        //}    
    }

}
} // namespace nifty::graph::opt
}
}
