#include <pybind11/pybind11.h>

#include <boost/algorithm/string.hpp>


// concrete solvers for concrete factories
#include "nifty/graph/opt/ho_multicut/ho_multicut_ilp.hxx"

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
    
    template<class OBJECTIVE, class BACKEND>
    void exportHoMulticutIlpWithBackendT(py::module & hoMulticutModule, const std::string & backendName){






        ///////////////////////////////////////////////////////////////
        // DOCSTRING HELPER
        ///////////////////////////////////////////////////////////////
        nifty::graph::opt::SolverDocstringHelper docHelper;
        docHelper.objectiveName = "ho multicut objective"; 
        docHelper.objectiveClsName = HoMulticutObjectiveName<OBJECTIVE>::name();
        docHelper.name = "ho multicut ilp " + backendName; 
        docHelper.mainText =  
            "Find a global optimal solution by a cutting plane ILP solver\n"
            "as described in :cite:`Kappes-2011` \n"
            "and :cite:`andres_2011_probabilistic` \n";
        docHelper.cites.emplace_back("Kappes-2011");
        docHelper.cites.emplace_back("andres_2011_probabilistic");
        docHelper.note = "This might take very long for large models.";
        docHelper.requirements.emplace_back(
            std::string("WITH_") + boost::to_upper_copy<std::string>(backendName)
        );


        
        typedef OBJECTIVE ObjectiveType;
        typedef BACKEND IlpSolver;
        typedef HoMulticutIlp<ObjectiveType, IlpSolver> Solver;
        
        typedef typename Solver::SettingsType SettingsType;        
        const auto solverName = std::string("HoMulticutIlp") + backendName;
        // todo exportHoMulticutSolver should be in the correct namespace

        nifty::graph::opt::ho_multicut::exportHoMulticutSolver<Solver>(hoMulticutModule, solverName.c_str(), docHelper)

            .def(py::init<>())
            //.def_readwrite("numberOfIterations", &SettingsType::numberOfIterations)
            //.def_readwrite("verbose", &SettingsType::verbose)
            //.def_readwrite("verboseIlp", &SettingsType::verboseIlp)
            .def_readwrite("addThreeCyclesConstraints", &SettingsType::addThreeCyclesConstraints)
            .def_readwrite("addOnlyViolatedThreeCyclesConstraints", &SettingsType::addOnlyViolatedThreeCyclesConstraints)
            .def_readwrite("ilpSettings",&SettingsType::ilpSettings)
            .def_readwrite("ilp",&SettingsType::ilp)
            .def_readwrite("integralHo",&SettingsType::integralHo)
            .def_readwrite("ilpSettings",&SettingsType::ilpSettings)
            .def_readwrite("timeLimit", &SettingsType::timeLimit)
            .def_readwrite("maxIterations", &SettingsType::maxIterations)
        ; 
    }

    template<class OBJECTIVE>
    void exportHoMulticutIlpT(py::module & hoMulticutModule) {
        typedef OBJECTIVE ObjectiveType;
        #ifdef WITH_CPLEX
            exportHoMulticutIlpWithBackendT<ObjectiveType, ilp_backend::Cplex>(hoMulticutModule, "Cplex");
        #endif
        #ifdef WITH_GUROBI
            exportHoMulticutIlpWithBackendT<ObjectiveType, ilp_backend::Gurobi>(hoMulticutModule, "Gurobi");
        #endif
        #ifdef WITH_GLPK
            exportHoMulticutIlpWithBackendT<ObjectiveType, ilp_backend::Glpk>(hoMulticutModule, "Glpk");
        #endif   
        
    }
    
    void exportHoMulticutIlp(py::module & hoMulticutModule){
        
        py::options options;
        options.disable_function_signatures();
                


        {
            typedef PyUndirectedGraph GraphType;
            typedef HoMulticutObjective<GraphType, double> ObjectiveType;
            exportHoMulticutIlpT<ObjectiveType>(hoMulticutModule);
        }  
    }
} // namespace nifty::graph::opt::multicut
} // namespace nifty::graph::opt
} // namespace nifty::graph
} // namespace nifty