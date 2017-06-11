#include <pybind11/pybind11.h>



// concrete solvers for concrete factories
#include "nifty/graph/optimization/multicut/multicut_decomposer.hxx"



#include "nifty/python/graph/undirected_list_graph.hxx"
#include "nifty/python/graph/edge_contraction_graph.hxx"
#include "nifty/python/graph/optimization/multicut/multicut_objective.hxx"
#include "nifty/python/converter.hxx"
#include "nifty/python/graph/optimization/multicut/export_multicut_solver.hxx"

namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

namespace nifty{
namespace graph{
namespace optimization{
namespace multicut{
    
    template<class OBJECTIVE>
    void exportMulticutDecomposerT(py::module & multicutModule){



            ///////////////////////////////////////////////////////////////
            // DOCSTRING HELPER
            ///////////////////////////////////////////////////////////////
            nifty::graph::optimization::SolverDocstringHelper docHelper;
            docHelper.objectiveName = "multicut objective";
            docHelper.objectiveClsName = MulticutObjectiveName<OBJECTIVE>::name();
            docHelper.name = "multicut decomposer";
            docHelper.mainText =  
                "This solver tries to decompose the model into\n"
                "sub-models  as described in :cite:`alush_2013_simbad`.\n"
                "If a model decomposes into components such that there are no\n"
                "positive weighted edges between the components one can\n"
                "optimize each model separately.\n";

        
            docHelper.cites.emplace_back("alush_2013_simbad");
            docHelper.note = "This solver should be warm started,"
                            "otherwise  results are very poor."
                            "Using :func:`greedyAdditiveFactory` to create "
                            "a solver for warm starting is suggested.";




        typedef OBJECTIVE ObjectiveType;
        typedef MulticutDecomposer<ObjectiveType> Solver;
        typedef typename Solver::SettingsType SettingsType;
        const auto solverName = std::string("MulticutDecomposer");
        exportMulticutSolver<Solver>(multicutModule, solverName.c_str())
            .def(py::init<>())
            .def_readwrite("submodelFactory",   &SettingsType::submodelFactory)
            .def_readwrite("fallthroughFactory",&SettingsType::fallthroughFactory)
        ; 
    }

    
    void exportMulticutDecomposer(py::module & multicutModule){

        {
            typedef PyUndirectedGraph GraphType;
            typedef MulticutObjective<GraphType, double> ObjectiveType;
            exportMulticutDecomposerT<ObjectiveType>(multicutModule);
        }
        {
            typedef PyContractionGraph<PyUndirectedGraph> GraphType;
            typedef MulticutObjective<GraphType, double> ObjectiveType;
            exportMulticutDecomposerT<ObjectiveType>(multicutModule);
        }    
         
    }
} // namespace nifty::graph::optimization::multicut
} // namespace nifty::graph::optimization
}
}
