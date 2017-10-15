#include <pybind11/pybind11.h>



// concrete solvers for concrete factories
#include "nifty/graph/opt/multicut/cgc.hxx"


#include "nifty/python/graph/undirected_list_graph.hxx"
#include "nifty/python/graph/edge_contraction_graph.hxx"
#include "nifty/python/graph/opt/multicut/multicut_objective.hxx"
#include "nifty/python/converter.hxx"

#include "nifty/python/graph/opt/solver_docstring.hxx"
#include "nifty/python/graph/opt/multicut/export_multicut_solver.hxx"

namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);


namespace nifty{
namespace graph{
namespace opt{
namespace multicut{
    
    template<class OBJECTIVE>
    void exportCgcT(py::module & multicutModule){


        ///////////////////////////////////////////////////////////////
        // DOCSTRING HELPER
        ///////////////////////////////////////////////////////////////
        nifty::graph::opt::SolverDocstringHelper docHelper;
        docHelper.objectiveName = "multicut objective";
        docHelper.objectiveClsName = MulticutObjectiveName<OBJECTIVE>::name();
        docHelper.name = "cgc";
        docHelper.mainText =  
        "Cut glue & and cut (CGC) :cite:`beier_14_cut` works by applying a series\n"
        "Of two-colorings / min-cuts to optimize the multicut problem.\n"
        "The algorithm works as illustrated in the figure below.\n\n"
        ".. figure:: ../images/cgc.png\n\n"
        "   Left: The graph is recursively partitioned by applying a min-cut on each\n"
        "   connected component.\n"
        "   Right: Each adjacent pair of connected components is merged and optimized\n"
        "   again by searching for a new min-cut / 2-coloring.\n"
        "   This is repeated until nothing improves anymore.\n";
        docHelper.cites.emplace_back("beier_14_cut");
        docHelper.note = "This solver should be warm started,"
                            "otherwise  the glue phase is very slow."
                            "Using :func:`greedyAdditiveFactory` to create "
                            "a solver for warm starting is suggested.";









        typedef OBJECTIVE ObjectiveType;
        typedef Cgc<ObjectiveType> Solver;
        typedef typename Solver::SettingsType SettingsType;
        const auto solverName = std::string("Cgc");


        nifty::graph::opt::multicut::exportMulticutSolver<Solver>(multicutModule, solverName.c_str(), docHelper)

            .def(py::init<>())

            .def_readwrite("doCutPhase", &SettingsType::doCutPhase)
            .def_readwrite("doBetterCutPhase", &SettingsType::doBetterCutPhase)
            .def_readwrite("sizeRegularizer", &SettingsType::sizeRegularizer)
            .def_readwrite("nodeNumStopCond", &SettingsType::nodeNumStopCond)
            .def_readwrite("doGlueAndCutPhase", &SettingsType::doGlueAndCutPhase)
            .def_readwrite("mincutFactory", &SettingsType::mincutFactory)
            .def_readwrite("multicutFactory", &SettingsType::multicutFactory)

        ; 
    }

    
    void exportCgc(py::module & multicutModule){


        py::options options;
        options.disable_function_signatures();
    
        {
            typedef PyUndirectedGraph GraphType;
            typedef MulticutObjective<GraphType, double> ObjectiveType;
            exportCgcT<ObjectiveType>(multicutModule);
        }
        {
            typedef PyContractionGraph<PyUndirectedGraph> GraphType;
            typedef MulticutObjective<GraphType, double> ObjectiveType;
            exportCgcT<ObjectiveType>(multicutModule);
        }    
         
    }

} // namespace nifty::graph::opt::multicut
} // namespace nifty::graph::opt
} // namespace nifty::graph
} // namespace nifty
