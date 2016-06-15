#include <pybind11/pybind11.h>

#include "nifty/graph/multicut/multicut_objective.hxx"
#include "nifty/graph/simple_graph.hxx"
#include "nifty/graph/multicut/fusion_move_based.hxx"
#include "nifty/graph/multicut/fusion_move.hxx"
#include "nifty/graph/multicut/proposal_generators/greedy_additive_proposals.hxx"

#include "../../converter.hxx"
#include "export_multicut_solver.hxx"

namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

namespace nifty{
namespace graph{




    void exportFusionMoveBased(py::module & multicutModule) {



        typedef UndirectedGraph<> Graph;
        typedef MulticutObjective<Graph, double> Objective;


        // the fusion mover parameter itself
        {
            typedef FusionMove<Objective> FusionMoveType;
            typedef typename FusionMoveType::Settings FusionMoveSettings;

            py::class_<FusionMoveSettings>(multicutModule, "FusionMoveSettingsUndirectedGraph")
                .def(py::init<>())
                .def_readwrite("mcFactory",&FusionMoveSettings::mcFactory)
            ;

        }



        // the inference 
        {
            typedef GreedyAdditiveProposals<Objective> ProposalGen;
            typedef typename ProposalGen::Settings ProposalGenSettings;
            typedef FusionMoveBased<ProposalGen> Solver;
            typedef typename Solver::Settings Settings;

            const std::string graphName = "UndirectedGraph";
            const std::string solverName = "FusionMoveBasedGreedyAdditive";
            const std::string pgenSettingsName = solverName + std::string("ProposalGenSettings") + graphName;

            py::class_<ProposalGenSettings>(multicutModule, pgenSettingsName.c_str())
                .def(py::init<>())
                .def_readwrite("sigma", &ProposalGenSettings::sigma)
                .def_readwrite("weightStopCond", &ProposalGenSettings::weightStopCond)
                .def_readwrite("nodeNumStopCond",  &ProposalGenSettings::nodeNumStopCond)
            ;

            exportMulticutSolver<Solver>(multicutModule,"FusionMoveBasedGreedyAdditive","UndirectedGraph")
                .def(py::init<>())
                .def_readwrite("verbose", &Settings::verbose)
                .def_readwrite("numberOfIterations", &Settings::numberOfIterations)
                .def_readwrite("numberOfParallelProposals",&Settings::numberOfParallelProposals)
                .def_readwrite("fuseN",&Settings::fuseN)
                .def_readwrite("stopIfNoImprovement",&Settings::stopIfNoImprovement)
                .def_readwrite("proposalGenSettings", &Settings::proposalGenSettings)
                .def_readwrite("fusionMoveSettings",  &Settings::fusionMoveSettings)

            ;
        }
     
    }

}
}
