#include <pybind11/pybind11.h>



#include "nifty/graph/opt/multicut/fusion_move_based.hxx"
#include "nifty/graph/opt/multicut/fusion_move.hxx"
#include "nifty/graph/opt/multicut/proposal_generators/greedy_additive_proposals.hxx"
#include "nifty/graph/opt/multicut/proposal_generators/watershed_proposals.hxx"

#include "nifty/python/graph/undirected_list_graph.hxx"
#include "nifty/python/graph/edge_contraction_graph.hxx"
#include "nifty/python/graph/opt/multicut/multicut_objective.hxx"
#include "nifty/python/converter.hxx"
#include "nifty/python/graph/opt/multicut/export_multicut_solver.hxx"

namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

namespace nifty{
namespace graph{
namespace opt{
namespace multicut{

    template<class OBJECTIVE>
    void exportFusionMoveBasedT(py::module & multicutModule) {



        typedef OBJECTIVE ObjectiveType;
        const auto objName = MulticutObjectiveName<ObjectiveType>::name();
        const std::string factoryBaseName = std::string("MulticutFactoryBase")+objName;
        const std::string solverBaseName = std::string("MulticutBase") + objName;
        


        // the fusion mover parameter itself
        {
            typedef FusionMove<ObjectiveType> FusionMoveType;
            typedef typename FusionMoveType::SettingsType FusionMoveSettings;
            const auto fmSettingsName = std::string("__FusionMoveSettingsType") + objName;
            py::class_<FusionMoveSettings>(multicutModule, fmSettingsName.c_str())
                .def(py::init<>())
                .def_readwrite("mcFactory",&FusionMoveSettings::mcFactory)
            ;

        }

        #if 0

        // the inference 
        {

            ///////////////////////////////////////////////////////////////
            // DOCSTRING HELPER
            ///////////////////////////////////////////////////////////////
            nifty::graph::opt::SolverDocstringHelper docHelper;
            docHelper.objectiveName = "fusion move based greedy additive"; 
            docHelper.objectiveClsName = MulticutObjectiveName<OBJECTIVE>::name();
            docHelper.name = "chained solvers "; 
            docHelper.mainText =  
                "Fusion moves for multicuts / correlation clustering \n"
                "as descried in :cite:`beier_15_funsion`\n";
            docHelper.cites.emplace_back("beier_15_funsion");


            typedef GreedyAdditiveProposals<ObjectiveType> ProposalGen;
            typedef typename ProposalGen::SettingsType ProposalGenSettings;
            typedef FusionMoveBased<ProposalGen> Solver;
            typedef typename Solver::SettingsType SettingsType;

            const std::string solverName = "FusionMoveBasedGreedyAdditive";
            const std::string pgenSettingsName = std::string("__") + solverName + std::string("ProposalGenSettings") + objName;

            py::class_<ProposalGenSettings>(multicutModule, pgenSettingsName.c_str())
                .def(py::init<>())
                .def_readwrite("sigma", &ProposalGenSettings::sigma)
                .def_readwrite("weightStopCond", &ProposalGenSettings::weightStopCond)
                .def_readwrite("nodeNumStopCond",  &ProposalGenSettings::nodeNumStopCond)
            ;

            exportMulticutSolver<Solver>(multicutModule,solverName.c_str(), docHelper)
                .def(py::init<>())
                .def_readwrite("verbose", &SettingsType::verbose)
                .def_readwrite("numberOfIterations", &SettingsType::numberOfIterations)
                .def_readwrite("numberOfParallelProposals",&SettingsType::numberOfParallelProposals)
                .def_readwrite("fuseN",&SettingsType::fuseN)
                .def_readwrite("stopIfNoImprovement",&SettingsType::stopIfNoImprovement)
                .def_readwrite("proposalGenSettings", &SettingsType::proposalGenSettings)
                .def_readwrite("fusionMoveSettings",  &SettingsType::fusionMoveSettings)
                .def_readwrite("numberOfThreads",  &SettingsType::numberOfThreads)
            ;
        }

        // the inference 
        {


            ///////////////////////////////////////////////////////////////
            // DOCSTRING HELPER
            ///////////////////////////////////////////////////////////////
            nifty::graph::opt::SolverDocstringHelper docHelper;
            docHelper.objectiveName = "fusion move based greedy additive"; 
            docHelper.objectiveClsName = MulticutObjectiveName<OBJECTIVE>::name();
            docHelper.name = "chained solvers "; 
            docHelper.mainText =  
                "Fusion moves for multicuts / correlation clustering \n"
                "as descried in :cite:`beier_15_funsion`\n";
            docHelper.cites.emplace_back("beier_15_funsion");


            typedef WatershedProposals<ObjectiveType> ProposalGen;
            typedef typename ProposalGen::SettingsType ProposalGenSettings;
            typedef FusionMoveBased<ProposalGen> Solver;
            typedef typename Solver::SettingsType SettingsType;

            const std::string solverName = "FusionMoveBasedWatershed";
            const std::string pgenSettingsName = std::string("__") + solverName + std::string("ProposalGenSettings") + objName;

            py::class_<ProposalGenSettings>(multicutModule, pgenSettingsName.c_str())
                .def(py::init<>())
                .def_readwrite("sigma", &ProposalGenSettings::sigma)
                .def_readwrite("seedFraction", &ProposalGenSettings::seedFraction)
            ;

            exportMulticutSolver<Solver>(multicutModule,solverName.c_str(), docHelper)
                .def(py::init<>())
                .def_readwrite("verbose", &SettingsType::verbose)
                .def_readwrite("numberOfIterations", &SettingsType::numberOfIterations)
                .def_readwrite("numberOfParallelProposals",&SettingsType::numberOfParallelProposals)
                .def_readwrite("fuseN",&SettingsType::fuseN)
                .def_readwrite("stopIfNoImprovement",&SettingsType::stopIfNoImprovement)
                .def_readwrite("proposalGenSettings", &SettingsType::proposalGenSettings)
                .def_readwrite("fusionMoveSettings",  &SettingsType::fusionMoveSettings)
                .def_readwrite("numberOfThreads",  &SettingsType::numberOfThreads)
            ;
        }
        #endif
    }

    void exportFusionMoveBased(py::module & multicutModule) {
        {
            typedef PyUndirectedGraph GraphType;
            typedef MulticutObjective<GraphType, double> ObjectiveType;
            exportFusionMoveBasedT<ObjectiveType>(multicutModule);
        }
        {
            typedef PyContractionGraph<PyUndirectedGraph> GraphType;
            typedef MulticutObjective<GraphType, double> ObjectiveType;
            exportFusionMoveBasedT<ObjectiveType>(multicutModule);
        }
    }

} // namespace nifty::graph::opt::multicut
} // namespace nifty::graph::opt
}
}
