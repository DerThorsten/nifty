#include <pybind11/pybind11.h>

//#include "nifty/graph/opt/multicut/fusion_move_based.hxx"

#include "nifty/python/converter.hxx"

#include "nifty/python/graph/undirected_list_graph.hxx"
#include "nifty/python/graph/edge_contraction_graph.hxx"
#include "nifty/python/graph/opt/multicut/multicut_objective.hxx"
#include "nifty/python/graph/opt/multicut/export_multicut_solver.hxx"

// proposal generator helper
#include "nifty/python/graph/opt/common/py_proposal_generator_factory_base.hxx"

// proposal generators
#include "nifty/graph/opt/common/proposal_generators/watershed_proposal_generator.hxx"
#include "nifty/graph/opt/common/proposal_generators/interface_flipper_proposal_generator.hxx"
#include "nifty/graph/opt/common/proposal_generators/random_node_color_proposal_generator.hxx"

// the solver
#include "nifty/graph/opt/multicut/cc_fusion_move_based.hxx"


namespace py = pybind11;
namespace optCommon = nifty::graph::opt::common;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

namespace nifty{
namespace graph{
namespace opt{
namespace multicut{




    
    template<class OBJECTIVE>
    void exportMulticutCcFusionMoveBasedT(py::module & module) {
        typedef OBJECTIVE ObjectiveType;

        // the base factory
        optCommon::exportCCProposalGeneratorFactoryBaseT<ObjectiveType>(
            module, MulticutObjectiveName<ObjectiveType>::name()
        );

        
        // concrete proposal generators

        { // watershed proposal generators
            typedef optCommon::WatershedProposalGenerator<ObjectiveType> ProposalGeneratorType;
            typedef typename ProposalGeneratorType::SettingsType PGenSettigns;
            typedef typename PGenSettigns::SeedingStrategie SeedingStrategie;
            auto pGenSettigns = optCommon::exportCCProposalGenerator<ProposalGeneratorType>(
                module, 
                "WatershedProposalGenerator",
                MulticutObjectiveName<ObjectiveType>::name()
            );
            py::enum_<SeedingStrategie>(pGenSettigns, "SeedingStrategie")
                .value("SEED_FROM_NEGATIVE", SeedingStrategie::SEED_FROM_NEGATIVE)
                .value("SEED_FROM_ALL", SeedingStrategie::SEED_FROM_ALL)
            ;
            pGenSettigns
                .def(py::init<>())
                .def_readwrite("seedingStrategie", &PGenSettigns::seedingStrategie)
                .def_readwrite("sigma", &PGenSettigns::sigma)
                .def_readwrite("numberOfSeeds", &PGenSettigns::numberOfSeeds)
            ;
        }

        { // interface flipper proposal generator
            typedef optCommon::InterfaceFlipperProposalGenerator<ObjectiveType> ProposalGeneratorType;
            typedef typename ProposalGeneratorType::SettingsType PGenSettigns;
            auto pGenSettigns = optCommon::exportCCProposalGenerator<ProposalGeneratorType>(
                module, 
                "InterfaceFlipperProposalGenerator",
                MulticutObjectiveName<ObjectiveType>::name()
            );

            pGenSettigns
                .def(py::init<>())
            ;
        }

        { // random node color proposal generator
            typedef optCommon::RandomNodeColorProposalGenerator<ObjectiveType> ProposalGeneratorType;
            typedef typename ProposalGeneratorType::SettingsType PGenSettigns;
            auto pGenSettigns = optCommon::exportCCProposalGenerator<ProposalGeneratorType>(
                module, 
                "RandomNodeColorProposalGenerator",
                MulticutObjectiveName<ObjectiveType>::name()
            );

            pGenSettigns
                .def(py::init<>())
                .def_readwrite("numberOfColors", &PGenSettigns::numberOfColors)
            ;
        }

    


        typedef CcFusionMoveBased<ObjectiveType> Solver;
        typedef typename Solver::SettingsType SettingsType;

        
        exportMulticutSolver<Solver>(module,"CcFusionMoveBased")
           .def(py::init<>())
           .def_readwrite("proposalGenerator", &SettingsType::proposalGeneratorFactory)
           .def_readwrite("numberOfThreads", &SettingsType::numberOfThreads)
           .def_readwrite("numberOfIterations",&SettingsType::numberOfIterations)
           .def_readwrite("stopIfNoImprovement",&SettingsType::stopIfNoImprovement)
           .def_readwrite("fusionMoveSettings",&SettingsType::fusionMoveSettings)
           //.def_readwrite("verbose", &SettingsType::verbose)
        ;
        
     
    }

    void exportMulticutCcFusionMoveBased(py::module & module) {
        {
            typedef PyUndirectedGraph GraphType;
            typedef MulticutObjective<GraphType, double> ObjectiveType;
            exportMulticutCcFusionMoveBasedT<ObjectiveType>(module);
        }
        //{
        //    typedef PyContractionGraph<PyUndirectedGraph> GraphType;
        //    typedef MulticutObjective<GraphType, double> ObjectiveType;
        //    exportMulticutCcFusionMoveBased<ObjectiveType>(module);
        //}
    }

} // namespace nifty::graph::opt::multicut
} // namespace nifty::graph::opt
}
}
