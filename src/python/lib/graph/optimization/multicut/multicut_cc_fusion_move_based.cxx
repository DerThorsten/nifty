#include <pybind11/pybind11.h>

//#include "nifty/graph/optimization/multicut/fusion_move_based.hxx"

#include "nifty/python/converter.hxx"

#include "nifty/python/graph/undirected_list_graph.hxx"
#include "nifty/python/graph/edge_contraction_graph.hxx"
#include "nifty/python/graph/optimization/multicut/multicut_objective.hxx"
#include "nifty/python/graph/optimization/multicut/export_multicut_solver.hxx"

#include "nifty/python/graph/optimization/common/py_proposal_generator_factory_base.hxx"
#include "nifty/graph/optimization/common/proposal_generators/watershed_proposal_generator.hxx"

#include "nifty/graph/optimization/multicut/multicut_cc_fusion_move_based.hxx"


namespace py = pybind11;
namespace optCommon = nifty::graph::optimization::common;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

namespace nifty{
namespace graph{
namespace optimization{
namespace multicut{




    
    template<class OBJECTIVE>
    void exportMulticutCcFusionMoveBasedT(py::module & module) {
        typedef OBJECTIVE ObjectiveType;

        // the base factory
        optCommon::exportCCProposalGeneratorFactoryBaseT<ObjectiveType>(
            module, MulticutObjectiveName<ObjectiveType>::name()
        );

        
        // concrete factories
        { // watershed factory
            typedef optCommon::WatershedProposalGenerator<ObjectiveType> ProposalGeneratorType;
            typedef typename ProposalGeneratorType::Settings PGenSettigns;
            typedef typename PGenSettigns::SeedingStrategie SeedingStrategie;
            auto pGenSettigns = optCommon::exportCCProposalGenerator<ProposalGeneratorType>(module, "WatershedProposalGenerator",
                MulticutObjectiveName<ObjectiveType>::name());
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
    

        // the fusion move itself (or at least the settings)
        typedef MulticutCcFusionMove<ObjectiveType>       MinCcFusionMoveType;
        typedef typename MinCcFusionMoveType::Settings  MinCcFusionMoveSettings;

        const std::string  fmSettingsName = std::string("MulticutCcFusionMoveSettings") + MulticutObjectiveName<ObjectiveType>::name();
        py::class_<MinCcFusionMoveSettings >(module, fmSettingsName.c_str())
            .def(py::init<>())
            .def_readwrite("multicutFactory", &MinCcFusionMoveSettings::multicutFactory)
        ;



        typedef MulticutCcFusionMoveBased<ObjectiveType> Solver;
        typedef typename Solver::Settings Settings;

        

        
        exportMulticutSolver<Solver>(module,"MulticutCcFusionMoveBased")
           .def(py::init<>())
           .def_readwrite("proposalGenerator", &Settings::proposalGeneratorFactory)
           .def_readwrite("numberOfThreads", &Settings::numberOfThreads)
           .def_readwrite("numberOfIterations",&Settings::numberOfIterations)
           .def_readwrite("stopIfNoImprovement",&Settings::stopIfNoImprovement)
           .def_readwrite("fusionMoveSettings",&Settings::fusionMoveSettings)
           //.def_readwrite("verbose", &Settings::verbose)
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

} // namespace nifty::graph::optimization::multicut
} // namespace nifty::graph::optimization
}
}
