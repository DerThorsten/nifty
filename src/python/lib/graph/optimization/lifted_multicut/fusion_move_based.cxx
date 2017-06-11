#include <pybind11/pybind11.h>

#include "nifty/graph/optimization/lifted_multicut/fusion_move_based.hxx"

#include "nifty/python/converter.hxx"

#include "nifty/python/graph/undirected_grid_graph.hxx"
#include "nifty/python/graph/undirected_list_graph.hxx"
#include "nifty/python/graph/edge_contraction_graph.hxx"
#include "nifty/python/graph/optimization/lifted_multicut/lifted_multicut_objective.hxx"
#include "nifty/python/graph/optimization/lifted_multicut/export_lifted_multicut_solver.hxx"
#include "nifty/python/graph/optimization/lifted_multicut/py_proposal_generator_factory_base.hxx"
#include "nifty/graph/optimization/lifted_multicut/proposal_generators/proposal_generator_factory.hxx"
#include "nifty/graph/optimization/lifted_multicut/proposal_generators/watershed_proposal_generator.hxx"

namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

namespace nifty{
namespace graph{
namespace optimization{
namespace lifted_multicut{





    template<class OBJECTIVE>
    void exportProposalGeneratorFactoryBaseT(py::module & liftedMulticutModule) {
        typedef OBJECTIVE ObjectiveType;
        const auto objName = LiftedMulticutObjectiveName<ObjectiveType>::name();
        const auto clsName = std::string("ProposalGeneratorFactoryBase") + objName;

        typedef ProposalGeneratorFactoryBase<ObjectiveType> LmcPropGenFactoryBase;
        typedef PyProposalGeneratorFactoryBase<ObjectiveType> PyLmcPropGenFactoryBase;
        
        // base factory
        py::class_<
            LmcPropGenFactoryBase, 
            std::shared_ptr<LmcPropGenFactoryBase>, 
           PyLmcPropGenFactoryBase
        >  proposalsGenFactoryBase(liftedMulticutModule, clsName.c_str());
        
        proposalsGenFactoryBase
            .def(py::init<>())
        ;
    }




    template<class PROPOSAL_GENERATOR>
    py::class_<typename PROPOSAL_GENERATOR::SettingsType> 
    exportProposalGenerator(
        py::module & liftedMulticutModule,
        const std::string & clsName
    ){
        typedef PROPOSAL_GENERATOR ProposalGeneratorType;
        typedef typename ProposalGeneratorType::ObjectiveType ObjectiveType;
        typedef typename ProposalGeneratorType::SettingsType SettingsType;
        typedef ProposalGeneratorFactory<ProposalGeneratorType> Factory;

        const auto objName = LiftedMulticutObjectiveName<ObjectiveType>::name();

        const std::string settingsName = clsName + std::string("SettingsType") + objName;
        const std::string factoryBaseName = std::string("ProposalGeneratorFactoryBase")+objName;
        const std::string factoryName = clsName + std::string("Factory") + objName;


         // settings
        auto settingsCls = py::class_< SettingsType >(liftedMulticutModule, settingsName.c_str())
        ;

        // factory
        py::object factoryBase = liftedMulticutModule.attr(factoryBaseName.c_str());
        py::class_<Factory, std::shared_ptr<Factory> >(liftedMulticutModule, factoryName.c_str(),  factoryBase)
            .def(py::init<const SettingsType &>(),
                py::arg_t<SettingsType>("setttings",SettingsType())
            )
        ;


        return settingsCls;

    }   






    
    template<class OBJECTIVE>
    void exportFusionMoveBasedT(py::module & liftedMulticutModule) {
        typedef OBJECTIVE ObjectiveType;

        // the base factory
        exportProposalGeneratorFactoryBaseT<ObjectiveType>(liftedMulticutModule);

        // concrete factories
        { // watershed factory
            typedef WatershedProposalGenerator<ObjectiveType> ProposalGeneratorType;
            typedef typename ProposalGeneratorType::SettingsType PGenSettigns;
            typedef typename PGenSettigns::SeedingStrategie SeedingStrategie;
            auto pGenSettigns = exportProposalGenerator<ProposalGeneratorType>(liftedMulticutModule, "WatershedProposalGenerator");

            py::enum_<SeedingStrategie>(pGenSettigns, "SeedingStrategie")
                .value("SEED_FROM_LIFTED", SeedingStrategie::SEED_FROM_LIFTED)
                .value("SEED_FROM_LOCAL", SeedingStrategie::SEED_FROM_LOCAL)
                .value("SEED_FROM_BOTH", SeedingStrategie::SEED_FROM_BOTH)
            ;

            pGenSettigns
                .def(py::init<>())
                .def_readwrite("seedingStrategie", &PGenSettigns::seedingStrategie)
                .def_readwrite("sigma", &PGenSettigns::sigma)
                .def_readwrite("numberOfSeeds", &PGenSettigns::numberOfSeeds)
            ;
        }

        
        typedef FusionMoveBased<ObjectiveType> Solver;
        typedef typename Solver::SettingsType SettingsType;
        
        exportLiftedMulticutSolver<Solver>(liftedMulticutModule,"FusionMoveBased")
           .def(py::init<>())
           .def_readwrite("proposalGenerator", &SettingsType::proposalGeneratorFactory)
           .def_readwrite("numberOfThreads", &SettingsType::numberOfThreads)
           .def_readwrite("numberOfIterations",&SettingsType::numberOfIterations)
           .def_readwrite("stopIfNoImprovement",&SettingsType::stopIfNoImprovement)
           
           //.def_readwrite("verbose", &SettingsType::verbose)
        ;
     
    }

    void exportFusionMoveBased(py::module & liftedMulticutModule) {
        {
            typedef PyUndirectedGraph GraphType;
            typedef LiftedMulticutObjective<GraphType, double> ObjectiveType;
            exportFusionMoveBasedT<ObjectiveType>(liftedMulticutModule);
        }
        {
            typedef nifty::graph::UndirectedGridGraph<2,true> GraphType;
            typedef LiftedMulticutObjective<GraphType, double> ObjectiveType;
            exportFusionMoveBasedT<ObjectiveType>(liftedMulticutModule);
        }
        //{
        //    typedef PyContractionGraph<PyUndirectedGraph> GraphType;
        //    typedef MulticutObjective<GraphType, double> ObjectiveType;
        //    exportFusionMoveBasedT<ObjectiveType>(liftedMulticutModule);
        //}
    }

}
} // namespace nifty::graph::optimization
}
}
