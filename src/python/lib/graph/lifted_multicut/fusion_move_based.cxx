#include <pybind11/pybind11.h>

#include "nifty/graph/optimization/lifted_multicut/fusion_move_based.hxx"

#include "nifty/python/converter.hxx"

#include "nifty/python/graph/undirected_list_graph.hxx"
#include "nifty/python/graph/edge_contraction_graph.hxx"
#include "nifty/python/graph/optimization/lifted_multicut/lifted_multicut_objective.hxx"
#include "nifty/python/graph/optimization/lifted_multicut/export_lifted_multicut_solver.hxx"
#include "nifty/python/graph/optimization/lifted_multicut/py_proposal_generator_factory_base.hxx"

namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

namespace nifty{
namespace graph{
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
    
    template<class OBJECTIVE>
    void exportFusionMoveBasedT(py::module & liftedMulticutModule) {

        // the base factroy
        exportProposalGeneratorFactoryBaseT<OBJECTIVE>(liftedMulticutModule);

        typedef OBJECTIVE ObjectiveType;
        typedef FusionMoveBased<ObjectiveType> Solver;
        typedef typename Solver::Settings Settings;
        
        exportLiftedMulticutSolver<Solver>(liftedMulticutModule,"LiftedMulticutFusionMoveBased")
           .def(py::init<>())
           //.def_readwrite("nodeNumStopCond", &Settings::nodeNumStopCond)
           //.def_readwrite("weightStopCond", &Settings::weightStopCond)
           //.def_readwrite("verbose", &Settings::verbose)
        ;
     
    }

    void exportFusionMoveBased(py::module & liftedMulticutModule) {
        {
            //typedef PyUndirectedGraph GraphType;
            //typedef LiftedMulticutObjective<GraphType, double> ObjectiveType;
            //exportFusionMoveBasedT<ObjectiveType>(liftedMulticutModule);
        }
        //{
        //    typedef PyContractionGraph<PyUndirectedGraph> GraphType;
        //    typedef MulticutObjective<GraphType, double> ObjectiveType;
        //    exportFusionMoveBasedT<ObjectiveType>(liftedMulticutModule);
        //}
    }

}
}
}
