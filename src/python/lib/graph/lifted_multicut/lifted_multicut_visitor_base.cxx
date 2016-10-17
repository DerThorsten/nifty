#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>


#include "nifty/graph/optimization/lifted_multicut/lifted_multicut_base.hxx"
#include "nifty/graph/optimization/lifted_multicut/lifted_multicut_visitor_base.hxx"

#include "nifty/python/graph/undirected_list_graph.hxx"
//#include "nifty/python/graph/edge_contraction_graph.hxx"
#include "nifty/python/graph/optimization/lifted_multicut/lifted_multicut_objective.hxx"
#include "nifty/python/graph/optimization/lifted_multicut/py_lifted_multicut_visitor_base.hxx"

#include "nifty/python/graph/optimization/lifted_multicut/lifted_multicut_objective.hxx"
#include "nifty/python/graph/optimization/lifted_multicut/weighted_lifted_multicut_objective.hxx"
#include "nifty/python/graph/optimization/lifted_multicut/loss_augmented_view_lifted_multicut_objective.hxx"


#include "nifty/python/converter.hxx"





namespace py = pybind11;


namespace nifty{
namespace graph{
namespace lifted_multicut{


    using namespace py;
    //PYBIND11_DECLARE_HOLDER_TYPE(McBase, std::shared_ptr<McBase>);

    template<class OBJECTIVE>
    void exportLiftedMulticutVisitorBaseT(py::module & liftedMulticutModule) {

        typedef OBJECTIVE ObjectiveType;
        typedef PyLiftedMulticutVisitorBase<ObjectiveType> PyLmcVisitorBase;
        typedef LiftedMulticutVisitorBase<ObjectiveType> LmcVisitorBase;

        
        const auto objName = LiftedMulticutObjectiveName<ObjectiveType>::name();
        const auto lmcVisitorBaseClsName = std::string("LiftedMulticutVisitorBase") + objName;
        const auto lmcVerboseVisitorClsName = std::string("LiftedMulticutVerboseVisitor") + objName;

        // base factory
        py::class_<
            LmcVisitorBase, 
            std::unique_ptr<LmcVisitorBase>, 
            PyLmcVisitorBase 
        > lmcVisitorBase(liftedMulticutModule, lmcVisitorBaseClsName.c_str());
        
        lmcVisitorBase
        ;


        // concrete visitors
        

        typedef LiftedMulticutVerboseVisitor<ObjectiveType> LmcVerboseVisitor; 
        
        py::class_<LmcVerboseVisitor, std::unique_ptr<LmcVerboseVisitor> >(liftedMulticutModule, lmcVerboseVisitorClsName.c_str(),  lmcVisitorBase)
            .def(py::init<const int >(),
                py::arg_t<int>("printNth",1)
            )
            .def("stopOptimize",&LmcVerboseVisitor::stopOptimize)
        ;
    }

    void exportLiftedMulticutVisitorBase(py::module & liftedMulticutModule) {

        {
            typedef PyUndirectedGraph GraphType;
            typedef LiftedMulticutObjective<GraphType, double> ObjectiveType;
            exportLiftedMulticutVisitorBaseT<ObjectiveType>(liftedMulticutModule);
        }

        {
            typedef PyUndirectedGraph GraphType;
            typedef WeightedLiftedMulticutObjective<GraphType, float> ObjectiveType;
            exportLiftedMulticutVisitorBaseT<ObjectiveType>(liftedMulticutModule);
        }
        {
            typedef PyUndirectedGraph GraphType;
            typedef WeightedLiftedMulticutObjective<GraphType, float> WeightedObjectiveType;
            typedef LossAugmentedViewLiftedMulticutObjective<WeightedObjectiveType> ObjectiveType;
            exportLiftedMulticutVisitorBaseT<ObjectiveType>(liftedMulticutModule);
        }
        
        //{
        //    typedef PyContractionGraph<PyUndirectedGraph> GraphType;
        //    typedef LiftedMulticutObjective<GraphType, double> ObjectiveType;
        //    exportLiftedMulticutVisitorBaseT<ObjectiveType>(liftedMulticutModule);
        //}
    }      

}
}
}
    
