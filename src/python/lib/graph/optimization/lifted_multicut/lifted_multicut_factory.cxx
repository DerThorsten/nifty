#include <pybind11/pybind11.h>


#include "nifty/python/graph/undirected_grid_graph.hxx"
#include "nifty/python/graph/undirected_list_graph.hxx"
//#include "nifty/python/graph/edge_contraction_graph.hxx"
#include "nifty/python/graph/optimization/lifted_multicut/lifted_multicut_objective.hxx"


#include "nifty/python/converter.hxx"
#include "nifty/python/graph/optimization/lifted_multicut/py_lifted_multicut_factory.hxx"





namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(LmcBase, std::shared_ptr<LmcBase>);

namespace nifty{
namespace graph{
namespace lifted_multicut{


    template<class OBJECTIVE>
    void exportLiftedMulticutFactoryT(py::module & liftedMulticutModule) {

        typedef OBJECTIVE ObjectiveType;
        typedef PyLiftedMulticutFactoryBase<ObjectiveType> PyLmcFactoryBase;
        typedef LiftedMulticutFactoryBase<ObjectiveType> LmcFactoryBase;


        const auto objName = LiftedMulticutObjectiveName<ObjectiveType>::name();
        const auto clsName = std::string("LiftedMulticutFactoryBase") + objName;

        // base factory
        py::class_<
            LmcFactoryBase, 
            std::shared_ptr<LmcFactoryBase>, 
            PyLmcFactoryBase 
        > lmcFactoryBase(liftedMulticutModule, clsName.c_str());
        
        lmcFactoryBase
            .def(py::init<>())

            .def("create", 
                //&LmcFactoryBase::create,
                [](LmcFactoryBase * self, const ObjectiveType & obj){
                    return self->createRawPtr(obj);
                },
                //,
                py::return_value_policy::take_ownership,
                py::keep_alive<0,2>()
                )
        ;

    }

    void exportLiftedMulticutFactory(py::module & liftedMulticutModule) {
        {
            typedef PyUndirectedGraph GraphType;
            typedef LiftedMulticutObjective<GraphType, double> ObjectiveType;
            exportLiftedMulticutFactoryT<ObjectiveType>(liftedMulticutModule);
        }
        {
            typedef nifty::graph::UndirectedGridGraph<2,true> GraphType;
            typedef LiftedMulticutObjective<GraphType, double> ObjectiveType;
            exportLiftedMulticutFactoryT<ObjectiveType>(liftedMulticutModule);
        }
        //{
        //    typedef PyContractionGraph<PyUndirectedGraph> GraphType;
        //    typedef LiftedMulticutObjective<GraphType, double> ObjectiveType;
        //    exportLiftedMulticutFactoryT<ObjectiveType>(liftedMulticutModule);
        //}
    }

}
}
}
