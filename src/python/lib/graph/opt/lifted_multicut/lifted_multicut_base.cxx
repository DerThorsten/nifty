#include <cstddef>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "xtensor-python/pytensor.hpp"

#include "nifty/python/graph/undirected_grid_graph.hxx"
#include "nifty/python/graph/undirected_list_graph.hxx"
#include "nifty/python/graph/opt/lifted_multicut/lifted_multicut_objective.hxx"
#include "nifty/python/graph/opt/lifted_multicut/py_lifted_multicut_base.hxx"



namespace py = pybind11;


namespace nifty{
namespace graph{
namespace opt{
namespace lifted_multicut{

    using namespace py;

    template<class OBJECTIVE>
    void exportLiftedMulticutBaseT(py::module & liftedMulticutModule) {


        typedef OBJECTIVE ObjectiveType;
        typedef PyLiftedMulticutBase<ObjectiveType> PyLmcBase;
        typedef LiftedMulticutBase<ObjectiveType> LmcBase;
        typedef LiftedMulticutEmptyVisitor<ObjectiveType> EmptyVisitor;
        typedef LiftedMulticutVisitorBase<ObjectiveType> LmcVisitorBase;
        typedef xt::pytensor<uint64_t, 1>::shape_type ShapeType;

        const auto objName = LiftedMulticutObjectiveName<ObjectiveType>::name();
        const auto clsName = std::string("LiftedMulticutBase") + objName;
        // base factory
        py::class_<
            LmcBase,
            std::unique_ptr<LmcBase>,
            PyLmcBase
        > lmcBase(liftedMulticutModule, clsName.c_str());

        lmcBase
            .def(py::init<>())
            .def("optimize",
                [](
                    LmcBase * self
                ){
                    const auto & graph = self->objective().graph();

                    typename LmcBase::NodeLabelsType nodeLabels(graph, 0);
                    {
                        py::gil_scoped_release allowThreads;
                        self->optimize(nodeLabels, nullptr);
                    }
                    ShapeType shape = {graph.nodeIdUpperBound() + 1};
                    xt::pytensor<uint64_t, 1> array = xt::zeros<uint64_t>(shape);
                    for(auto node : graph.nodes()){
                        array(node) = nodeLabels[node];
                    }
                    return array;

                }
            )
            .def("optimize",
                [](
                    LmcBase * self,
                    LmcVisitorBase * visitor
                ){
                    const auto & graph = self->objective().graph();

                    typename LmcBase::NodeLabelsType nodeLabels(graph,0);
                    {
                        py::gil_scoped_release allowThreads;
                        self->optimize(nodeLabels, visitor);
                    }
                    ShapeType shape = {graph.nodeIdUpperBound() + 1};
                    xt::pytensor<uint64_t, 1> array = xt::zeros<uint64_t>(shape);
                    for(auto node : graph.nodes()){
                        array(node) = nodeLabels[node];
                    }
                    return array;

                },
                py::arg("visitor")
            )
            .def("optimize",
                [](
                    LmcBase * self,
                    xt::pytensor<uint64_t, 1> & array
                ){
                    const auto & graph = self->objective().graph();
                    typename LmcBase::NodeLabelsType nodeLabels(graph,0);

                    if(array.size() == graph.nodeIdUpperBound()+1){
                        for(auto node : graph.nodes()){
                            nodeLabels[node] = array(node);
                        }
                        {
                            py::gil_scoped_release allowThreads;
                            self->optimize(nodeLabels, nullptr);
                        }
                        for(auto node : graph.nodes()){
                            array(node) = nodeLabels[node];
                        }
                        return array;
                    }
                    else{
                        throw std::runtime_error("input node labels have wrong shape");
                    }
                },
                py::arg("nodeLabels")
            )
            .def("optimize",
                [](
                    LmcBase * self,
                    LmcVisitorBase * visitor,
                    xt::pytensor<uint64_t, 1> & array
                ){
                    const auto & graph = self->objective().graph();
                    typename LmcBase::NodeLabelsType nodeLabels(graph,0);

                    if(array.size() == graph.nodeIdUpperBound()+1){
                        for(auto node : graph.nodes()){
                            nodeLabels[node] = array(node);
                        }
                        {
                            py::gil_scoped_release allowThreads;
                            self->optimize(nodeLabels, visitor);
                        }
                        for(auto node : graph.nodes()){
                            array(node) = nodeLabels[node];
                        }
                        return array;
                    }
                    else{
                        throw std::runtime_error("input node labels have wrong shape");
                    }
                },
                py::arg("visitor"),
                py::arg("nodeLabels")
            )
            ;
        ;
    }

    void exportLiftedMulticutBase(py::module & liftedMulticutModule) {

        {
            typedef PyUndirectedGraph GraphType;
            typedef LiftedMulticutObjective<GraphType, double> ObjectiveType;
            exportLiftedMulticutBaseT<ObjectiveType>(liftedMulticutModule);
        }
        {
            typedef nifty::graph::UndirectedGridGraph<2,true> GraphType;
            typedef LiftedMulticutObjective<GraphType, double> ObjectiveType;
            exportLiftedMulticutBaseT<ObjectiveType>(liftedMulticutModule);
        }
        {
            typedef nifty::graph::UndirectedGridGraph<3,true> GraphType;
            typedef LiftedMulticutObjective<GraphType, double> ObjectiveType;
            exportLiftedMulticutBaseT<ObjectiveType>(liftedMulticutModule);
        }
        //{
        //    typedef PyContractionGraph<PyUndirectedGraph> GraphType;
        //    typedef LiftedMulticutObjective<GraphType, double> ObjectiveType;
        //    exportLiftedMulticutBaseT<ObjectiveType>(liftedMulticutModule);
        //}
    }

}
} // namespace nifty::graph::opt
}
}
