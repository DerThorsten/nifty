#include <cstddef>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "xtensor-python/pytensor.hpp"

#include "nifty/python/graph/undirected_list_graph.hxx"
#include "nifty/python/graph/edge_contraction_graph.hxx"
#include "nifty/python/graph/opt/multicut/multicut_objective.hxx"

#include "nifty/python/graph/opt/multicut/py_multicut_base.hxx"



namespace py = pybind11;


namespace nifty{
namespace graph{
namespace opt{
namespace common{

    using namespace py;
    //PYBIND11_DECLARE_HOLDER_TYPE(McBase, std::shared_ptr<McBase>);

    template<class OBJECTIVE>
    void exportMulticutBaseT(py::module & multicutModule) {


        typedef OBJECTIVE ObjectiveType;
        typedef PyMulticutBase<ObjectiveType> PyMcBase;
        typedef MulticutBase<ObjectiveType> McBase;
        typedef MulticutEmptyVisitor<ObjectiveType> EmptyVisitor;
        typedef MulticutVisitorBase<ObjectiveType> McVisitorBase;
        //PYBIND11_DECLARE_HOLDER_TYPE(McBase, std::shared_ptr<McBase>);
        typedef xt::pytensor<uint64_t, 1>::shape_type ShapeType;


        const auto objName = MulticutObjectiveName<ObjectiveType>::name();
        const auto clsName = std::string("MulticutBase") + objName;
        // base factory
        py::class_<
            McBase,
            std::unique_ptr<McBase>,
            PyMcBase
        > mcBase(multicutModule, clsName.c_str());

        mcBase
            .def(py::init<>())

            .def("optimize",
                [](
                    McBase * self
                ){
                    const auto & graph = self->objective().graph();

                    typename McBase::NodeLabelsType nodeLabels(graph,0);
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
                    McBase * self,
                    McVisitorBase * visitor
                ){
                    const auto & graph = self->objective().graph();

                    typename McBase::NodeLabelsType nodeLabels(graph,0);
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
                    McBase * self,
                    xt::pytensor<uint64_t, 1> & array
                ){
                    const auto & graph = self->objective().graph();
                    typename McBase::NodeLabelsType nodeLabels(graph,0);

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
                    McBase * self,
                    McVisitorBase * visitor,
                    xt::pytensor<uint64_t, 1> & array
                ){
                    const auto & graph = self->objective().graph();
                    typename McBase::NodeLabelsType nodeLabels(graph,0);

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


} // namespace nifty::graph::opt::conmmon
} // namespace nifty::graph::opt
}
}
