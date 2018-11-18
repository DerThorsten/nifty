#include <cstddef>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <xtensor-python/pytensor.hpp>

#include "nifty/python/graph/undirected_list_graph.hxx"
#include "nifty/python/graph/edge_contraction_graph.hxx"
#include "nifty/python/graph/opt/multicut/multicut_objective.hxx"

#include "nifty/python/graph/opt/multicut/py_multicut_base.hxx"



namespace py = pybind11;


namespace nifty{
namespace graph{
namespace opt{
namespace multicut{

    using namespace py;
    //PYBIND11_DECLARE_HOLDER_TYPE(McBase, std::shared_ptr<McBase>);

    template<class OBJECTIVE>
    void exportMulticutBaseT(py::module & multicutModule) {


        typedef OBJECTIVE ObjectiveType;
        typedef PyMulticutBase<ObjectiveType> PyMcBase;
        typedef MulticutBase<ObjectiveType> McBase;
        typedef MulticutEmptyVisitor<ObjectiveType> EmptyVisitor;
        typedef MulticutVisitorBase<ObjectiveType> McVisitorBase;
        typedef typename xt::pytensor<uint64_t, 1>::shape_type ShapeType;


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
                    ShapeType shape = {graph.nodeIdUpperBound() + 1};
                    xt::pytensor<uint64_t, 1> array(shape);
                    {
                        py::gil_scoped_release allowThreads;
                        self->optimize(nodeLabels, nullptr);
                        for(auto node : graph.nodes()){
                            array(node) = nodeLabels[node];
                        }
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
                    ShapeType shape = {graph.nodeIdUpperBound() + 1};
                    xt::pytensor<uint64_t, 1> array(shape);
                    {
                        py::gil_scoped_release allowThreads;
                        self->optimize(nodeLabels, visitor);
                        for(auto node : graph.nodes()){
                            array(node) = nodeLabels[node];
                        }
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
                    {
                        py::gil_scoped_release allowThreads;
                        const auto & graph = self->objective().graph();
                        typename McBase::NodeLabelsType nodeLabels(graph,0);
                        if(array.size() != graph.nodeIdUpperBound()+1){
                            throw std::runtime_error("input node labels have wrong shape");
                        }
                        for(auto node : graph.nodes()){
                            nodeLabels[node] = array(node);
                        }
                        self->optimize(nodeLabels, nullptr);
                        for(auto node : graph.nodes()){
                            array(node) = nodeLabels[node];
                        }
                    }
                    return array;
                },
                py::arg("nodeLabels")
            )

            .def("optimize",
                [](
                    McBase * self,
                    McVisitorBase * visitor,
                    xt::pytensor<uint64_t, 1> & array
                ){
                    {
                        py::gil_scoped_release allowThreads;
                        const auto & graph = self->objective().graph();
                        typename McBase::NodeLabelsType nodeLabels(graph,0);

                        if(array.size() != graph.nodeIdUpperBound()+1){
                            throw std::runtime_error("input node labels have wrong shape");
                        }

                        for(auto node : graph.nodes()){
                            nodeLabels[node] = array(node);
                        }
                        self->optimize(nodeLabels, visitor);
                        for(auto node : graph.nodes()){
                            array(node) = nodeLabels[node];
                        }
                    }
                    return array;
                },
                py::arg("visitor"),
                py::arg("nodeLabels")
            )
            ;
        ;
    }

    void exportMulticutBase(py::module & multicutModule) {

        {
            typedef PyUndirectedGraph GraphType;
            typedef MulticutObjective<GraphType, double> ObjectiveType;
            exportMulticutBaseT<ObjectiveType>(multicutModule);
        }
        {
            typedef PyContractionGraph<PyUndirectedGraph> GraphType;
            typedef MulticutObjective<GraphType, double> ObjectiveType;
            exportMulticutBaseT<ObjectiveType>(multicutModule);
        }
    }

} // namespace nifty::graph::opt::multicut
} // namespace nifty::graph::opt
}
}
