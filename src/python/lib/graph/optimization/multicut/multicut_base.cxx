#include <cstddef>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "nifty/python/graph/undirected_list_graph.hxx"
#include "nifty/python/graph/edge_contraction_graph.hxx"
#include "nifty/python/graph/optimization/multicut/multicut_objective.hxx"

#include "nifty/python/converter.hxx"


#include "nifty/python/graph/optimization/multicut/py_multicut_base.hxx"



namespace py = pybind11;


namespace nifty{
namespace graph{
namespace optimization{
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
        //PYBIND11_DECLARE_HOLDER_TYPE(McBase, std::shared_ptr<McBase>);


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
                    //std::cout<<"without arg\n";
                    const auto & graph = self->objective().graph();
                    //std::cout<<"optimize that damn thing\n";



                    typename McBase::NodeLabels nodeLabels(graph,0);
                    {
                        py::gil_scoped_release allowThreads;
                        self->optimize(nodeLabels, nullptr);
                    }
                    std::vector<std::size_t> shape = {std::size_t(graph.nodeIdUpperBound()+1)};
                    nifty::marray::PyView<uint64_t> array(shape.begin(),shape.end());
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
                    //std::cout<<"with visitor\n";
                    const auto & graph = self->objective().graph();
                    //std::cout<<"optimize that damn thing\n";



                    typename McBase::NodeLabels nodeLabels(graph,0);
                    {
                        py::gil_scoped_release allowThreads;
                        self->optimize(nodeLabels, visitor);
                    }
                    std::vector<std::size_t> shape = {std::size_t(graph.nodeIdUpperBound()+1)};
                    nifty::marray::PyView<uint64_t> array(shape.begin(),shape.end());
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
                    nifty::marray::PyView<uint64_t> array
                ){
                    //std::cout<<"opt array\n";
                    const auto & graph = self->objective().graph();
                    typename McBase::NodeLabels nodeLabels(graph,0);


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
                    nifty::marray::PyView<uint64_t> array
                ){
                    //std::cout<<"opt with both\n";
                    const auto & graph = self->objective().graph();
                    typename McBase::NodeLabels nodeLabels(graph,0);


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

} // namespace nifty::graph::optimization::multicut
} // namespace nifty::graph::optimization
}
}
