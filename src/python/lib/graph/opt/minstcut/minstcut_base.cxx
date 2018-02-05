#include <cstddef>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "nifty/python/graph/undirected_list_graph.hxx"
#include "nifty/python/graph/edge_contraction_graph.hxx"
#include "nifty/python/graph/opt/minstcut/minstcut_objective.hxx"

#include "nifty/python/converter.hxx"


#include "nifty/python/graph/opt/minstcut/py_minstcut_base.hxx"



namespace py = pybind11;


namespace nifty{
namespace graph{
namespace opt{
namespace minstcut{

    using namespace py;
    //PYBIND11_DECLARE_HOLDER_TYPE(McBase, std::shared_ptr<McBase>);

    template<class OBJECTIVE>
    void exportMinstcutBaseT(py::module & minstcutModule) {


        typedef OBJECTIVE ObjectiveType;
        typedef PyMinstcutBase<ObjectiveType> PyMcBase;
        typedef MinstcutBase<ObjectiveType> McBase;
        typedef MinstcutEmptyVisitor<ObjectiveType> EmptyVisitor;
        typedef MinstcutVisitorBase<ObjectiveType> McVisitorBase;
        //PYBIND11_DECLARE_HOLDER_TYPE(McBase, std::shared_ptr<McBase>);


        const auto objName = MinstcutObjectiveName<ObjectiveType>::name();
        const auto clsName = std::string("MinstcutBase") + objName;
        // base factory
        py::class_<
            McBase,
            std::unique_ptr<McBase>,
            PyMcBase
        > mcBase(minstcutModule, clsName.c_str());

        mcBase
            .def(py::init<>())

            .def("optimize",
                [](
                    McBase * self
                ){
                    //std::cout<<"without arg\n";
                    const auto & graph = self->objective().graph();
                    //std::cout<<"optimize that damn thing\n";



                    typename McBase::NodeLabelsType nodeLabels(graph,0);
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



                    typename McBase::NodeLabelsType nodeLabels(graph,0);
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
                    nifty::marray::PyView<uint64_t> array
                ){
                    //std::cout<<"opt with both\n";
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

    void exportMinstcutBase(py::module & minstcutModule) {

        {
            typedef PyUndirectedGraph GraphType;
            typedef MinstcutObjective<GraphType, double> ObjectiveType;
            exportMinstcutBaseT<ObjectiveType>(minstcutModule);
        }
        {
            typedef PyContractionGraph<PyUndirectedGraph> GraphType;
            typedef MinstcutObjective<GraphType, double> ObjectiveType;
            exportMinstcutBaseT<ObjectiveType>(minstcutModule);
        }
    }

} // namespace nifty::graph::opt::minstcut
} // namespace nifty::graph::opt
}
}
