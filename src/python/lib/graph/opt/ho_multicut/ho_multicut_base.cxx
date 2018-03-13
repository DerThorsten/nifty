#include <cstddef>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "nifty/python/graph/undirected_list_graph.hxx"
#include "nifty/python/graph/edge_contraction_graph.hxx"
#include "nifty/python/graph/opt/ho_multicut/ho_multicut_objective.hxx"

#include "nifty/python/converter.hxx"


#include "nifty/python/graph/opt/ho_multicut/py_ho_multicut_base.hxx"



namespace py = pybind11;


namespace nifty{
namespace graph{
namespace opt{
namespace ho_multicut{

    using namespace py;
    //PYBIND11_DECLARE_HOLDER_TYPE(McBase, std::shared_ptr<McBase>);

    template<class OBJECTIVE>
    void exportHoMulticutBaseT(py::module & hoMulticutModule) {


        typedef OBJECTIVE ObjectiveType;
        typedef PyHoMulticutBase<ObjectiveType> PyMcBase;
        typedef HoMulticutBase<ObjectiveType> McBase;
        typedef HoMulticutEmptyVisitor<ObjectiveType> EmptyVisitor;
        typedef HoMulticutVisitorBase<ObjectiveType> McVisitorBase;
        //PYBIND11_DECLARE_HOLDER_TYPE(McBase, std::shared_ptr<McBase>);


        const auto objName = HoMulticutObjectiveName<ObjectiveType>::name();
        const auto clsName = std::string("HoMulticutBase") + objName;
        // base factory
        py::class_<
            McBase,
            std::unique_ptr<McBase>,
            PyMcBase
        > mcBase(hoMulticutModule, clsName.c_str());

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

    void exportHoMulticutBase(py::module & hoMulticutModule) {

        {
            typedef PyUndirectedGraph GraphType;
            typedef HoMulticutObjective<GraphType, double> ObjectiveType;
            exportHoMulticutBaseT<ObjectiveType>(hoMulticutModule);
        }
    }

} // namespace nifty::graph::opt::ho_multicut
} // namespace nifty::graph::opt
}
}
