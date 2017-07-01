#include <cstddef>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "nifty/python/graph/undirected_grid_graph.hxx"
#include "nifty/python/graph/undirected_list_graph.hxx"
//#include "nifty/python/graph/edge_contraction_graph.hxx"
#include "nifty/python/graph/optimization/lifted_multicut/lifted_multicut_objective.hxx"

#include "nifty/python/converter.hxx"


#include "nifty/python/graph/optimization/lifted_multicut/py_lifted_multicut_base.hxx"



namespace py = pybind11;


namespace nifty{
namespace graph{
namespace optimization{
namespace lifted_multicut{

    using namespace py;
    //PYBIND11_DECLARE_HOLDER_TYPE(LmcBase, std::shared_ptr<LmcBase>);

    template<class OBJECTIVE>
    void exportLiftedMulticutBaseT(py::module & liftedMulticutModule) {


        typedef OBJECTIVE ObjectiveType;
        typedef PyLiftedMulticutBase<ObjectiveType> PyLmcBase;
        typedef LiftedMulticutBase<ObjectiveType> LmcBase;
        typedef LiftedMulticutEmptyVisitor<ObjectiveType> EmptyVisitor;
        typedef LiftedMulticutVisitorBase<ObjectiveType> LmcVisitorBase;
        //PYBIND11_DECLARE_HOLDER_TYPE(LmcBase, std::shared_ptr<LmcBase>);


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
                    //std::cout<<"without arg\n";
                    const auto & graph = self->objective().graph();
                    //std::cout<<"optimize that damn thing\n";



                    typename LmcBase::NodeLabelsType nodeLabels(graph,0);
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
                    LmcBase * self,
                    LmcVisitorBase * visitor
                ){
                    //std::cout<<"with visitor\n";
                    const auto & graph = self->objective().graph();
                    //std::cout<<"optimize that damn thing\n";



                    typename LmcBase::NodeLabelsType nodeLabels(graph,0);
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
                    LmcBase * self,
                    nifty::marray::PyView<uint64_t> array
                ){
                    //std::cout<<"opt array\n";
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
                    nifty::marray::PyView<uint64_t> array
                ){
                    //std::cout<<"opt with both\n";
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
        //{
        //    typedef PyContractionGraph<PyUndirectedGraph> GraphType;
        //    typedef LiftedMulticutObjective<GraphType, double> ObjectiveType;
        //    exportLiftedMulticutBaseT<ObjectiveType>(liftedMulticutModule);
        //}
    }

}
} // namespace nifty::graph::optimization
}
}
