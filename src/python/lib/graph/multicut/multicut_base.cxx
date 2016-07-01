#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "nifty/graph/multicut/multicut_objective.hxx"
#include "nifty/graph/simple_graph.hxx"

#include "../../converter.hxx"
#include "py_multicut_base.hxx"




namespace py = pybind11;


namespace nifty{
namespace graph{

    typedef UndirectedGraph<> Graph;
    typedef MulticutObjective<Graph, double> Objective;
    typedef PyMulticutBase<Objective> PyMcBase;
    typedef MulticutBase<Objective> McBase;

    using namespace py;
    //PYBIND11_DECLARE_HOLDER_TYPE(McBase, std::shared_ptr<McBase>);

    void exportMulticutBase(py::module & multicutModule) {

        typedef UndirectedGraph<> Graph;
        typedef MulticutObjective<Graph, double> Objective;
        typedef PyMulticutBase<Objective> PyMcBase;
        typedef MulticutBase<Objective> McBase;
        typedef MulticutEmptyVisitor<Objective> EmptyVisitor;
        typedef MulticutVisitorBase<Objective> McVisitorBase;
        //PYBIND11_DECLARE_HOLDER_TYPE(McBase, std::shared_ptr<McBase>);

        // base factory
        py::class_<
            McBase, 
            std::shared_ptr<McBase>, 
            PyMcBase 
        > mcBase(multicutModule, "MulticutBaseUndirectedGraph");
        
        mcBase
            .def(py::init<>())
            .def("optimizeWithVisitor", 
                [](
                    McBase * self,
                    McVisitorBase * visitor,
                    nifty::marray::PyView<uint64_t> array
                ){
                    const auto graph = self->objective().graph();
                    //std::cout<<"optimize that damn thing\n";
            


                    typename McBase::NodeLabels nodeLabels(graph,0);

                    if(array.size() == 0 ){

                        {
                            py::gil_scoped_release allowThreads;
                            self->optimize(nodeLabels, visitor);
                        }
                        std::vector<size_t> shape = {size_t(graph.numberOfNodes())};
                        nifty::marray::PyView<uint64_t> rarray(shape.begin(),shape.end());
                        for(auto node : graph.nodes()){
                            rarray(node) = nodeLabels[node];
                        }
                        return rarray;

                    }
                    else if(array.size() == graph.numberOfNodes()){
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
                py::arg_t< py::array_t<uint64_t> >("nodeLabels", py::list() )
            )
            ;
        ;

    }

}
}
    
