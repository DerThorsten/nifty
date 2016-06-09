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
        typedef MulticutVerboseVisitor<Objective> VerboseVisitor;
        //PYBIND11_DECLARE_HOLDER_TYPE(McBase, std::shared_ptr<McBase>);

        // base factory
        py::class_<
            McBase, 
            std::shared_ptr<McBase>, 
            PyMcBase 
        > mcBase(multicutModule, "MulticutBaseUndirectedGraph");
        
        mcBase
            .def(py::init<>())
            .def("optimize", 
                [](
                    McBase * self,
                    py::array_t<uint64_t> pyArray
                ){
                    const auto graph = self->objective().graph();
                    std::cout<<"optimize that damn thing\n";
                    NumpyArray<uint64_t> array(pyArray);


                    typename McBase::NodeLabels nodeLabels(graph,0);
                    VerboseVisitor visitor;

                    if(array.size() == 0 ){

                        self->optimize(nodeLabels, &visitor);

                        std::vector<size_t> strides = {sizeof(uint64_t)};
                        std::vector<size_t> shape = {size_t(graph.numberOfNodes())};
                        size_t ndim = 1;

                        py::array_t<uint64_t> retArray =  py::array(py::buffer_info(NULL, sizeof(uint64_t),
                            py::format_descriptor<uint64_t>::value,
                            ndim, shape, strides)
                        );
                        NumpyArray<uint64_t> rarray(retArray);
                        for(auto node : graph.nodes()){
                            rarray(node) = nodeLabels[node];
                        }
                        return retArray;


                    }
                    else if(array.size() == graph.numberOfNodes()){
                        for(auto node : graph.nodes()){
                            nodeLabels[node] = array(node);
                        }
                        self->optimize(nodeLabels, &visitor);
                        for(auto node : graph.nodes()){
                             array(node) = nodeLabels[node];
                        }
                        return pyArray;
                    }
                    else{
                        throw std::runtime_error("input node labels have wrong shape");
                    }
                },
                py::arg_t< py::array_t<uint64_t> >("nodeLabels", py::list() )
            );
        ;

    }

}
}
