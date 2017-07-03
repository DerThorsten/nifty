#include <pybind11/pybind11.h>
#include <iostream>
#include <sstream>
#include <pybind11/numpy.h>

#include "nifty/python/graph/undirected_list_graph.hxx"
#include "nifty/python/graph/edge_contraction_graph.hxx"
#include "nifty/python/graph/opt/minstcut/minstcut_objective.hxx"
#include "nifty/python/converter.hxx"

namespace py = pybind11;


namespace nifty{
namespace graph{
namespace opt{
namespace minstcut{

    template<class GRAPH>
    void exportMinstcutObjectiveT(py::module & minstcutModule) {

        typedef GRAPH GraphType;
        typedef MinstcutObjective<GraphType, double> ObjectiveType;
        const auto clsName = MinstcutObjectiveName<ObjectiveType>::name();

        auto minstcutObjectiveCls = py::class_<ObjectiveType>(minstcutModule, clsName.c_str());
        minstcutObjectiveCls
            .def_property_readonly("graph", &ObjectiveType::graph)
            .def("evalNodeLabels",[](const ObjectiveType & objective,  nifty::marray::PyView<uint64_t> array){
                return objective.evalNodeLabels(array);
            })
        ;


        minstcutModule.def("minstcutObjective",
            [](const GraphType & graph,  nifty::marray::PyView<double> array){
                NIFTY_CHECK_OP(array.dimension(),==,1,"wrong dimensions");
                NIFTY_CHECK_OP(array.shape(0),==,graph.edgeIdUpperBound()+1,"wrong shape");
                
                auto obj = new ObjectiveType(graph);
                auto & weights = obj->weights();
                //auto & unaries = obj->unaries();
                graph.forEachEdge([&](int64_t edge){
                    weights[edge] += array(edge);
                });
                return obj;
            },
            py::return_value_policy::take_ownership,
            py::keep_alive<0, 1>(),
            py::arg("graph"),py::arg("weights")  
        );
    }

    void exportMinstcutObjective(py::module & minstcutModule) {

        {
            typedef PyUndirectedGraph GraphType;
            exportMinstcutObjectiveT<GraphType>(minstcutModule);
        }
        {
            typedef PyContractionGraph<PyUndirectedGraph> GraphType;
            exportMinstcutObjectiveT<GraphType>(minstcutModule);
        }        

    }
 
} // namespace nifty::graph::opt::minstcut   
} // namespace nifty::graph::opt
}
}
