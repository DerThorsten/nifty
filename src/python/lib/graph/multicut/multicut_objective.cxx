#include <pybind11/pybind11.h>
#include <iostream>
#include <sstream>
#include <pybind11/numpy.h>

#include "nifty/python/graph/undirected_list_graph.hxx"
#include "nifty/python/graph/edge_contraction_graph.hxx"
#include "nifty/python/graph/multicut/multicut_objective.hxx"
#include "nifty/python/converter.hxx"

namespace py = pybind11;


namespace nifty{
namespace graph{


    template<class GRAPH>
    void exportMulticutObjectiveT(py::module & multicutModule) {

        typedef GRAPH Graph;
        typedef MulticutObjective<Graph, double> ObjectiveType;
        const auto clsName = MulticutObjectiveName<ObjectiveType>::name();

        auto multicutObjectiveCls = py::class_<ObjectiveType>(multicutModule, clsName.c_str());
        multicutObjectiveCls
            .def("evalNodeLabels",[](const ObjectiveType & objective,  nifty::marray::PyView<uint64_t> array){
                return objective.evalNodeLabels(array);
            })
        ;


        multicutModule.def("multicutObjective",
            [](const Graph & graph,  nifty::marray::PyView<double> array){
                NIFTY_CHECK_OP(array.dimension(),==,1,"wrong dimensions");
                NIFTY_CHECK_OP(array.shape(0),==,graph.edgeIdUpperBound()+1,"wrong shape");
                
                auto obj = new ObjectiveType(graph);
                auto & weights = obj->weights();
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

    void exportMulticutObjective(py::module & multicutModule) {

        {
            typedef PyUndirectedGraph GraphType;
            exportMulticutObjectiveT<GraphType>(multicutModule);
        }
        {
            typedef PyContractionGraph<PyUndirectedGraph> GraphType;
            exportMulticutObjectiveT<GraphType>(multicutModule);
        }        

    }

}
}
