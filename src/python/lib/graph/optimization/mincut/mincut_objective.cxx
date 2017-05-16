#include <pybind11/pybind11.h>
#include <iostream>
#include <sstream>
#include <pybind11/numpy.h>

#include "nifty/python/graph/undirected_list_graph.hxx"
#include "nifty/python/graph/edge_contraction_graph.hxx"
#include "nifty/python/graph/optimization/mincut/mincut_objective.hxx"
#include "nifty/python/converter.hxx"

namespace py = pybind11;


namespace nifty{
namespace graph{


    template<class GRAPH>
    void exportMincutObjectiveT(py::module & mincutModule) {

        typedef GRAPH Graph;
        typedef MincutObjective<Graph, double> ObjectiveType;
        const auto clsName = MincutObjectiveName<ObjectiveType>::name();

        auto mincutObjectiveCls = py::class_<ObjectiveType>(mincutModule, clsName.c_str());
        mincutObjectiveCls
            .def_property_readonly("graph", &ObjectiveType::graph)
            .def("evalNodeLabels",[](const ObjectiveType & objective,  nifty::marray::PyView<uint64_t> array){
                return objective.evalNodeLabels(array);
            })
        ;


        mincutModule.def("mincutObjective",
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

    void exportMincutObjective(py::module & mincutModule) {

        {
            typedef PyUndirectedGraph GraphType;
            exportMincutObjectiveT<GraphType>(mincutModule);
        }
        {
            typedef PyContractionGraph<PyUndirectedGraph> GraphType;
            exportMincutObjectiveT<GraphType>(mincutModule);
        }        

    }

}
}
