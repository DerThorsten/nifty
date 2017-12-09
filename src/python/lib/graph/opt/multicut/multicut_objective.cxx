#include <pybind11/pybind11.h>
#include <iostream>
#include <sstream>
#include <pybind11/numpy.h>

#include <boost/format.hpp>

#include "nifty/python/graph/undirected_list_graph.hxx"
#include "nifty/python/graph/edge_contraction_graph.hxx"
#include "nifty/python/graph/opt/multicut/multicut_objective.hxx"
#include "nifty/python/converter.hxx"

namespace py = pybind11;


namespace nifty{
namespace graph{
namespace opt{
namespace multicut{

    template<class GRAPH>
    void exportMulticutObjectiveT(py::module & multicutModule) {

        typedef GRAPH GraphType;
        typedef MulticutObjective<GraphType, double> ObjectiveType;

        const auto graphClsName = GraphName<GraphType>::name();
        const auto clsName = MulticutObjectiveName<ObjectiveType>::name();

        auto multicutObjectiveCls = py::class_<ObjectiveType>(multicutModule, clsName.c_str(),
            (
                boost::format(
                        "Multicut objective for a graph of type nifty.graph.%s\n\n"
                        "The multicut objective function is given by:\n\n"
                        ".. math::\n"
                        "      E(y) = \\sum_{e \\in E} w_e \\cdot y_e \n\n"
                        "      st. y \\in MulticutPolytop_{G}        \n\n"
                        "This energy function can be used to find the optimal multicut:\n\n"
                        ".. math::\n"
                        "      y^* = argmin_{y} \\sum_{e \\in E} w_e \\cdot y_e \n\n"
                        "      st. y \\in MulticutPolytop_{G}                     \n"
                    )%graphClsName
            ).str().c_str()

        );
        multicutObjectiveCls

            .def(py::init([](const GraphType & graph,
                            nifty::marray::PyView<double> array){
                    NIFTY_CHECK_OP(array.dimension(),==,1,"wrong dimensions");
                    NIFTY_CHECK_OP(array.shape(0),==,graph.edgeIdUpperBound()+1,"wrong shape");

                    auto objective = new ObjectiveType(graph);

                    auto & weights = objective->weights();
                    graph.forEachEdge([&](int64_t edge){
                        weights[edge] += array(edge);
                    });
                    return objective;
                }),
                py::keep_alive<1, 2>(),
                py::arg("graph"),
                py::arg("weights")
                ,
                (boost::format("Factory function to create a multicut objective\n\n"
                "Args:\n"
                "   graph: (%s) : The graph\n"
                "   weights: (numpy.ndarray) : weights map\n\n"
                "Returns:\n"
                "  %s :  multicut objective"
                )%graphClsName%clsName).str().c_str()
            )

            .def_property_readonly("graph", &ObjectiveType::graph)

            .def("evalNodeLabels",[](const ObjectiveType & objective,
                                     nifty::marray::PyView<uint64_t> array){
                return objective.evalNodeLabels(array);
            })
        ;


        multicutModule.def("multicutObjective",
            [](const GraphType & graph,  nifty::marray::PyView<double> array){
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
} // namespace nifty::graph::opt::multicut
} // namespace nifty::graph::opt
}
}
