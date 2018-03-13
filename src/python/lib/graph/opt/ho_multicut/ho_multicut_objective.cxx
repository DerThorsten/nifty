#include <pybind11/pybind11.h>
#include <iostream>
#include <sstream>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <boost/format.hpp>

#include "nifty/python/graph/undirected_list_graph.hxx"
#include "nifty/python/graph/opt/ho_multicut/ho_multicut_objective.hxx"
#include "nifty/python/converter.hxx"

#include "xtensor-python/pyarray.hpp"
//#include "xtensor-python/pyarray.hpp"

namespace py = pybind11;


namespace nifty{
namespace graph{
namespace opt{
namespace ho_multicut{

    template<class GRAPH>
    void exportHoMulticutObjectiveT(py::module & hoMulticutModule) {

        typedef GRAPH GraphType;
        typedef HoMulticutObjective<GraphType, double> ObjectiveType;

        const auto graphClsName = GraphName<GraphType>::name();
        const auto clsName = HoMulticutObjectiveName<ObjectiveType>::name();

        auto hoMulticutObjectiveCls = py::class_<ObjectiveType>(hoMulticutModule, clsName.c_str(),
            (
                boost::format(
                        "HoMulticut objective for a graph of type nifty.graph.%s\n\n"
                        // "The hoMulticut objective function is given by:\n\n"
                        // ".. math::\n"
                        // "      E(y) = \\sum_{e \\in E} w_e \\cdot y_e \n\n"
                        // "      st. y \\in HoMulticutPolytop_{G}        \n\n"
                        // "This energy function can be used to find the optimal hoMulticut:\n\n"
                        // ".. math::\n"
                        // "      y^* = argmin_{y} \\sum_{e \\in E} w_e \\cdot y_e \n\n"
                        // "      st. y \\in HoMulticutPolytop_{G}                     \n"
                    )%graphClsName
            ).str().c_str()

        );
        hoMulticutObjectiveCls

            .def("__init__",
                [](
                    ObjectiveType & instance,
                    const GraphType & graph,  
                    nifty::marray::PyView<double> array
                ){
                    NIFTY_CHECK_OP(array.dimension(),==,1,"wrong dimensions");
                    NIFTY_CHECK_OP(array.shape(0),==,graph.edgeIdUpperBound()+1,"wrong shape");


                    new (&instance) ObjectiveType(graph);

                    auto & weights = instance.weights();
                    graph.forEachEdge([&](int64_t edge){
                        weights[edge] += array(edge);
                    });
                },
                py::keep_alive<1, 2>(),
                py::arg("graph"),
                py::arg("weights")
                ,
                (boost::format("Factory function to create a hoMulticut objective\n\n"
                "Args:\n"
                "   graph: (%s) : The graph\n"
                "   weights: (numpy.ndarray) : weights map\n\n"
                "Returns:\n"
                "  %s :  hoMulticut objective"
                )%graphClsName%clsName).str().c_str()
            )
            .def_property_readonly("graph", &ObjectiveType::graph)
            .def("evalNodeLabels",[](const ObjectiveType & objective,  nifty::marray::PyView<uint64_t> array){
                return objective.evalNodeLabels(array);
            })
            .def("addHigherOrderFactor",
            [](
                ObjectiveType & objective,
                xt::pyarray<double> valueTable,
                std::vector<uint64_t> edges
            )
            {
                objective.addHigherOrderFactor(valueTable, edges);
            })
        ;


        hoMulticutModule.def("hoMulticutObjective",
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

    void exportHoMulticutObjective(py::module & hoMulticutModule) {

        {
            typedef PyUndirectedGraph GraphType;
            exportHoMulticutObjectiveT<GraphType>(hoMulticutModule);
        }       

    }
} // namespace nifty::graph::opt::multicut
} // namespace nifty::graph::opt
}
}
