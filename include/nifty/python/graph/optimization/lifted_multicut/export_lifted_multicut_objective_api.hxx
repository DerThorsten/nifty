#pragma once 

#include <pybind11/pybind11.h>
#include <iostream>
#include <sstream> 
#include <pybind11/numpy.h>

#include "nifty/python/graph/undirected_list_graph.hxx"
#include "nifty/python/graph/optimization/lifted_multicut/lifted_multicut_objective.hxx"
#include "nifty/python/converter.hxx"


namespace py = pybind11;


namespace nifty{
namespace graph{
namespace lifted_multicut{


    template<class OBJECTIVE, class PY_OBJECTIVE>
    void exportLiftedMulticutObjectiveApi(
        PY_OBJECTIVE & liftedMulticutObjectiveCls
    ) {
        typedef OBJECTIVE ObjectiveType;
        typedef typename ObjectiveType::GraphType GraphType;
        typedef typename ObjectiveType::LiftedGraphType LiftedGraphType;


        liftedMulticutObjectiveCls
            .def_property_readonly("numberOfLiftedEdges", [](const ObjectiveType & obj){
                return obj.numberOfLiftedEdges();
            })
            

            .def_property_readonly("graph", &ObjectiveType::graph)
            .def_property_readonly("liftedGraph", 
                    [](const ObjectiveType & self) -> const LiftedGraphType & {
                    return self.liftedGraph();
                },
                py::return_value_policy::reference_internal
            )

            .def("evalNodeLabels",[](const ObjectiveType & objective,  nifty::marray::PyView<uint64_t> array){
                return objective.evalNodeLabels(array);
            })
            .def_property_readonly("graph", &ObjectiveType::graph)
            .def_property_readonly("liftedGraph", 
                    [](const ObjectiveType & self) -> const LiftedGraphType & {
                    return self.liftedGraph();
                },
                py::return_value_policy::reference_internal
            )

            .def("liftedUvIds",
                [](ObjectiveType & self) {
                    nifty::marray::PyView<uint64_t> out({uint64_t(self.numberOfLiftedEdges()), uint64_t(2)});
                    auto i = 0; 
                    self.forEachLiftedeEdge([&](const uint64_t edge){
                        const auto uv = self.liftedGraph().uv(edge);
                        out(i,0) = uv.first;
                        out(i,1) = uv.second;
                        ++i;
                    });
                    
                    return out;
                }
            )

        ;
    }

}
}
}
