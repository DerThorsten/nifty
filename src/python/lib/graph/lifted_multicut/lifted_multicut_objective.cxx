#include <pybind11/pybind11.h>
#include <iostream>
#include <sstream> 
#include <pybind11/numpy.h>

#include "nifty/python/graph/undirected_list_graph.hxx"
#include "nifty/python/graph/edge_contraction_graph.hxx"
#include "nifty/python/graph/lifted_multicut/lifted_multicut_objective.hxx"
#include "nifty/python/converter.hxx"

namespace py = pybind11;


namespace nifty{
namespace graph{
namespace lifted_multicut{

    template<class GRAPH>
    void exportLiftedMulticutObjectiveT(py::module & liftedMulticutModule) {

        typedef GRAPH Graph;
        typedef LiftedMulticutObjective<Graph, double> ObjectiveType;
        const auto clsName = LiftedMulticutObjectiveName<ObjectiveType>::name();

        auto liftedMulticutObjectiveCls = py::class_<ObjectiveType>(liftedMulticutModule, clsName.c_str());

        liftedMulticutObjectiveCls
            .def("setCost", &ObjectiveType::setCost,
                py::arg("u"),
                py::arg("v"),
                py::arg("weight"),
                py::arg("overwrite") = false
            )
            .def("setCosts",[]
            (
                ObjectiveType & objective,  
                nifty::marray::PyView<uint64_t> uvIds,
                nifty::marray::PyView<double>   weights,
                bool overwrite = false
            ){
                NIFTY_CHECK_OP(uvIds.dimension(),==,2,"wrong dimensions");
                NIFTY_CHECK_OP(weights.dimension(),==,1,"wrong dimensions");
                NIFTY_CHECK_OP(uvIds.shape(1),==,2,"wrong shape");
                NIFTY_CHECK_OP(uvIds.shape(0),==,weights.shape(0),"wrong shape");

                for(size_t i=0; i<uvIds.shape(0); ++i){
                    objective.setCost(uvIds(i,0), uvIds(i,1), weights(i));
                }
                
            })
            .def("evalNodeLabels",[](const ObjectiveType & objective,  nifty::marray::PyView<uint64_t> array){
               const auto & g = objective.graph();
               NIFTY_CHECK_OP(array.dimension(),==,1,"wrong dimensions");
               NIFTY_CHECK_OP(array.shape(0),==,g.nodeIdUpperBound()+1,"wrong shape");
               
               double sum = static_cast<double>(0.0);
               const auto & w = objective.weights();
               for(const auto edge: g.edges()){
                   const auto uv = g.uv(edge);
                   if(array(uv.first) != array(uv.second)){
                       sum += w[edge];
                   }
               }
               return sum;
           })
        ;


        // liftedMulticutModule.def("multicutObjective",
        //     [](const Graph & graph,  nifty::marray::PyView<double> array){
        //         NIFTY_CHECK_OP(array.dimension(),==,1,"wrong dimensions");
        //         NIFTY_CHECK_OP(array.shape(0),==,graph.edgeIdUpperBound()+1,"wrong shape");
                
        //         auto obj = new ObjectiveType(graph);
        //         auto & weights = obj->weights();
        //         graph.forEachEdge([&](int64_t edge){
        //             weights[edge] += array(edge);
        //         });
        //         return obj;
        //     },
        //     py::return_value_policy::take_ownership,
        //     py::keep_alive<0, 1>(),
        //     py::arg("graph"),py::arg("weights")  
        // );
    }

    void exportLiftedMulticutObjective(py::module & liftedMulticutModule) {

        {
            typedef PyUndirectedGraph GraphType;
            exportLiftedMulticutObjectiveT<GraphType>(liftedMulticutModule);
        }
    }

}
}
}
