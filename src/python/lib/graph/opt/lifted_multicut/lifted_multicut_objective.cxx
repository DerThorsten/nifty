#include <pybind11/pybind11.h>
#include <iostream>
#include <sstream>
#include <pybind11/numpy.h>
#include "xtensor-python/pytensor.hpp"

#include "nifty/python/graph/undirected_list_graph.hxx"
#include "nifty/python/graph/undirected_grid_graph.hxx"

#include "nifty/python/graph/opt/lifted_multicut/lifted_multicut_objective.hxx"
#include "nifty/python/converter.hxx"

namespace py = pybind11;


namespace nifty{
namespace graph{
namespace opt{
namespace lifted_multicut{

    template<class GRAPH>
    void exportLiftedMulticutObjectiveT(py::module & liftedMulticutModule) {

        typedef GRAPH GraphType;
        typedef LiftedMulticutObjective<GraphType, double> ObjectiveType;
        typedef typename ObjectiveType::LiftedGraphType LiftedGraphType;
        const auto clsName = LiftedMulticutObjectiveName<ObjectiveType>::name();

        auto liftedMulticutObjectiveCls = py::class_<ObjectiveType>(liftedMulticutModule,
                                                                    clsName.c_str());

        liftedMulticutObjectiveCls
            .def_property_readonly("numberOfLiftedEdges", [](const ObjectiveType & obj){
                return obj.numberOfLiftedEdges();
            })
            .def("setCost", &ObjectiveType::setCost,
                py::arg("u"),
                py::arg("v"),
                py::arg("weight"),
                py::arg("overwrite")=false)

            .def("setCosts",[](ObjectiveType & objective,
                               xt::pytensor<uint64_t, 1> uvIds,
                               xt::pytensor<double, 1> weights,
                               bool overwrite){
                NIFTY_CHECK_OP(uvIds.dimension(),==,2,"wrong dimensions");
                NIFTY_CHECK_OP(weights.dimension(),==,1,"wrong dimensions");
                NIFTY_CHECK_OP(uvIds.shape()[1],==,2,"wrong shape");
                NIFTY_CHECK_OP(uvIds.shape()[0],==,weights.shape()[0],"wrong shape");

                for(size_t i=0; i < uvIds.shape()[0]; ++i){
                    objective.setCost(uvIds(i, 0), uvIds(i, 1), weights(i), overwrite);
                }
            },
                py::arg("uv"),
                py::arg("weight"),
                py::arg("overwrite")=false
            )

            .def("setLiftedEdgesCosts",[](ObjectiveType & objective,
                                          const xt::pytensor<double, 1> & weights,
                                          bool overwrite){
                NIFTY_CHECK_OP(weights.dimension(),==,1,"wrong dimensions");
                NIFTY_CHECK_OP(weights.shape()[0], ==,
                               objective.numberOfLiftedEdges(), "wrong shape");

                auto c = 0;
                objective.forEachLiftedeEdge([&](const uint64_t edge){
                    const auto uv = objective.liftedGraph().uv(edge);
                    objective.setCost(uv.first, uv.second, weights[c], overwrite);
                    ++c;
                });
            },
                py::arg("weights"),
                py::arg("overwrite")=false
            )

            .def("setGraphEdgesCosts",[](ObjectiveType & objective,
                                         xt::pytensor<double, 1> & weights,
                                         bool overwrite){
                NIFTY_CHECK_OP(weights.dimension(), ==, 1, "wrong dimensions");
                NIFTY_CHECK_OP(weights.shape()[0], ==,
                        objective.graph().numberOfEdges(),"wrong shape");

                auto c = 0;
                objective.forEachGraphEdge([&](const uint64_t edge){
                    const auto uv = objective.liftedGraph().uv(edge);
                    objective.setCost(uv.first, uv.second, weights[c], overwrite);
                    ++c;
                });
            },
                py::arg("weights"),
                py::arg("overwrite")=false
            )


            .def("evalNodeLabels",[](const ObjectiveType & objective,
                                     xt::pytensor<uint64_t, 1> & array){
                return objective.evalNodeLabels(array);
            })
            .def_property_readonly("graph", &ObjectiveType::graph)
            .def_property_readonly("liftedGraph",
                    [](const ObjectiveType & self) -> const LiftedGraphType & {
                    return self.liftedGraph();
                },
                py::return_value_policy::reference_internal
            )
            .def("_insertLiftedEdgesBfs",
                [](ObjectiveType & self, const uint32_t maxDistance){
                    self.insertLiftedEdgesBfs(maxDistance);
                },
                py::arg("maxDistance")
            )

            .def("_insertLiftedEdgesBfsReturnDist",
                [](ObjectiveType & self, const uint32_t maxDistance){

                    std::vector<uint32_t> dist;
                    self.insertLiftedEdgesBfs(maxDistance, dist);

                    typedef typename xt::pytensor<uint64_t, 1>::shape_type ShapeType;
                    ShapeType shape = {int64_t(dist.size())};
                    xt::pytensor<uint64_t, 1> array(shape);

                    for(size_t i=0; i<dist.size(); ++i){
                        array(i) = dist[i];
                    }
                    return array;
                },
                py::arg("maxDistance")
            )
            .def("liftedUvIds",
                [](ObjectiveType & self) {
                    xt::pytensor<uint64_t, 2> out({int64_t(self.numberOfLiftedEdges()), 2L});
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


        liftedMulticutModule.def("liftedMulticutObjective",
            [](const GraphType & graph){

                auto obj = new ObjectiveType(graph);
                return obj;
            },
            py::return_value_policy::take_ownership,
            py::keep_alive<0, 1>(),
            py::arg("graph")
        );
    }

    void exportLiftedMulticutObjective(py::module & liftedMulticutModule) {

        {
            typedef PyUndirectedGraph GraphType;
            exportLiftedMulticutObjectiveT<GraphType>(liftedMulticutModule);
        }
        {
            typedef nifty::graph::UndirectedGridGraph<2,true> GraphType;
            exportLiftedMulticutObjectiveT<GraphType>(liftedMulticutModule);
        }
        {
            typedef nifty::graph::UndirectedGridGraph<3,true> GraphType;
            exportLiftedMulticutObjectiveT<GraphType>(liftedMulticutModule);
        }
    }

}
} // namespace nifty::graph::opt
}
}
