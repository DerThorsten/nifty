#include <pybind11/pybind11.h>
#include <iostream>
#include <sstream>
#include <pybind11/numpy.h>
#include "xtensor-python/pytensor.hpp"

#include "nifty/python/graph/undirected_list_graph.hxx"
#include "nifty/python/graph/edge_contraction_graph.hxx"
#include "nifty/python/graph/opt/minstcut/minstcut_objective.hxx"

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
            .def("evalNodeLabels",[](const ObjectiveType & objective,
                                     xt::pytensor<uint64_t, 1> & array){
                return objective.evalNodeLabels(array);
            })
        ;


        minstcutModule.def("minstcutObjective",
            [](const GraphType & graph,
               xt::pytensor<double, 1> & weightsArray,
               xt::pytensor<double, 1> & unrariesArray){

                NIFTY_CHECK_OP(weightsArray.dimension(),==,1,"wrong dimensions");
                NIFTY_CHECK_OP(weightsArray.shape()[0],==,graph.edgeIdUpperBound()+1,
                               "wrong shape");

                NIFTY_CHECK_OP(unrariesArray.dimension(),==,2,"wrong dimensions");
                NIFTY_CHECK_OP(unrariesArray.shape()[0],==,graph.nodeIdUpperBound()+1,
                               "wrong shape");
                NIFTY_CHECK_OP(unrariesArray.shape()[1],==,2, "wrong shape");

                auto obj = new ObjectiveType(graph);
                auto & weights = obj->weights();
                auto & unaries = obj->unaries();

                graph.forEachEdge([&](int64_t edge){
                    weights[edge] += weightsArray(edge);
                });

                graph.forEachNode([&](int64_t node){
                    unaries[node].first += unrariesArray(node,0);
                    unaries[node].second += unrariesArray(node,1);
                });


                return obj;
            },
            py::return_value_policy::take_ownership,
            py::keep_alive<0, 1>(),
            py::arg("graph"),py::arg("weights"),py::arg("unaries")
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
