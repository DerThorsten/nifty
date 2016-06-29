#include <pybind11/pybind11.h>
#include <iostream>
#include <sstream>
#include <pybind11/numpy.h>

#include "nifty/graph/simple_graph.hxx"
#include "nifty/graph/multicut/multicut_objective.hxx"
#include "nifty/graph/multicut/perturb_and_map.hxx"
#include "../../converter.hxx"

namespace py = pybind11;


PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);



namespace nifty{
namespace graph{



    void exportPerturbAndMap(py::module & multicutModule) {

        typedef UndirectedGraph<> Graph;
        typedef typename Graph:: template EdgeMap<double>   EdgeState;
        typedef typename Graph:: template NodeMap<uint64_t> NodeLabels;
        typedef MulticutObjective<Graph, double> McObjective;
        typedef PerturbAndMap<McObjective> PerturbAndMapType;
        typedef typename PerturbAndMapType::Settings PerturbAndMapSettingsType;
        const auto clsName = std::string("PerturbAndMapUndirectedGraph");
        const auto settingsClsName = std::string("PerturbAndMapSettingsUndirectedGraph");


        auto multicutObjectiveCls = py::class_<PerturbAndMapType>(multicutModule, clsName.c_str())
            .def("optimize",
            [](
                PerturbAndMapType * self,
                py::array_t<uint64_t> pyNodeLabels
            )
            {
                const auto & graph = self->graph();
                const auto nNodes = graph.numberOfNodes();
                const auto nEdges = graph.numberOfEdges();

                NumpyArray<uint64_t> nodeLabelsArray(pyNodeLabels);

                EdgeState edgeState(graph);

                if(nodeLabelsArray.size() == 0 ){
                    py::gil_scoped_release allowThreads;
                    self->optimize(edgeState);
                }
                else{
                    NIFTY_CHECK_OP(nodeLabelsArray.size(),==,graph.numberOfNodes(),"nodes labels has wrong shape");
                    py::gil_scoped_release allowThreads;
                    NodeLabels nodeLabels(graph);
                    for(auto node : graph.nodes()){
                        nodeLabels[node] = nodeLabelsArray(node);
                    }
                    self->optimize(nodeLabels, edgeState);
                }


                NumpyArray<double> rarray({nEdges},{1});
                for(auto edge: graph.edges())
                    rarray(edge) = edgeState[edge];
                return rarray.pyArray();
    

            },
            py::arg_t< py::array_t<uint64_t> >("nodeLabels", py::list() )
            )
        ;


        py::class_<PerturbAndMapSettingsType>(multicutModule, settingsClsName.c_str())
            .def(py::init<>())
            .def_readwrite("mcFactory",&PerturbAndMapSettingsType::mcFactory)
        ;



        multicutModule.def("perturbAndMap",
            [](const McObjective & objective, const PerturbAndMapSettingsType & s){
                auto perturbAndMap = new PerturbAndMapType(objective);
                return perturbAndMap;
            },
            py::return_value_policy::take_ownership,
            py::keep_alive<0, 1>(),
            py::arg("objective"),py::arg("settings")
        );

        
    }

}
}
