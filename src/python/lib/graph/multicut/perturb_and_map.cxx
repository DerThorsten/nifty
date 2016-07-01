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


        auto pAndMapCls = py::class_<PerturbAndMapType>(multicutModule, clsName.c_str())
            .def("optimize",
            [](
                PerturbAndMapType * self,
                nifty::marray::PyView<uint64_t> nodeLabelsArray
            )
            {
                const auto & graph = self->graph();
                const auto nNodes = graph.numberOfNodes();
                const auto nEdges = graph.numberOfEdges();



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


                nifty::marray::PyView<double> rarray(&nEdges,&nEdges+1);
                for(auto edge: graph.edges())
                    rarray(edge) = edgeState[edge];
                return rarray;
    

            },
            py::arg_t< py::array_t<uint64_t> >("nodeLabels", py::list() )
            )
        ;


        py::enum_<PerturbAndMapType::NoiseType>(pAndMapCls, "NoiseType")
            .value("UNIFORM_NOISE", PerturbAndMapType::UNIFORM_NOISE)
            .value("NORMAL_NOISE", PerturbAndMapType::NORMAL_NOISE)
            .value("MAKE_LESS_CERTAIN", PerturbAndMapType::MAKE_LESS_CERTAIN)
            .export_values();


        auto settings = py::class_<PerturbAndMapSettingsType>(multicutModule, settingsClsName.c_str())
            .def(py::init<>())
            .def_readwrite("mcFactory",&PerturbAndMapSettingsType::mcFactory)
            .def_readwrite("numberOfIterations",&PerturbAndMapSettingsType::numberOfIterations)
            .def_readwrite("numberOfThreads",&PerturbAndMapSettingsType::numberOfThreads)
            .def_readwrite("verbose",&PerturbAndMapSettingsType::verbose)
            .def_readwrite("noiseType",&PerturbAndMapSettingsType::noiseType)
            .def_readwrite("noiseMagnitude",&PerturbAndMapSettingsType::noiseMagnitude)
        ;





        multicutModule.def("perturbAndMap",
            [](const McObjective & objective, const PerturbAndMapSettingsType & s){
                auto perturbAndMap = new PerturbAndMapType(objective, s);
                return perturbAndMap;
            },
            py::return_value_policy::take_ownership,
            py::keep_alive<0, 1>(),
            py::arg("objective"),py::arg("settings")
        );

        
    }

}
}
