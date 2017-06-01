#include <pybind11/pybind11.h>
#include <iostream>
#include <sstream>
#include <pybind11/numpy.h>


#include "nifty/graph/optimization/multicut/perturb_and_map.hxx"
#include "nifty/python/converter.hxx"

#include "nifty/python/graph/undirected_list_graph.hxx"
#include "nifty/python/graph/edge_contraction_graph.hxx"
#include "nifty/python/graph/optimization/multicut/multicut_objective.hxx"

namespace py = pybind11;


PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);



namespace nifty{
namespace graph{
namespace optimization{
namespace multicut{

    template<class OBJECTIVE>
    void exportPerturbAndMapT(py::module & multicutModule) {
        typedef OBJECTIVE ObjectiveType;
        typedef typename ObjectiveType::Graph Graph;
        typedef typename Graph:: template EdgeMap<double>   EdgeState;
        typedef typename Graph:: template NodeMap<uint64_t> NodeLabels;
        typedef PerturbAndMap<ObjectiveType> PerturbAndMapType;
        typedef typename PerturbAndMapType::Settings PerturbAndMapSettingsType;


        const auto objName = MulticutObjectiveName<ObjectiveType>::name();
        const auto clsName = std::string("PerturbAndMap") + objName;
        const auto settingsClsName = std::string("PerturbAndMapSettings") + objName;


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


        py::enum_<typename PerturbAndMapType::NoiseType>(pAndMapCls, "NoiseType")
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
            [](const ObjectiveType & objective, const PerturbAndMapSettingsType & s){
                auto perturbAndMap = new PerturbAndMapType(objective, s);
                return perturbAndMap;
            },
            py::return_value_policy::take_ownership,
            py::keep_alive<0, 1>(),
            py::arg("objective"),py::arg("settings")
        );
    }

    void exportPerturbAndMap(py::module & multicutModule) {
        {
            typedef PyUndirectedGraph GraphType;
            typedef MulticutObjective<GraphType, double> ObjectiveType;
            exportPerturbAndMapT<ObjectiveType>(multicutModule);
        }
        {
            typedef PyContractionGraph<PyUndirectedGraph> GraphType;
            typedef MulticutObjective<GraphType, double> ObjectiveType;
            exportPerturbAndMapT<ObjectiveType>(multicutModule);
        }
    }
} // namespace nifty::graph::optimization::multicut
} // namespace nifty::graph::optimization
}
}
