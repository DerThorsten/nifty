#include <pybind11/pybind11.h>
#include <iostream>
#include <sstream>
#include <pybind11/numpy.h>
#include <xtensor-python/pytensor.hpp>


#include "nifty/graph/opt/multicut/perturb_and_map.hxx"
#include "nifty/python/graph/undirected_list_graph.hxx"
#include "nifty/python/graph/edge_contraction_graph.hxx"
#include "nifty/python/graph/opt/multicut/multicut_objective.hxx"

namespace py = pybind11;


PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);



namespace nifty{
namespace graph{
namespace opt{
namespace multicut{

    template<class OBJECTIVE>
    void exportPerturbAndMapT(py::module & multicutModule) {
        typedef OBJECTIVE ObjectiveType;
        typedef typename ObjectiveType::GraphType GraphType;
        typedef typename GraphType:: template EdgeMap<double>   EdgeState;
        typedef typename GraphType:: template NodeMap<uint64_t> NodeLabels;
        typedef PerturbAndMap<ObjectiveType> PerturbAndMapType;
        typedef typename PerturbAndMapType::SettingsType PerturbAndMapSettingsType;


        const auto objName = MulticutObjectiveName<ObjectiveType>::name();
        const auto clsName = std::string("PerturbAndMap") + objName;
        const auto settingsClsName = std::string("PerturbAndMapSettings") + objName;


        auto pAndMapCls = py::class_<PerturbAndMapType>(multicutModule, clsName.c_str())
            .def("optimize",
            [](
                PerturbAndMapType * self,
                xt::pytensor<uint64_t, 1> nodeLabelsArray
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

                typedef xt::pytensor<double, 1>::shape_type ShapeType;
                ShapeType shape = {static_cast<int64_t>(nEdges)};
                xt::pytensor<double, 1> rarray(shape);
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
} // namespace nifty::graph::opt::multicut
} // namespace nifty::graph::opt
}
}
