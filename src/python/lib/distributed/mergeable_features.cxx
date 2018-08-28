#ifdef WITH_Z5
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <xtensor-python/pytensor.hpp>

#include "nifty/distributed/mergeable_features.hxx"

namespace py = pybind11;

namespace nifty {
namespace distributed {

    template<class T>
    void exportMergeableFeaturesT(py::module & module, const std::string & typeName) {

        const std::string fuName1 = "extractBlockFeaturesFromBoundaryMaps" + typeName;
        module.def(fuName1.c_str(), [](const std::string & blockPrefix,
                                       const std::string & dataPath,
                                       const std::string & dataKey,
                                       const std::string & labelPath,
                                       const std::string & labelKey,
                                       const std::vector<size_t> & blockIds,
                                       const std::string & tmpFeatureStorage,
                                       const FeatureType dataMin,
                                       const FeatureType dataMax) {
            py::gil_scoped_release allowThreads;
            extractBlockFeaturesFromBoundaryMaps<T>(blockPrefix, dataPath, dataKey,
                                                    labelPath, labelKey, blockIds,
                                                    tmpFeatureStorage,
                                                    dataMin, dataMax);

        }, py::arg("blockPrefix"),
           py::arg("dataPath"), py::arg("dataKey"),
           py::arg("labelPath"), py::arg("labelKey"),
           py::arg("blockIds"), py::arg("tmpFeatureStorage"),
           py::arg("dataMin")=0., py::arg("dataMax")=1.);


        const std::string fuName2 = "extractBlockFeaturesFromAffinityMaps" + typeName;
        module.def(fuName2.c_str(), [](const std::string & blockPrefix,
                                       const std::string & dataPath,
                                       const std::string & dataKey,
                                       const std::string & labelPath,
                                       const std::string & labelKey,
                                       const std::vector<size_t> & blockIds,
                                       const std::string & tmpFeatureStorage,
                                       const std::vector<OffsetType> & offsets,
                                       const FeatureType dataMin,
                                       const FeatureType dataMax) {
            py::gil_scoped_release allowthreads;
            extractBlockFeaturesFromAffinityMaps<T>(blockPrefix, dataPath, dataKey,
                                                    labelPath, labelKey, blockIds,
                                                    tmpFeatureStorage, offsets,
                                                    dataMin, dataMax);

        }, py::arg("blockPrefix"),
           py::arg("dataPath"), py::arg("dataKey"),
           py::arg("labelPath"), py::arg("labelKey"),
           py::arg("blockIds"), py::arg("tmpFeatureStorage"), py::arg("offsets"),
           py::arg("dataMin")=0., py::arg("dataMax")=1.);
    }


    void exportAccumulateInput(py::module & module) {
        module.def("accumulateInput", [](const Graph & graph,
                                         const xt::pytensor<float, 3> & input,
                                         const xt::pytensor<uint64_t, 3> & labels,
                                         const bool ignoreLabel,
                                         const bool withSize,
                                         const FeatureType dataMin,
                                         const FeatureType dataMax) {

            const unsigned int nEdges = graph.numberOfEdges();
            const unsigned int nFeatures = withSize ? 10 : 9;
            xt::pytensor<float, 2> features({nEdges, nFeatures});
            {
                py::gil_scoped_release allowthreads;
                accumulateInput(graph, input, labels,
                                ignoreLabel, withSize,
                                dataMin, dataMax,
                                features);
            }
            return features;

        }, py::arg("graph"), py::arg("input"), py::arg("labels"),
           py::arg("ignoreLabel"), py::arg("withSize"),
           py::arg("dataMin"), py::arg("dataMax"));

    }


    void exportFeatureMerging(py::module & module) {
        module.def("mergeFeatureBlocks", [](const std::string & graphBlockPrefix,
                                            const std::string & featureBlockPrefix,
                                            const std::string & featuresOut,
                                            const std::vector<size_t> & blockIds,
                                            const size_t edgeIdBegin,
                                            const size_t edgeIdEnd,
                                            const int numberOfThreads) {
            py::gil_scoped_release allowThreads;
            mergeFeatureBlocks(graphBlockPrefix,
                               featureBlockPrefix,
                               featuresOut,
                               blockIds,
                               edgeIdBegin,
                               edgeIdEnd,
                               numberOfThreads);

        }, py::arg("graphBlockPrefix"), py::arg("featureBlockPrefix"),
           py::arg("featuresOut"), py::arg("blockIds"), py::arg("edgeIdBegin"),
           py::arg("edgeIdEnd"), py::arg("numberOfThreads")=1);
    }


    void exportMergeableFeatures(py::module & module) {
        exportMergeableFeaturesT<uint8_t>(module, "_uint8");
        exportMergeableFeaturesT<float>(module, "_float32");
        exportFeatureMerging(module);
        exportAccumulateInput(module);
    }

}
}
#endif
