#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

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
                                       FeatureType dataMin, FeatureType dataMax) {
            py::gil_scoped_release allowThreads;
            extractBlockFeaturesFromBoundaryMaps<T>(blockPrefix, dataPath, dataKey,
                                                    labelPath, labelKey, blockIds, tmpFeatureStorage,
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
                                       FeatureType dataMin, FeatureType dataMax) {
            py::gil_scoped_release allowThreads;
            extractBlockFeaturesFromAffinityMaps<T>(blockPrefix, dataPath, dataKey,
                                                    labelPath, labelKey, blockIds, tmpFeatureStorage,
                                                    offsets, dataMin, dataMax);

        }, py::arg("blockPrefix"),
           py::arg("dataPath"), py::arg("dataKey"),
           py::arg("labelPath"), py::arg("labelKey"),
           py::arg("blockIds"), py::arg("tmpFeatureStorage"), py::arg("offsets"),
           py::arg("dataMin")=0., py::arg("dataMax")=1.);


    }


    void exportFeatureMerging(py::module & module) {
        module.def("mergeFeatureBlocks", [](const std::string & graphBlockPrefix,
                                            const std::string & featureBlockPrefix,
                                            const std::string & featuresOut,
                                            const size_t numberOfBlocks,
                                            const size_t edgeIdBegin,
                                            const size_t edgeIdEnd,
                                            const int numberOfThreads) {
            py::gil_scoped_release allowThreads;
            mergeFeatureBlocks(graphBlockPrefix,
                               featureBlockPrefix,
                               featuresOut,
                               numberOfBlocks,
                               edgeIdBegin,
                               edgeIdEnd,
                               numberOfThreads);

        }, py::arg("graphBlockPrefix"), py::arg("featureBlockPrefix"),
           py::arg("featuresOut"), py::arg("numberOfBlocks"), py::arg("edgeIdBegin"),
           py::arg("edgeIdEnd"), py::arg("numberOfThreads")=1);
    }


    void exportMergeableFeatures(py::module & module) {
        exportMergeableFeaturesT<uint8_t>(module, "_uint8");
        exportMergeableFeaturesT<float>(module, "_float32");
        exportFeatureMerging(module);
    }

}
}
