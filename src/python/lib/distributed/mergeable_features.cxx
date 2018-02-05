#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "nifty/distributed/mergeable_features.hxx"

namespace py = pybind11;

namespace nifty {
namespace distributed {

    void exportMergeableFeatures(py::module & module) {

        module.def("extractBlockFeaturesFromBoundaryMaps", [](const std::string & groupPath,
                                                              const std::string & blockPrefix,
                                                              const std::string & dataPath,
                                                              const std::string & dataKey,
                                                              const std::string & labelPath,
                                                              const std::string & labelKey,
                                                              const std::vector<size_t> & blockIds,
                                                              const std::string & tmpFeatureStorage,
                                                              float dataMin, float dataMax) {
            py::gil_scoped_release allowThreads;
            extractBlockFeaturesFromBoundaryMaps(groupPath, blockPrefix, dataPath, dataKey,
                                                 labelPath, labelKey, blockIds, tmpFeatureStorage,
                                                 dataMin, dataMax);

        }, py::arg("groupPath"), py::arg("blockPrefix"),
           py::arg("dataPath"), py::arg("dataKey"),
           py::arg("labelPath"), py::arg("labelKey"),
           py::arg("blockIds"), py::arg("tmpFeatureStorage"),
           py::arg("dataMin")=0., py::arg("dataMax")=1.);


        module.def("extractBlockFeaturesFromAffinityMaps", [](const std::string & groupPath,
                                                              const std::string & blockPrefix,
                                                              const std::string & dataPath,
                                                              const std::string & dataKey,
                                                              const std::string & labelPath,
                                                              const std::string & labelKey,
                                                              const std::vector<size_t> & blockIds,
                                                              const std::string & tmpFeatureStorage,
                                                              const std::vector<OffsetType> & offsets,
                                                              float dataMin, float dataMax) {
            py::gil_scoped_release allowThreads;
            extractBlockFeaturesFromAffinityMaps(groupPath, blockPrefix, dataPath, dataKey,
                                                 labelPath, labelKey, blockIds, tmpFeatureStorage,
                                                 offsets, dataMin, dataMax);

        }, py::arg("groupPath"), py::arg("blockPrefix"),
           py::arg("dataPath"), py::arg("dataKey"),
           py::arg("labelPath"), py::arg("labelKey"),
           py::arg("blockIds"), py::arg("offsets"), py::arg("tmpFeatureStorage"),
           py::arg("dataMin")=0., py::arg("dataMax")=1.);


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
           py::arg("featuresOut"), py::arg("numberOfBLocks"), py::arg("edgeIdBegin"),
           py::arg("edgeIdEnd"), py::arg("numberOfThreads")=1);
    }


}
}
