#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "nifty/skeletons/evaluation.hxx"


namespace py = pybind11;

namespace nifty {
namespace skeletons {

    void exportEvaluation(py::module & module) {

        // TODO actually we want to expose other functionality
        module.def("getSkeletonNodeAssignments", [](const std::string & segmentationPath,
                                                    const std::string & skeletonTopFolder,
                                                    const std::vector<size_t> & skeletonIds,
                                                    const int numberOfThreads){
            py::gil_scoped_release allowThreads;
            SkeletonDictionary out;
            {
                getSkeletonNodeAssignments(segmentationPath, skeletonTopFolder,
                                           skeletonIds, numberOfThreads, out);
            }

        }, py::arg("segmentationPath"), py::arg("skeletonTopFolder"),
           py::arg("skeletonIds"), py::arg("numberOfThreads")=-1
        );

    }

}
}
