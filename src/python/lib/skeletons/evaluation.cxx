#ifdef WITH_Z5
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "nifty/skeletons/evaluation.hxx"
#include "xtensor-python/pytensor.hpp"


namespace py = pybind11;

namespace nifty {
namespace skeletons {

    void exportEvaluation(py::module & module) {

        typedef SkeletonMetrics SelfType;

        // TODO lift gil for init with call wrapper
        py::class_<SelfType>(module, "SkeletonMetrics")
            .def(py::init<const std::string &, const std::string &,
                          const std::vector<std::size_t> &, const int>())
            // can't build with boost serialization
            //.def(py::init<const std::string &, const std::string &,
            //              const std::vector<std::size_t> &, const std::string &>())
            //
            .def("getNodeAssignments", [](const SelfType & self){return self.getNodeAssignments();})
            .def("computeSplitScores", [](const SelfType & self, const int numberOfThreads){
                std::map<std::size_t, double> out;
                self.computeSplitScores(out, numberOfThreads);
                // can't lift gil because we mess with python exposed objects internally
                //{
                //    py::gil_scoped_release allowThreads;
                //    self.computeSplitScore(out, numberOfThreads);
                //}
                return out;
            }, py::arg("numberOfThreads")=-1)
            //
            .def("computeSplitRunlengths", [](const SelfType & self, const std::array<double, 3> & resolution, const int numberOfThreads){
                std::map<std::size_t, double> skeletonRunlens;
                std::map<std::size_t, std::map<std::size_t, double>> fragmentRunlens;
                self.computeSplitRunlengths(resolution, skeletonRunlens,
                                            fragmentRunlens, numberOfThreads);
                // can't lift gil because we mess with python exposed objects internally
                //{
                //    py::gil_scoped_release allowThreads;
                //    self.computeSplitRunlength(resolution, skeletonRunlens, fragmentRunlens, numberOfThreads);
                //}
                return std::make_pair(skeletonRunlens, fragmentRunlens);
            }, py::arg("resolution"), py::arg("numberOfThreads")=-1)
            //
            .def("computeExplicitMerges", [](const SelfType & self, const int numberOfThreads) {
                std::map<std::size_t, std::vector<std::size_t>> out;
                self.computeExplicitMerges(out, numberOfThreads);
                return out;
            }, py::arg("numberOfThreads")=-1)
            //
            .def("computeExplicitMergeScores", [](const SelfType & self,
                                                  const int numberOfThreads) {
                std::map<std::size_t, double> mergeScore;
                std::map<std::size_t, std::size_t> mergePoints;
                self.computeExplicitMergeScores(mergeScore, mergePoints, numberOfThreads);
                return std::make_pair(mergeScore, mergePoints);
            }, py::arg("numberOfThreads")=-1)
            //
            .def("computeGoogleScore", [](const SelfType & self, const int numberOfThreads) {
                double correctScore, splitScore, mergeScore;
                std::size_t mergePoints;
                self.computeGoogleScore(correctScore, splitScore,
                                        mergeScore, mergePoints, numberOfThreads);
                return std::make_tuple(correctScore, splitScore, mergeScore, mergePoints);
            }, py::arg("numberOfThreads")=-1)
            //
            .def("computeHeuristicMerges", [](const SelfType & self,
                                              const std::array<double, 3> & resolution,
                                              const double maxDistance,
                                              const int numberOfThreads) {
                std::map<std::size_t, std::vector<std::size_t>> out;
                self.computeHeuristicMerges(resolution, maxDistance, out, numberOfThreads);
                return out;
            }, py::arg("resolution"), py::arg("maxDistance"), py::arg("numberOfThreads")=-1)
            //
            .def("computeDistanceStatistics", [](const SelfType & self,
                                                 const std::array<double, 3> & resolution,
                                                 const int numberOfThreads) {
                SelfType::SkeletonDistanceStatistics out;
                self.computeDistanceStatistics(resolution, out, numberOfThreads);
                return out;
            }, py::arg("resolution"), py::arg("numberOfThreads")=-1)
            //
            // can't build with boost serialization
            //.def("serialize", [](const SelfType & self,
            //                     const std::string & serializationPath){
            //    self.serialize(serializationPath);
            //}, py::arg("serializationPath"))
            //
            .def("mergeFalseSplitNodes", [](const SelfType & self, const int numberOfThreads){
                std::map<std::size_t, std::set<std::pair<std::size_t, std::size_t>>> out;
                self.mergeFalseSplitNodes(out, numberOfThreads);
                return out;
            }, py::arg("numberOfThreads")=-1)
            // hacky export of group skeletons to blocks ...
            .def("groupSkeletonBlocks", [](SelfType & self, const int numberOfThreads){
                parallel::ThreadPool tp(numberOfThreads);

                // group the skeleton parts by the chunks of the segmentation
                // dataset they fall into
                SelfType::SkeletonBlockStorage skeletonsToBlocks;
                std::vector<std::size_t> nonEmptyChunks;
                self.groupSkeletonBlocks(skeletonsToBlocks, nonEmptyChunks, tp);

                // need to copy to pytensor to return this to python
                typedef xt::pytensor<std::size_t, 2> OutArray;
                typedef typename OutArray::shape_type ArrayShape;
                typedef std::map<std::size_t, OutArray> OutStorage;
                typedef std::map<std::size_t, OutStorage> OutBlockStorage;

                // TODO should be parallelized
                OutBlockStorage out;
                for(const auto & blockItem : skeletonsToBlocks) {
                    out[blockItem.first] = OutStorage();
                    auto & currentOut = out[blockItem.first];
                    for(const auto & skelItem : blockItem.second) {
                        ArrayShape shape = {int64_t(skelItem.second.shape()[0]),
                                            int64_t(skelItem.second.shape()[1])};
                        currentOut[skelItem.first] = OutArray(shape);
                        std::copy(skelItem.second.begin(),
                                  skelItem.second.end(),
                                  currentOut[skelItem.first].begin());
                    }
                }
                return std::make_tuple(nonEmptyChunks, out);
            }, py::arg("numberOfThreads")=-1)
            //
            .def("getNodesInFalseMergeLabels", [](const SelfType & self,
                                                  const int numberOfThreads) {
                std::map<std::size_t, std::vector<std::size_t>> out;
                self.getNodesInFalseMergeLabels(out, numberOfThreads);
                return out;
            }, py::arg("numberOfThreads")=-1)

            ;
    }

}
}
#endif
