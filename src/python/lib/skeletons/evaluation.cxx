#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "nifty/skeletons/evaluation.hxx"


namespace py = pybind11;

namespace nifty {
namespace skeletons {

    void exportEvaluation(py::module & module) {

        typedef SkeletonMetrics SelfType;

        // TODO lift gil for init with call wrapper
        py::class_<SelfType>(module, "SkeletonMetrics")
            .def(py::init<const std::string &, const std::string &, const std::vector<size_t> &, const int>())
            //
            .def("getNodeAssignments", [](const SelfType & self){return self.getNodeAssignments();})
            .def("computeSplitScores", [](const SelfType & self, const int numberOfThreads){
                std::map<size_t, double> out;
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
                std::map<size_t, double> skeletonRunlens;
                std::map<size_t, std::map<size_t, double>> fragmentRunlens;
                self.computeSplitRunlengths(resolution, skeletonRunlens, fragmentRunlens, numberOfThreads);
                // can't lift gil because we mess with pytthon exposed objects internally
                //{
                //    py::gil_scoped_release allowThreads;
                //    self.computeSplitRunlength(resolution, skeletonRunlens, fragmentRunlens, numberOfThreads);
                //}
                return std::make_pair(skeletonRunlens, fragmentRunlens);
            }, py::arg("resolution"), py::arg("numberOfThreads")=-1)
            //
            .def("computeExplicitMerges", [](const SelfType & self, const int numberOfThreads) {
                std::map<size_t, std::vector<size_t>> out;
                self.computeExplicitMerges(out, numberOfThreads);
                return out;
            }, py::arg("numberOfThreads")=-1)
            //
            .def("computeHeuristicMerges", [](const SelfType & self,
                                              const std::array<double, 3> & resolution,
                                              const double maxDistance,
                                              const int numberOfThreads) {
                std::map<size_t, std::vector<size_t>> out;
                self.computeHeuristicMerges(resolution, maxDistance, out, numberOfThreads);
                return out;
            }, py::arg("resolution"), py::arg("maxDistance"), py::arg("numberOfThreads")=-1)
        ;
    }

}
}
