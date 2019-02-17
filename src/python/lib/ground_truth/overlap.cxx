#include <iostream>
#include <sstream>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"

#include "nifty/ground_truth/overlap.hxx"

namespace py = pybind11;


namespace nifty{
namespace ground_truth{




    void exportOverlap(py::module & groundTruthModule){

        typedef Overlap<> OverlapType;

        py::class_<OverlapType>(groundTruthModule, "Overlap")

            .def(py::init([](const uint64_t maxLabelA,
                             xt::pyarray<uint64_t> labelA,
                             xt::pyarray<uint64_t> labelB) {
                    return new OverlapType(maxLabelA, labelA, labelB);
                })
            )
            .def("differentOverlaps",[](
                const OverlapType & self,
                const uint64_t u, const uint64_t v
            ){
                return self.differentOverlap(u, v);
            })
            .def("differentOverlaps",[](
                const OverlapType & self,
                xt::pytensor<uint64_t, 2> uv
            ){
                xt::pytensor<float, 1> out = xt::zeros<float>({uv.shape()[0]});

                {
                    py::gil_scoped_release allowThreads;
                    for(auto i=0; i<uv.shape()[0]; ++i){
                        out(i) = self.differentOverlap(uv(i,0), uv(i,1));
                    }
                }

                return out;
            })

            .def("bleeding",[](
                const OverlapType & self,
                xt::pytensor<uint64_t, 1> ids
            ){
                xt::pytensor<float, 1> out = xt::zeros<float>({ids.shape()[0]});

                {
                    py::gil_scoped_release allowThreads;
                    for(auto i=0; i<ids.shape()[0]; ++i){
                        out(i) = self.bleeding(ids(i));
                    }
                }
                return out;
            })
            .def("counts",[](const OverlapType & self){

                const auto & counts = self.counts();
                xt::pytensor<uint64_t, 1> out = xt::zeros<uint64_t>({counts.size()});

                {
                    py::gil_scoped_release allowThreads;
                    for(auto i=0; i<counts.size(); ++i){
                        out(i) = counts[i];
                    }
                }

                return out;
            })
            .def("overlapArrays", [](const OverlapType & self, const std::size_t index, const bool sorted){

                typedef xt::pytensor<uint64_t, 1> ArrayType;
                const auto & overlaps = self.overlaps();
                const auto & olMap = overlaps[index];

                ArrayType olIndices = xt::zeros<uint64_t>({olMap.size()});
                ArrayType olCounts  = xt::zeros<uint64_t>({olMap.size()});
                {
                    py::gil_scoped_release allowThreads;
                    const auto & counts = self.counts();

                    if(!sorted){
                        auto c=0;
                        for(const auto & kv : olMap){
                            olIndices(c) = kv.first;
                            olCounts(c) = kv.second;
                            ++c;
                        }
                    }
                    else{
                        typedef std::pair<uint64_t, uint64_t> PairType;
                        std::vector<PairType> pairVec(olMap.size());
                        auto c=0;
                        for(const auto & kv : olMap){
                            pairVec[c] = PairType(kv.first, kv.second);
                            ++c;
                        }
                        std::sort(pairVec.begin(), pairVec.end(),[](const PairType & pA, const PairType & pB){
                            return pA.second > pB.second;
                        });
                        for(c=0; c<pairVec.size(); ++c){
                            olIndices(c) = pairVec[c].first;
                            olCounts(c) = pairVec[c].second;
                        }

                    }
                }
                return std::make_pair(olIndices, olCounts);
            }, py::arg("index"),py::arg("sorted") = false
            )

            .def("overlapArraysNormalized", [](const OverlapType & self, const std::size_t index, const bool sorted){

                const auto & overlaps = self.overlaps();
                const auto & olMap = overlaps[index];
                xt::pytensor<uint64_t, 1> olIndices = xt::zeros<uint64_t>({olMap.size()});
                xt::pytensor<float, 1> olCounts = xt::zeros<float>({olMap.size()});

                {
                    py::gil_scoped_release allowThreads;
                    const float count = self.counts()[index];

                    if(!sorted){
                        auto c=0;
                        for(const auto & kv : olMap){
                            olIndices(c) = kv.first;
                            olCounts(c) = float(kv.second) / count;
                            ++c;
                        }
                    }
                    else{
                        typedef std::pair<uint64_t, uint64_t> PairType;
                        std::vector<PairType> pairVec(olMap.size());
                        auto c=0;
                        for(const auto & kv : olMap){
                            pairVec[c] = PairType(kv.first, kv.second);
                            ++c;
                        }
                        std::sort(pairVec.begin(), pairVec.end(),[](const PairType & pA, const PairType & pB){
                            return pA.second > pB.second;
                        });
                        for(c=0; c<pairVec.size(); ++c){
                            olIndices(c) = pairVec[c].first;
                            olCounts(c) = float(pairVec[c].second)/ count;
                        }
                    }
                }
                return std::make_pair(olIndices, olCounts);
            },py::arg("index"),py::arg("sorted") = false)
        ;

    }
}
}
