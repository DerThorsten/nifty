#include <pybind11/pybind11.h>
#include <iostream>
#include <sstream>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "nifty/python/converter.hxx"

#include "nifty/ground_truth/overlap.hxx"

namespace py = pybind11;


namespace nifty{
namespace ground_truth{




    void exportOverlap(py::module & groundTruthModule){

        typedef Overlap<> OverlapType;

        py::class_<OverlapType>(groundTruthModule, "Overlap")

            .def("__init__",
                [](
                    OverlapType &instance,
                    const uint64_t maxLabelA,
                    nifty::marray::PyView<uint64_t> labelA,
                    nifty::marray::PyView<uint64_t> labelB
                ) {
                    new (&instance) OverlapType(maxLabelA, labelA, labelB);
                }
            )
            .def("differentOverlaps",[](
                const OverlapType & self,
                const uint64_t u, const uint64_t v
            ){
                return self.differentOverlap(u, v);
            })
            .def("differentOverlaps",[](
                const OverlapType & self,
                nifty::marray::PyView<uint64_t> uv
            ){
                nifty::marray::PyView<float> out({uv.shape(0)});

                for(auto i=0; i<uv.shape(0); ++i){
                    out(i) = self.differentOverlap(uv(i,0),uv(i,1));
                }

                return out;
            })
            .def("differentMaxOverlaps",[](
                const OverlapType & self,
                nifty::marray::PyView<uint64_t> uv
            ){
                nifty::marray::PyView<uint8_t> out({uv.shape(0)});

                for(auto i=0; i<uv.shape(0); ++i){
                    out(i) = self.differentMaxOverlap(uv(i,0),uv(i,1));
                }

                return out;
            })
            .def("bleeding",[](
                const OverlapType & self,
                nifty::marray::PyView<uint64_t> ids
            ){
                nifty::marray::PyView<float> out({ids.shape(0)});

                for(auto i=0; i<ids.shape(0); ++i){
                    out(i) = self.bleeding(ids(i));
                }
                return out;
            })
            .def("counts",[](const OverlapType & self){

                const auto & counts = self.counts();
                nifty::marray::PyView<uint64_t> out({counts.size()});

                for(auto i=0; i<counts.size(); ++i){
                    out(i) = counts[i];
                }

                return out;
            })
            .def("overlapArrays", [](const OverlapType & self, const size_t index, const bool sorted){

                const auto & counts = self.counts();
                const auto & overlaps = self.overlaps();

                const auto & olMap = overlaps[index];

                typedef nifty::marray::PyView<uint64_t>  ArrayType;
                
                ArrayType olIndices({olMap.size()});
                ArrayType olCounts({olMap.size()});

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
                return std::make_pair(olIndices, olCounts);
            }, py::arg("index"),py::arg("sorted") = false
            )

            .def("overlapArraysNormalized", [](const OverlapType & self, const size_t index, const bool sorted){

                const float count = self.counts()[index];
                const auto & overlaps = self.overlaps();

                const auto & olMap = overlaps[index];
       
                nifty::marray::PyView<uint64_t> olIndices({olMap.size()});
                nifty::marray::PyView<float> olCounts({olMap.size()});

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
                return std::make_pair(olIndices, olCounts);
            },py::arg("index"),py::arg("sorted") = false)

            .def(
                "transferCutProbabilities",
                [](
                    const OverlapType & self,
                    const nifty::marray::PyView<uint64_t, 2> & uv_reference,
                    //const nifty::marray::PyView<float, 1>    & w_reference,
                    const nifty::marray::PyView<uint64_t, 2> & uv_groundtruth,
                    const nifty::marray::PyView<float, 1>    & p_groundtruth
                    //const nifty::marray::PyView<float, 1>    & w_groundtruth
                )
                {
                    nifty::marray::PyView<float> out({uv_reference.shape(0)});
                    nifty::marray::PyView<float> w_out({uv_reference.shape(0)});
                    self.transferCutProbabilities(
                        uv_reference,
                        //w_reference,
                        uv_groundtruth,
                        p_groundtruth,
                        // w_groundtruth,
                        out,
                        w_out
                    );

                    return std::make_pair(out, w_out);
                }
            )
        ;
        
    }
}
}