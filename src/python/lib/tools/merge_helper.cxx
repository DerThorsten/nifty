#include <pybind11/pybind11.h>

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "xtensor-python/pytensor.hpp"
#include "nifty/tools/merge_helper.hxx"

namespace py = pybind11;

namespace nifty{
namespace tools{

    template<class NODE_TYPE>
    void exportMergeHelperT(py::module & m) {

        typedef NODE_TYPE NodeType;
        m.def("computeMergeVotes", [](const xt::pytensor<NodeType, 2> & uvIds,
                                      const xt::pytensor<uint8_t, 1> & indicators,
                                      const xt::pytensor<std::size_t, 1> & sizes,
                                      const bool weightEdges) {

            std::map<std::pair<NodeType, NodeType>,
                     std::pair<std::size_t, std::size_t>> mergeVotes;

            {
                py::gil_scoped_release allowThreads;
                computeMergeVotes(uvIds, indicators, sizes, mergeVotes, weightEdges);
            }

            const std::size_t nPairs = mergeVotes.size();

            typedef typename xt::pytensor<NodeType, 2>::shape_type ShapeType;
            ShapeType shape = {static_cast<int64_t>(nPairs), 2L};
            xt::pytensor<NodeType, 2> uvIdsOut = xt::zeros<NodeType>(shape);
            xt::pytensor<std::size_t, 2> votesOut = xt::zeros<NodeType>(shape);

            std::size_t i = 0;
            for(const auto & vote : mergeVotes) {
                uvIdsOut(i, 0) = vote.first.first;
                uvIdsOut(i, 1) = vote.first.second;
                votesOut(i, 0) = vote.second.first;
                votesOut(i, 1) = vote.second.second;
                ++i;
            }

            return std::make_pair(uvIdsOut, votesOut);

        }, py::arg("uvIds"),
           py::arg("indicators"),
           py::arg("sizes"),
           py::arg("weightEdges")=false);


        m.def("mergeMergeVotes", [](const xt::pytensor<NodeType, 2> & uvIds,
                                    const xt::pytensor<std::size_t, 2> & votes) {

            std::map<std::pair<NodeType, NodeType>,
                     std::pair<std::size_t, std::size_t>> mergeVotes;
            {
                py::gil_scoped_release allowThreads;
                mergeMergeVotes(uvIds, votes, mergeVotes);
            }

            const std::size_t nPairs = mergeVotes.size();

            typedef typename xt::pytensor<NodeType, 2>::shape_type ShapeType;
            ShapeType shape = {static_cast<int64_t>(nPairs), 2L};
            xt::pytensor<NodeType, 2> uvIdsOut = xt::zeros<NodeType>(shape);
            xt::pytensor<std::size_t, 2> votesOut = xt::zeros<NodeType>(shape);

            std::size_t i = 0;
            for(const auto & vote : mergeVotes) {
                // write out the uv ids
                uvIdsOut(i, 0) = vote.first.first;
                uvIdsOut(i, 1) = vote.first.second;
                // write out the votes
                votesOut(i, 0) = vote.second.first;
                votesOut(i, 1) = vote.second.second;
                ++i;
            }

            return std::make_pair(uvIdsOut, votesOut);

        }, py::arg("uvIds"),
           py::arg("votes"));
    }


    void exportMergeHelper(py::module & m) {
        exportMergeHelperT<uint64_t>(m);
    }


}
}
