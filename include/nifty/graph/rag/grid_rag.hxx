#pragma once

#include <random>
#include <functional>
#include <ctime>
#include <stack>
#include <algorithm>

#include <unordered_set>

#include "nifty/array/arithmetic_array.hxx"
#include "nifty/graph/undirected_list_graph.hxx"
#include "nifty/parallel/threadpool.hxx"
#include "nifty/tools/timer.hxx"
#include "nifty/tools/array_tools.hxx"

//#include "nifty/graph/rag/grid_rag_labels_proxy.hxx"
#include "nifty/graph/rag/detail_rag/compute_grid_rag.hxx"

#include "xtensor/xtensor.hpp"
#include "nifty/xtensor/xtensor.hxx"

#ifdef WITH_HDF5
#include "nifty/hdf5/hdf5_array.hxx"
#endif

#ifdef WITH_Z5
#include "nifty/z5/z5.hxx"
#endif


namespace nifty{
namespace graph{


template<class COORD>
COORD makeCoord2(const COORD & coord,const size_t axis){
    COORD coord2 = coord;
    coord2[axis] += 1;
    return coord2;
};


template<size_t DIM, class LABELS>
class GridRag : public UndirectedGraph<>{
public:
    struct DontComputeRag{};
    typedef LABELS LabelsType;
    typedef typename LabelsType::value_type value_type;
    typedef array::StaticArray<int64_t, DIM> ShapeType;
    typedef GridRag<DIM, LABELS> SelfType;
    typedef tools::BlockStorage<value_type> BlockStorageType;

    friend class detail_rag::ComputeRag< SelfType >;

    struct SettingsType{
        SettingsType()
        :   numberOfThreads(-1),
            blockShape(),
            haveIgnoreLabel(false),
            ignoreLabel(0)
        {
            for(auto d=0; d<DIM; ++d)
                blockShape[d] = 100;
        }
        int numberOfThreads;
        ShapeType blockShape;
        bool haveIgnoreLabel;
        uint64_t ignoreLabel;
    };


    GridRag(const LabelsType & labels,
            const std::size_t numberOfLabels,
            const SettingsType & settings = SettingsType())
    :   settings_(settings),
        numberOfLabels_(numberOfLabels),
        labels_(std::make_unique<LabelsType>(labels)) {

        // get the shape
        const auto & tmpShape = labels.shape();
        for(int d = 0; d < DIM; ++d) {
            shape_[d] = tmpShape[d];
        }

        // compute the rag
        detail_rag::ComputeRag<SelfType>::computeRag(*this, settings_);
    }

    template<class ITER>
    GridRag(const LabelsType & labels,
            const std::size_t numberOfLabels,
            ITER serializationBegin,
            const SettingsType & settings = SettingsType())
    :   settings_(settings),
        numberOfLabels_(numberOfLabels),
        labels_(std::make_unique<LabelsType>(std::move(labels))) {

        // get the shape
        const auto & tmpShape = labels.shape();
        for(int d = 0; d < DIM; ++d) {
            shape_[d] = tmpShape[d];
        }

        // deserialize
        this->deserialize(serializationBegin);
    }

    const LabelsType & labels() const {
        return *labels_;
    }

    const ShapeType & shape() const {
        return shape_;
    }

    const std::size_t numberOfLabels() const {
        return numberOfLabels_;
    }

    UndirectedGraph<> extractSubgraphFromRoi(const ShapeType & begin,
                                             const ShapeType & end,
                                             std::vector<int64_t> & innerEdgesOut) const {
        typedef typename xt::xtensor<value_type, DIM>::shape_type ArrayShapeType;
        innerEdgesOut.clear();

        ShapeType subShape;
        ArrayShapeType subArrayShape;
        for(int d = 0; d < DIM; ++d) {
            subShape[d] = end[d] - begin[d];
            subArrayShape[d] = subShape[d];
        }
        xt::xtensor<value_type, DIM> subLabels(subArrayShape);

        tools::readSubarray(*labels_, begin, end, subLabels);

        std::vector<value_type> uniqueNodes;
        tools::uniques(subLabels, uniqueNodes);

        std::map<value_type, value_type> globalToLocalNodes;
        for(int ii = 0; ii < uniqueNodes.size(); ++ii) {
            globalToLocalNodes[uniqueNodes[ii]] = ii;
        }
        UndirectedGraph<> subGraph(uniqueNodes.size());

        auto makeCoord2 = [](const ShapeType & coord, const int d) {
            ShapeType coord2 = coord;
            coord2[d] += 1;
            return coord2;
        };

        // extract the inner uv ids
        // TODO parallelize
        std::set<std::pair<int64_t, int64_t>> innerEdges;
        tools::forEachCoordinate(subShape, [&](const ShapeType & coord){
            const auto lU = xtensor::read(subLabels, coord.asStdArray());
            for(int d = 0; d < DIM; ++d) {
                if(coord[d] < subShape[d] - 1) {
                    auto coordV = makeCoord2(coord, d);
                    const auto lV = xtensor::read(subLabels, coordV.asStdArray());
                    if(lU != lV) {
                        innerEdges.emplace( std::make_pair(std::min(lU, lV), std::max(lU, lV)) );
                    }
                }
            }
        });

        // convert the inner uv ids to global edge ids and construct new graph in local coordinates
        innerEdgesOut.reserve(innerEdges.size());
        for(auto it = innerEdges.begin(); it != innerEdges.end(); ++it) {
            const auto lU = it->first;
            const auto lV = it->second;
            innerEdgesOut.push_back(this->findEdge(lU, lV));

            subGraph.insertEdge(globalToLocalNodes[lU], globalToLocalNodes[lV]);
        }
        return subGraph;
    }

protected:
    GridRag(const LabelsType & labels,
            const std::size_t numberOfLabels,
            const SettingsType & settings,
            const DontComputeRag)
    :   settings_(settings),
        numberOfLabels_(numberOfLabels),
        labels_(std::make_unique<LabelsType>(labels)) {

        // get the shape
        const auto & tmpShape = labels.shape();
        for(int d = 0; d < DIM; ++d) {
            shape_[d] = tmpShape[d];
        }
    }

protected:
    typedef std::unique_ptr<LabelsType> StorageType;
    SettingsType settings_;
    std::size_t numberOfLabels_;
    StorageType labels_;
    ShapeType shape_;
};

} // end namespace graph
} // end namespace nifty
