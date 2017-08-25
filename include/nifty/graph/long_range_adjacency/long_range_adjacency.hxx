#pragma once

#include "nifty/parallel/threadpool.hxx"
#include "nifty/tools/array_tools.hxx"
#include "nifty/graph/undirected_list_graph.hxx"

namespace nifty {
namespace graph {

template<class LABELS>
class LongRangeAdjacency : public UndirectedGraph<>{

public:
    typedef LABELS Labels;
    typedef typename Labels::DataType LabelType;
    typedef UndirectedGraph<> BaseType;

    typedef array::StaticArray<int64_t, 3> Coord;
    typedef array::StaticArray<int64_t, 2> Coord2;

    // constructor from data
    LongRangeAdjacency(
        const Labels & labels,
        const size_t range,
        const size_t numberOfLabels,
        const int numberOfThreads=-1
    ) : range_(range),
        shape_({labels.shape(0), labels.shape(1), labels.shape(2)}),
        numberOfEdgesInSlice_(shape_[0]),
        edgeOffset_(shape_[0])
    {
        initAdjacency(labels, numberOfLabels, numberOfThreads);
    }

    // constructor from serialization
    template<class ITER>
    LongRangeAdjacency(
        const Labels & labels,
        ITER & iter
    ) : range_(0),
        shape_({labels.shape(0), labels.shape(1), labels.shape(2)}),
        numberOfEdgesInSlice_(shape_[0]),
        edgeOffset_(shape_[0])
    {
        deserializeAdjacency(iter);
    }

    // API
    
    size_t range() const {
        return range_;
    }
    
    size_t numberOfEdgesInSlice(const size_t z) const {
        return numberOfEdgesInSlice_[z];
    }

    size_t edgeOffset(const size_t z) const {
        return edgeOffset_[z];
    }

    size_t serializationSize() const {
        size_t size = BaseType::serializationSize();
        size += 1;
        size += shape_[0] * 2;
        return size;
    }

    int64_t shape(const size_t i) const {
        return shape_[i];
    }

    const Coord & shape() const {
        return shape_;
    }

    template<class ITER>
    void serialize(ITER & iter) const {
        *iter = range_;
        ++iter;
        size_t nSlices = shape_[0];
        for(size_t slice = 0; slice < nSlices; ++slice) {
            *iter = numberOfEdgesInSlice_[slice];
            ++iter;
            *iter = edgeOffset_[slice];
            ++iter;
        }
        BaseType::serialize(iter);
    }


private:
    void initAdjacency(const Labels & labels, const size_t numberOfLabels, const int numberOfThreads);

    template<class ITER>
    void deserializeAdjacency(ITER & iter) {
        range_ = *iter;
        ++iter;
        size_t nSlices = shape_[0];
        for(size_t slice = 0; slice < nSlices; ++slice) {
            numberOfEdgesInSlice_[slice] = *iter;
            ++iter;
            edgeOffset_[slice] = *iter;
            ++iter;
        }
        BaseType::deserialize(iter);
    }

    Coord shape_;
    size_t range_;
    std::vector<size_t> numberOfEdgesInSlice_;
    std::vector<size_t> edgeOffset_;
};


template<class LABELS>
void LongRangeAdjacency<LABELS>::initAdjacency(const LABELS & labels, const size_t numberOfLabels, const int numberOfThreads) {

    typedef tools::BlockStorage<LabelType> LabelStorage;

    // set the number of nodes in the graph == number of labels
    BaseType::assign(numberOfLabels);

    // get the shape, number of slices and slice shapes
    const size_t nSlices = shape_[0];
    Coord2 sliceShape2({shape_[1], shape_[2]});
    Coord sliceShape3({1L, shape_[1], shape_[2]});

    // threadpool and actual number of threads
    nifty::parallel::ThreadPool threadpool(numberOfThreads);
    const size_t nThreads = threadpool.nThreads();

    std::vector<LabelType> minNodeInSlice(nSlices, numberOfLabels + 1);
    std::vector<LabelType> maxNodeInSlice(nSlices);

    // loop over the slices in parallel, for each slice find the edges
    // to nodes in the next 2 to 'range' slices
    {
        // instantiate the label storages
        LabelStorage labelsAStorage(threadpool, sliceShape3, nThreads);
        LabelStorage labelsBStorage(threadpool, sliceShape3, nThreads);

        parallel::parallel_foreach(threadpool, nSlices-2, [&](const int tid, const int slice) {

            // get segmentation in base slice
            Coord beginA ({int64_t(slice), 0L, 0L});
            Coord endA({int64_t(slice + 1), shape_[1], shape_[2]});
            auto labelsA = labelsAStorage.getView(tid);
            tools::readSubarray(labels, beginA, endA, labelsA);
            auto labelsASqueezed = labelsA.squeezedView();

            // iterate over the xy-coordinates and find the min and max nodes
            LabelType lU;
            auto & minNode = minNodeInSlice[slice];
            auto & maxNode = maxNodeInSlice[slice];
            tools::forEachCoordinate(sliceShape2, [&](const Coord2 coord){
                lU = labelsASqueezed(coord.asStdArray());
                minNode = std::min(minNode, lU);
                maxNode = std::max(maxNode, lU);
            });

            // get view for segmenation in upper slice
            auto labelsB = labelsBStorage.getView(tid);

            // iterate over the next 2 - range_ slices
            for(int64_t z = 2; z <= range_; ++z) {

                // we continue if the long range affinity would reach out of the data
                if(slice + z >= shape_[0]) {
                    continue;
                }

                // get upper segmentation
                Coord beginB ({slice + z, 0L, 0L});
                Coord endB({slice + z + 1, shape_[1], shape_[2]});
                tools::readSubarray(labels, beginB, endB, labelsB);
                auto labelsBSqueezed = labelsB.squeezedView();

                // iterate over the xy-coordinates and insert the long range edges
                LabelType lU, lV;
                tools::forEachCoordinate(sliceShape2, [&](const Coord2 coord){
                    lU = labelsASqueezed(coord.asStdArray());
                    lV = labelsBSqueezed(coord.asStdArray());
                    if(BaseType::insertEdgeOnlyInNodeAdj(lU, lV)){
                        ++numberOfEdgesInSlice_[slice]; // if this is the first time we hit this edge, increase the edge count
                    }
                });
            }
        });
    }

    // set up the edge offsets
    size_t offset = numberOfEdgesInSlice_[0];
    {
        edgeOffset_[0] = 0;
        for(size_t slice = 1; slice < nSlices-2; ++slice) {
            edgeOffset_[slice] = offset;
            offset += numberOfEdgesInSlice_[slice];
        }
    }

    // set up the edge indices
    {
        auto & edges = BaseType::edges_;
        auto & nodes = BaseType::nodes_;
        edges.resize(offset);
        parallel::parallel_foreach(threadpool, nSlices-2, [&](const int tid, const int64_t slice){

            auto edgeIndex = edgeOffset_[slice];
            const auto startNode = minNodeInSlice[slice];
            const auto endNode   = maxNodeInSlice[slice] + 1;

            for(uint64_t u = startNode; u < endNode; ++u){
                for(auto & vAdj : nodes[u]){
                    const auto v = vAdj.node();
                    if(u < v){
                        auto e = BaseType::EdgeStorage(u, v);
                        edges[edgeIndex] = e;
                        vAdj.changeEdgeIndex(edgeIndex);
                        auto fres =  nodes[v].find(NodeAdjacency(u));
                        fres->changeEdgeIndex(edgeIndex);
                        // increase the edge index
                        ++edgeIndex;
                    }
                }
            }
        });
    }
}

} // end namespace graph
} // end namespace nifty
