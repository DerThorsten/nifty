#pragma once

#include <vector>
#include <cmath>

#include "nifty/graph/rag/grid_rag.hxx"
#include "nifty/graph/rag/grid_rag_stacked_2d.hxx"

#include "nifty/marray/marray.hxx"
#include "nifty/tools/for_each_block.hxx"
#include "nifty/tools/array_tools.hxx"
#include "nifty/parallel/threadpool.hxx"
#include "vigra/accumulator.hxx"

namespace nifty {
namespace graph {

    // TODO implementations suitable for hdf5 and flat rag
    template<size_t DIM, class RAG_TYPE>
    class RagCoordinates {

    public:
        typedef RAG_TYPE RagType;
        typedef typename RagType:: template EdgeMap<std::vector<int32_t>> CoordinateStorageType;
        typedef array::StaticArray<int64_t, DIM> Coord;

        // constructor for complete volume
        RagCoordinates(const RagType & rag, const int nThreads = -1)
            : rag_(rag), storage_(rag) {
            initStorage(nThreads);
        }

        // returns the (topological) edge coordinates
        const std::vector<int32_t> & edgeCoordinates(const int64_t edgeId) const {
            return storage_[edgeId];
        }

        template<class T>
        void edgesToVolume(
                const std::vector<T> & edgeValues,
                marray::View<T> & out,
                const int edgeDirection = 0, // 0 -> both edge coordinates, 1-> lower edge coordinate, 2-> upper
                const T ignoreValue = 0,
                const int nThreads = -1) const;
        
        template<class T>
        void edgesToSubVolume(
                const std::vector<T> & edgeValues,
                marray::View<T> & out,
                const std::vector<int64_t> & begin,
                const std::vector<int64_t> & end,
                const int edgeDirection = 0, // 0 -> both edge coordinates, 1-> lower edge coordinate, 2-> upper
                const T ignoreValue = 0,
                const int nThreads = -1) const;

        const RagType & rag() const {
            return rag_;
        }

        size_t storageLengths() const {
            return storage_.size();
        }

    private:
        void initStorage(const int nThreads);

        template<class T>
        void writeBothCoordinates(
                const int64_t edgeId,
                const T edgeVal,
                marray::View<T> & out,
                const std::vector<int64_t> & offset = std::vector<int64_t>()
        ) const;

        template<class T>
        void writeLowerCoordinates(
                const int64_t edgeId,
                const T edgeVal,
                marray::View<T> & out,
                const std::vector<int64_t> & offset = std::vector<int64_t>()
        ) const;

        template<class T>
        void writeUpperCoordinates(
                const int64_t edgeId,
                const T edgeVal,
                marray::View<T> & out,
                const std::vector<int64_t> & offset = std::vector<int64_t>()
        ) const;

        const RagType & rag_;
        CoordinateStorageType storage_;
    };


    template<size_t DIM, class RAG_TYPE>
    template<class T>
    inline void RagCoordinates<DIM, RAG_TYPE>::writeBothCoordinates(const int64_t edgeId, const T edgeVal, marray::View<T> & out, const std::vector<int64_t> & offset) const {

        const auto & coords = edgeCoordinates(edgeId);
        Coord coordUp;
        Coord coordDn;
        for(size_t ii = 0; ii < coords.size() / DIM; ++ii) {
            for(size_t d = 0; d < DIM; ++d) {
                coordUp[d] = static_cast<int64_t>( ceil(float(coords[DIM * ii + d]) / 2) );
                coordDn[d] = static_cast<int64_t>( floor(float(coords[DIM * ii + d]) / 2) );
            }
            if( !offset.empty() ) {
                for(int d = 0; d < DIM; ++d) {
                    coordUp[d] -= offset[d];
                    coordDn[d] -= offset[d];
                }
            }
            out(coordUp.asStdArray()) = edgeVal;
            out(coordDn.asStdArray()) = edgeVal;
        }
    }


    template<size_t DIM, class RAG_TYPE>
    template<class T>
    inline void RagCoordinates<DIM, RAG_TYPE>::writeLowerCoordinates(const int64_t edgeId, const T edgeVal, marray::View<T> & out, const std::vector<int64_t> & offset) const {

        const auto & coords = edgeCoordinates(edgeId);
        Coord coord;
        for(size_t ii = 0; ii < coords.size() / DIM; ++ii) {
            for(size_t d = 0; d < DIM; ++d) {
                coord[d] = static_cast<int64_t>( floor(float(coords[DIM * ii + d]) / 2) );
            }
            if( !offset.empty() ) {
                for(int d = 0; d < DIM; ++d) {
                    coord[d] -= offset[d];
                }
            }
            out(coord.asStdArray()) = edgeVal;
        }
    }


    template<size_t DIM, class RAG_TYPE>
    template<class T>
    inline void RagCoordinates<DIM, RAG_TYPE>::writeUpperCoordinates(const int64_t edgeId, const T edgeVal, marray::View<T> & out, const std::vector<int64_t> & offset) const {

        const auto & coords = edgeCoordinates(edgeId);
        Coord coord;
        for(size_t ii = 0; ii < coords.size() / DIM; ++ii) {
            for(size_t d = 0; d < DIM; ++d) {
                coord[d] = static_cast<int64_t>( ceil(float(coords[DIM * ii + d]) / 2) );
            }
            if( !offset.empty() ) {
                for(int d = 0; d < DIM; ++d) {
                    coord[d] -= offset[d];
                }
            }
            out(coord.asStdArray()) = edgeVal;
        }
    }


    template<size_t DIM, class RAG_TYPE>
    void RagCoordinates<DIM, RAG_TYPE>::initStorage(const int nThreads) {

        typedef std::vector<std::vector<int32_t>> CoordinateVectorType;

        const auto numEdges = rag_.numberOfEdges();

        // read the labels
        const auto & shape  = rag_.shape();
        const auto & labelsProxy = rag_.labelsProxy();

        // FIXME dtype shouldn't be hard-coded
        marray::Marray<uint32_t> labels(shape.begin(), shape.end());
        Coord begin;
        for(int d = 0; d < DIM; ++d)
            begin[d] = 0;
        tools::readSubarray(labelsProxy, begin, shape, labels);

        nifty::parallel::ThreadPool threadpool(nThreads);
        std::vector<CoordinateVectorType> perThreadDataVec(threadpool.nThreads());
        for(size_t i=0; i<perThreadDataVec.size(); ++i)
            perThreadDataVec[i].resize(numEdges);

        auto makeCoord2 = [](const Coord & coord,const size_t axis){
            Coord coord2 = coord;
            coord2[axis] += 1;
            return coord2;
        };

        // extract the coordinates in parallel
        nifty::tools::parallelForEachCoordinate(threadpool, shape,[&](const int tid, const Coord & coord){

            auto & edgeCoords = perThreadDataVec[tid];
            const auto lU = labels(coord.asStdArray());
            for(size_t axis=0; axis<DIM; ++axis){
                const auto coord2 = makeCoord2(coord, axis);
                if(coord2[axis] < shape[axis]){
                    const auto lV = labels(coord2.asStdArray());
                    if(lU != lV){
                        const auto edgeId = rag_.findEdge(lU,lV);
                        for(int d = 0; d < DIM; ++d) {
                            edgeCoords[edgeId].push_back(coord[d] + coord2[d]); // we append the topological coordinate == sum
                        }
                    }
                }
            }

        });

        // merge the coordinates
        parallel::parallel_foreach(threadpool, numEdges, [&](const int tid, const int edgeId) {
            auto & outData = storage_[edgeId];
            for(int t = 0; t < perThreadDataVec.size(); ++t) {
                const auto & threadData = perThreadDataVec[t][edgeId];
                for(const auto coordVal : threadData)
                    outData.push_back(coordVal);
            }
        });

    }


    template<size_t DIM, class RAG_TYPE>
    template<class T>
    void RagCoordinates<DIM, RAG_TYPE>::edgesToVolume(
            const std::vector<T> & edgeValues,
            marray::View<T> & out,
            const int edgeDirection, // 0 -> both edge coordinates, 1-> lower edge coordinate, 2-> upper
            const T ignoreValue,
            const int nThreads) const {

        NIFTY_CHECK_OP(edgeValues.size(),==,storageLengths(),"Wrong number of edges");
        const auto numEdges = edgeValues.size();
        nifty::parallel::ThreadPool threadpool(nThreads);

        parallel::parallel_foreach(threadpool, numEdges, [&](const int tid, const int edgeId) {

            auto edgeVal = edgeValues[edgeId];
            if(edgeVal == ignoreValue) {
                return;
            }

            if(edgeDirection == 0) {
                writeBothCoordinates(edgeId, edgeVal, out);
            }
            else if(edgeDirection == 1) {
                writeLowerCoordinates(edgeId, edgeVal, out);
            }
            else if(edgeDirection == 2) {
                writeUpperCoordinates(edgeId, edgeVal, out);
            }
            else {
                throw std::runtime_error("Invalid edge direction value");
            }

        });

    }


    template<size_t DIM, class RAG_TYPE>
    template<class T>
    void RagCoordinates<DIM, RAG_TYPE>::edgesToSubVolume(
            const std::vector<T> & edgeValues,
            marray::View<T> & out,
            const std::vector<int64_t> & roiBegin,
            const std::vector<int64_t> & roiEnd,
            const int edgeDirection, // 0 -> both edge coordinates, 1-> lower edge coordinate, 2-> upper
            const T ignoreValue,
            const int nThreads) const {

        NIFTY_CHECK_OP(edgeValues.size(),==,storageLengths(),"Wrong number of edges");
        const auto numEdges = edgeValues.size();

        Coord shape, begin, end;
        for(int d = 0; d < DIM; ++d) {
            begin[d] = roiBegin[d];
            end[d] = roiEnd[d];
            shape[d] = end[d] - begin[d];
        }
        // TODO check for correct out shape

        // keep only the edge values that are in our subvolume

        std::vector<int64_t> subEdges;
        std::cout << "Before subgraph" << std::endl;
        const auto subGraph = rag_.extractSubgraphFromRoi(begin, end, subEdges);

        std::cout << "Have subgraph, going to loop" << std::endl;
        nifty::parallel::ThreadPool threadpool(nThreads);
        parallel::parallel_foreach(threadpool, subEdges.size(), [&](const int tid, const int subEdgeId) {

            auto edgeId = subEdges[subEdgeId];
            auto edgeVal = edgeValues[edgeId];
            if(edgeVal == ignoreValue) {
                return;
            }

            if(edgeDirection == 0) {
                writeBothCoordinates(edgeId, edgeVal, out, roiBegin);
            }
            else if(edgeDirection == 1) {
                writeLowerCoordinates(edgeId, edgeVal, out, roiBegin);
            }
            else if(edgeDirection == 2) {
                writeUpperCoordinates(edgeId, edgeVal, out, roiBegin);
            }
            else {
                throw std::runtime_error("Invalid edge direction value");
            }

        });
        std::cout << "Have looped" << std::endl;

    }

} // namespace graph
} // namespace nifty
