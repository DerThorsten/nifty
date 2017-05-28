#pragma once

#include <vector>
#include <cmath>

#include "nifty/graph/rag/grid_rag.hxx"
#include "nifty/marray/marray.hxx"
#include "nifty/tools/for_each_block.hxx"
#include "nifty/parallel/threadpool.hxx"
#include "vigra/accumulator.hxx"

namespace nifty {
namespace graph {

    // TODO implementations suitable for hdf5 and flat rag
    template<size_t DIM, class LABELS_PROXY>
    class RagCoordinates {

    public:
        typedef GridRag<DIM, LABELS_PROXY> RagType;
        typedef typename RagType:: template EdgeMap<std::vector<int32_t>> CoordinateStorageType;
        typedef array::StaticArray<int64_t, DIM> Coord;

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

        const RagType & rag() const {
            return rag_;
        }

    private:
        void initStorage(const int nThreads);

        template<class T>
        void writeBothCoordinates(const int64_t edgeId, const T edgeVal, marray::View<T> & out) const;

        template<class T>
        void writeLowerCoordinates(const int64_t edgeId, const T edgeVal, marray::View<T> & out) const;

        template<class T>
        void writeUpperCoordinates(const int64_t edgeId, const T edgeVal, marray::View<T> & out) const;

        const RagType & rag_;
        CoordinateStorageType storage_;
    };


    template<size_t DIM, class LABELS_PROXY>
    template<class T>
    inline void RagCoordinates<DIM, LABELS_PROXY>::writeBothCoordinates(const int64_t edgeId, const T edgeVal, marray::View<T> & out) const {

        const auto & coords = edgeCoordinates(edgeId);
        Coord coordUp;
        Coord coordDn;
        for(size_t ii = 0; ii < coords.size() / DIM; ++ii) {
            for(size_t d = 0; d < DIM; ++d) {
                coordUp[d] = static_cast<int64_t>( ceil(float(coords[DIM * ii + d]) / 2) );
                coordDn[d] = static_cast<int64_t>( floor(float(coords[DIM * ii + d]) / 2) );
            }
            out(coordUp.asStdArray()) = edgeVal;
            out(coordDn.asStdArray()) = edgeVal;
        }
    }


    template<size_t DIM, class LABELS_PROXY>
    template<class T>
    inline void RagCoordinates<DIM, LABELS_PROXY>::writeLowerCoordinates(const int64_t edgeId, const T edgeVal, marray::View<T> & out) const {

        const auto & coords = edgeCoordinates(edgeId);
        Coord coord;
        for(size_t ii = 0; ii < coords.size() / DIM; ++ii) {
            for(size_t d = 0; d < DIM; ++d) {
                coord[d] = static_cast<int64_t>( floor(float(coords[DIM * ii + d]) / 2) );
            }
            out(coord.asStdArray()) = edgeVal;
        }
    }


    template<size_t DIM, class LABELS_PROXY>
    template<class T>
    inline void RagCoordinates<DIM, LABELS_PROXY>::writeUpperCoordinates(const int64_t edgeId, const T edgeVal, marray::View<T> & out) const {

        const auto & coords = edgeCoordinates(edgeId);
        Coord coord;
        for(size_t ii = 0; ii < coords.size() / DIM; ++ii) {
            for(size_t d = 0; d < DIM; ++d) {
                coord[d] = static_cast<int64_t>( ceil(float(coords[DIM * ii + d]) / 2) );
            }
            out(coord.asStdArray()) = edgeVal;
        }
    }


    template<size_t DIM, class LABELS_PROXY>
    void RagCoordinates<DIM, LABELS_PROXY>::initStorage(const int nThreads) {

        typedef std::vector<std::vector<int32_t>> CoordinateVectorType;

        const auto numEdges = rag_.numberOfEdges();
        const auto & shape  = rag_.shape();
        const auto & labelsProxy = rag_.labelsProxy();
        const auto & labels = labelsProxy.labels();

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


    template<size_t DIM, class LABELS_PROXY>
    template<class T>
    void RagCoordinates<DIM, LABELS_PROXY>::edgesToVolume(
            const std::vector<T> & edgeValues,
            marray::View<T> & out,
            const int edgeDirection, // 0 -> both edge coordinates, 1-> lower edge coordinate, 2-> upper
            const T ignoreValue,
            const int nThreads) const {

        NIFTY_CHECK_OP(edgeValues.size(),==,rag_.numberOfEdges(),"Wrong number of edges");
        const auto numEdges = rag_.numberOfEdges();
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

} // namespace graph
} // namespace nifty
