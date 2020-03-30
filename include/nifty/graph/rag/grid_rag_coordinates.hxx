#pragma once

#include <vector>
#include <cmath>

#include "nifty/graph/rag/grid_rag.hxx"
#include "nifty/graph/rag/grid_rag_stacked_2d.hxx"

#include "nifty/tools/for_each_block.hxx"
#include "nifty/tools/array_tools.hxx"
#include "nifty/parallel/threadpool.hxx"
#include "vigra/accumulator.hxx"

#include "nifty/xtensor/xtensor.hxx"

namespace nifty {
namespace graph {

    // TODO implementations suitable for hdf5 and flat rag
    template<std::size_t DIM, class RAG_TYPE>
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

        template<class EDGES, class ARRAY>
        void edgesToVolume(const xt::xexpression<EDGES> & edgeValues,
                           xt::xexpression<ARRAY> & out,
                           const int edgeDirection = 0, // 0 -> both edge coordinates, 1-> lower edge coordinate, 2-> upper
                           const typename EDGES::value_type ignoreValue = 0,
                           const int nThreads = -1) const;

        template<class T, class ARRAY>
        void edgesToSubVolume(const std::vector<T> & edgeValues,
                              xt::xexpression<ARRAY> & out,
                              const std::vector<int64_t> & begin,
                              const std::vector<int64_t> & end,
                              const int edgeDirection = 0, // 0 -> both edge coordinates, 1-> lower edge coordinate, 2-> upper
                              const T ignoreValue = 0,
                              const int nThreads = -1) const;

        const RagType & rag() const {
            return rag_;
        }

        std::size_t storageLengths() const {
            return storage_.size();
        }

    private:
        void initStorage(const int nThreads);

        template<class T, class ARRAY>
        void writeBothCoordinates(
                const int64_t edgeId,
                const T edgeVal,
                xt::xexpression<ARRAY> & out,
                const std::vector<int64_t> & offset = std::vector<int64_t>()
        ) const;

        template<class T, class ARRAY>
        void writeLowerCoordinates(
                const int64_t edgeId,
                const T edgeVal,
                xt::xexpression<ARRAY> & out,
                const std::vector<int64_t> & offset = std::vector<int64_t>()
        ) const;

        template<class T, class ARRAY>
        void writeUpperCoordinates(
                const int64_t edgeId,
                const T edgeVal,
                xt::xexpression<ARRAY> & out,
                const std::vector<int64_t> & offset = std::vector<int64_t>()
        ) const;

        // center the coordinate to new offset and return false if it is not in the ROI
        static bool cropCoordinate(Coord & coordinate, const std::vector<int64_t> & offset, const Coord & outShape) {
            bool inRoi = true;
            for(int d = 0; d < DIM; ++d) {
                coordinate[d] -= offset[d];
                if(coordinate[d] >= outShape[d]) {
                    inRoi = false;
                }
            }
            return inRoi;
        }

        const RagType & rag_;
        CoordinateStorageType storage_;
    };


    template<std::size_t DIM, class RAG_TYPE>
    template<class T, class ARRAY>
    inline void RagCoordinates<DIM, RAG_TYPE>::writeBothCoordinates(const int64_t edgeId,
                                                                    const T edgeVal,
                                                                    xt::xexpression<ARRAY> & outExp,
                                                                    const std::vector<int64_t> & offset) const {

        auto & out = outExp.derived_cast();
        const auto & arrayShape = out.shape();

        const auto & coords = edgeCoordinates(edgeId);
        Coord coordUp, coordDn, outShape;
        for(int d = 0; d < DIM; ++d) {
            outShape[d] = arrayShape[d];
        }

        for(std::size_t ii = 0; ii < coords.size() / DIM; ++ii) {
            for(std::size_t d = 0; d < DIM; ++d) {
                coordUp[d] = static_cast<int64_t>( ceil(float(coords[DIM * ii + d]) / 2) );
                coordDn[d] = static_cast<int64_t>( floor(float(coords[DIM * ii + d]) / 2) );
            }

            // we need to shift and crop the coordinates if we deal with a subvolume
            if( !offset.empty() ) {
                auto inRoiUp = cropCoordinate(coordUp, offset, outShape);
                auto inRoiDn = cropCoordinate(coordDn, offset, outShape);
                if(!(inRoiUp && inRoiDn)) {
                    continue;
                }
            }

            xtensor::write(out, coordUp.asStdArray(), edgeVal);
            xtensor::write(out, coordDn.asStdArray(), edgeVal);
        }
    }


    template<std::size_t DIM, class RAG_TYPE>
    template<class T, class ARRAY>
    inline void RagCoordinates<DIM, RAG_TYPE>::writeLowerCoordinates(const int64_t edgeId, 
                                                                     const T edgeVal, 
                                                                     xt::xexpression<ARRAY> & outExp, 
                                                                     const std::vector<int64_t> & offset) const {

        auto & out = outExp.derived_cast();
        const auto & arrayShape = out.shape();

        const auto & coords = edgeCoordinates(edgeId);
        Coord coord, outShape;
        for(int d = 0; d < DIM; ++d) {
            outShape[d] = arrayShape[d];
        }

        for(std::size_t ii = 0; ii < coords.size() / DIM; ++ii) {
            for(std::size_t d = 0; d < DIM; ++d) {
                coord[d] = static_cast<int64_t>( floor(float(coords[DIM * ii + d]) / 2) );
            }
            if(!offset.empty()) {
                if(!cropCoordinate(coord, offset, outShape)) {
                    continue;
                }
            }
            xtensor::write(out, coord.asStdArray(), edgeVal);
        }
    }


    template<std::size_t DIM, class RAG_TYPE>
    template<class T, class ARRAY>
    inline void RagCoordinates<DIM, RAG_TYPE>::writeUpperCoordinates(const int64_t edgeId,
                                                                     const T edgeVal,
                                                                     xt::xexpression<ARRAY> & outExp,
                                                                     const std::vector<int64_t> & offset) const {
        auto & out = outExp.derived_cast();
        const auto & arrayShape = out.shape();

        const auto & coords = edgeCoordinates(edgeId);
        Coord coord, outShape;
        for(int d = 0; d < DIM; ++d) {
            outShape[d] = arrayShape[d];
        }

        for(std::size_t ii = 0; ii < coords.size() / DIM; ++ii) {
            for(std::size_t d = 0; d < DIM; ++d) {
                coord[d] = static_cast<int64_t>( ceil(float(coords[DIM * ii + d]) / 2) );
            }
            if( !offset.empty() ) {
                if(!cropCoordinate(coord, offset, outShape)) {
                    continue;
                }
            }
            xtensor::write(out, coord.asStdArray(), edgeVal);
        }
    }


    template<std::size_t DIM, class RAG_TYPE>
    void RagCoordinates<DIM, RAG_TYPE>::initStorage(const int nThreads) {

        typedef std::vector<std::vector<int32_t>> CoordinateVectorType;
        typedef typename xt::xtensor<uint32_t, DIM>::shape_type ShapeType;

        const auto numEdges = rag_.numberOfEdges();

        // read the labels
        const auto & shape  = rag_.shape();
        const auto & ragLabels = rag_.labels();

        ShapeType arrayShape;
        std::copy(shape.begin(), shape.end(), arrayShape.begin());

        // FIXME dtype shouldn't be hard-coded
        xt::xtensor<uint32_t, DIM> labels(arrayShape);
        Coord begin;
        for(int d = 0; d < DIM; ++d)
            begin[d] = 0;
        tools::readSubarray(ragLabels, begin, shape, labels);

        nifty::parallel::ThreadPool threadpool(nThreads);
        std::vector<CoordinateVectorType> perThreadDataVec(threadpool.nThreads());
        for(std::size_t i=0; i<perThreadDataVec.size(); ++i)
            perThreadDataVec[i].resize(numEdges);

        auto makeCoord2 = [](const Coord & coord,const std::size_t axis){
            Coord coord2 = coord;
            coord2[axis] += 1;
            return coord2;
        };

        // extract the coordinates in parallel
        nifty::tools::parallelForEachCoordinate(threadpool, shape,[&](const int tid, const Coord & coord){

            auto & edgeCoords = perThreadDataVec[tid];
            const auto lU = xtensor::read(labels, coord.asStdArray());
            for(std::size_t axis=0; axis<DIM; ++axis){
                const auto coord2 = makeCoord2(coord, axis);
                if(coord2[axis] < shape[axis]){
                    const auto lV = xtensor::read(labels, coord2.asStdArray());
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


    template<std::size_t DIM, class RAG_TYPE>
    template<class EDGES, class ARRAY>
    void RagCoordinates<DIM, RAG_TYPE>::edgesToVolume(
            const xt::xexpression<EDGES> & edgeValuesExp,
            xt::xexpression<ARRAY> & outExp,
            const int edgeDirection, // 0 -> both edge coordinates, 1-> lower edge coordinate, 2-> upper
            const typename EDGES::value_type ignoreValue,
            const int nThreads) const {
        const auto & edgeValues = edgeValuesExp.derived_cast();
        auto & out = outExp.derived_cast();

        NIFTY_CHECK_OP(edgeValues.size(),==,storageLengths(),"Wrong number of edges");
        const auto numEdges = edgeValues.size();
        nifty::parallel::ThreadPool threadpool(nThreads);

        parallel::parallel_foreach(threadpool, numEdges, [&](const int tid, const int edgeId) {

            const auto edgeVal = edgeValues(edgeId);
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


    template<std::size_t DIM, class RAG_TYPE>
    template<class T, class ARRAY>
    void RagCoordinates<DIM, RAG_TYPE>::edgesToSubVolume(
            const std::vector<T> & edgeValues,
            xt::xexpression<ARRAY> & outExp,
            const std::vector<int64_t> & roiBegin,
            const std::vector<int64_t> & roiEnd,
            const int edgeDirection, // 0 -> both edge coordinates, 1-> lower edge coordinate, 2-> upper
            const T ignoreValue,
            const int nThreads) const {

        auto & out = outExp.derived_cast();

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
        const auto subGraph = rag_.extractSubgraphFromRoi(begin, end, subEdges);

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

    }

} // namespace graph
} // namespace nifty
