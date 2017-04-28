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

    typedef std::vector<std::vector<uint64_t>> CoordinateVectorType;

    // FIXME this only works for a grid rag with explicit labels
    // for hdf5 labels we need to iterate over the blocks
    template<size_t DIM, class LABELS_PROXY>
    void computeEdgeCoordinates(
        const GridRag<DIM, LABELS_PROXY> & rag,
        CoordinateVectorType & coordinatesOut,
        const int numberOfThreads = -1
    ){
        // returns flattened coordinate vector

        typedef array::StaticArray<int64_t, DIM> Coord;
        
        const auto numEdges = rag.numberOfEdges();
        const auto & shape  = rag.shape(); 
        const auto labelsProxy = rag.labelsProxy();
        const auto numberOfLabels = labelsProxy.numberOfLabels();
        const auto labels = labelsProxy.labels(); 

        nifty::parallel::ThreadPool threadpool(numberOfThreads);
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
                        const auto edgeId = rag.findEdge(lU,lV);
                        for(int d = 0; d < DIM; ++d) {
                            edgeCoords[edgeId].push_back(coord[d] + coord2[d]); // we append the topological coordinate == sum 
                        }
                    }
                }
            }
        });

        // merge the coordinates
        coordinatesOut.resize(numEdges);
        nifty::parallel::parallel_foreach(threadpool, numEdges,
                [&perThreadDataVec, &coordinatesOut](const int tid, const int edgeId) {
            auto & outData = coordinatesOut[edgeId];
            for(int t = 0; t < perThreadDataVec.size(); ++t) {
                const auto & threadData = perThreadDataVec[t][edgeId];
                for(const auto coordVal : threadData)
                    outData.push_back(coordVal);
            }
        });
    }

} // namespace graph
} // namespace nifty
