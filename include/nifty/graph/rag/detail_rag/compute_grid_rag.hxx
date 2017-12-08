#pragma once

#include <functional>
#include <algorithm>
#include <map>
#include <cstddef>

#include "nifty/parallel/threadpool.hxx"
#include "nifty/container/boost_flat_set.hxx"
#include "nifty/array/arithmetic_array.hxx"
#include "nifty/tools/for_each_block.hxx"

#include "nifty/graph/rag/grid_rag_labels_proxy.hxx"
#include "nifty/graph/undirected_list_graph.hxx"

#include "nifty/xtensor/xtensor.hxx"


namespace nifty{
namespace graph{


template<std::size_t DIM, class LABELS_PROXY>
class GridRag;


// \cond SUPPRESS_DOXYGEN
namespace detail_rag{

template<class GRID_RAG>
struct ComputeRag;


template<std::size_t DIM, class LABELS_PROXY>
struct ComputeRag<GridRag<DIM, LABELS_PROXY>> {

    typedef LABELS_PROXY LabelsProxyType;
    typedef typename LabelsProxyType::LabelType LabelType;;

    template<class S>
    static void computeRag(GridRag<DIM, LabelsProxyType> & rag,
                           const S & settings){
        //
        typedef array::StaticArray<int64_t, DIM> Coord;

        const auto & labelsProxy = rag.labelsProxy();
        const auto & shape = labelsProxy.shape();

        const auto numberOfLabels = labelsProxy.numberOfLabels();
        rag.assign(numberOfLabels);

        nifty::parallel::ParallelOptions pOpts(settings.numberOfThreads);
        nifty::parallel::ThreadPool threadpool(pOpts);
        const auto nThreads = pOpts.getActualNumThreads();

        // allocate / create data for each thread
        Coord blockShapeWithBorder;
        for(auto d=0; d<DIM; ++d){
            blockShapeWithBorder[d] = std::min(settings.blockShape[d]+1, shape[d]);
        }

        struct PerThreadData{
            xt::xtensor<LabelType, DIM> blockLabels;
            std::vector< container::BoostFlatSet<uint64_t> > adjacency;
        };

        // FIXME xarrays can't be constructed with the nifty shape type
        // we should get rid of it....
        std::vector<size_t> arrayShape(blockShapeWithBorder.begin(), blockShapeWithBorder.end());
        std::vector<PerThreadData> perThreadDataVec(nThreads);
        parallel::parallel_foreach(threadpool, nThreads, [&](const int tid, const int i){
            perThreadDataVec[i].blockLabels.reshape(arrayShape);
            perThreadDataVec[i].adjacency.resize(numberOfLabels);
        });

        auto makeCoord2 = [](const Coord & coord,const std::size_t axis){
            Coord coord2 = coord;
            coord2[axis] += 1;
            return coord2;
        };

        const Coord overlapBegin(0), overlapEnd(1);
        const Coord zeroCoord(0);

        std::mutex mtx;
        auto doneBlocks = 0;
        auto nBlocks = 0;

        nifty::parallel::ThreadPool stSp(nifty::parallel::ParallelOptions(1));
        tools::parallelForEachBlockWithOverlap(stSp, shape, settings.blockShape, overlapBegin, overlapEnd,
        [&](
            const int tid,
            const Coord & blockCoreBegin, const Coord & blockCoreEnd,
            const Coord & blockBegin, const Coord & blockEnd
        ){
            nBlocks += 1;
        });


        std::cout<<"nBlocks "<<nBlocks<<"\n";

        tools::parallelForEachBlockWithOverlap(threadpool, shape, settings.blockShape, overlapBegin, overlapEnd,
        [&](
            const int tid,
            const Coord & blockCoreBegin, const Coord & blockCoreEnd,
            const Coord & blockBegin, const Coord & blockEnd
        ){
            const Coord actualBlockShape = blockEnd - blockBegin;
            auto & blockView = perThreadDataVec[tid].blockLabels;
            xt::slice_vector slice(blockView);
            xtensor::sliceFromRoi(slice, zeroCoord, actualBlockShape);
            auto blockLabels = xt::dynamic_view(blockView, slice);

            // TODO the lock should be unnecessary !
            //mtx.lock();
            labelsProxy.readSubarray(blockBegin, blockEnd, blockLabels);
            //mtx.unlock();

            auto & adjacency = perThreadDataVec[tid].adjacency;
            nifty::tools::forEachCoordinate(actualBlockShape,[&](const Coord & coord){
                const auto lU = xtensor::read(blockLabels, coord.asStdArray());
                for(std::size_t axis=0; axis<DIM; ++axis){
                    const auto coord2 = makeCoord2(coord, axis);
                    if(coord2[axis] < actualBlockShape[axis]){
                        const auto lV = xtensor::read(blockLabels, coord2.asStdArray());
                        if(lU != lV){
                            adjacency[lV].insert(lU);
                            adjacency[lU].insert(lV);
                        }
                    }
                }
            });
            mtx.lock();
                doneBlocks += 1;
                std::cout<<doneBlocks<<" / "<<nBlocks<<" "<<100.0*float(doneBlocks+1)/float(nBlocks)<<"\n";
            mtx.unlock();


        });

        rag.mergeAdjacencies(perThreadDataVec, threadpool);
    }
};


} // end namespace detail_rag
// \endcond

} // end namespace graph
} // end namespace nifty
