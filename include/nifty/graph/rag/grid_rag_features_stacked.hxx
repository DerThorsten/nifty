#pragma once

#include "nifty/graph/rag/grid_rag_stacked_2d.hxx"

#ifdef WITH_HDF5
#include "nifty/hdf5/hdf5_array.hxx"
#endif

#include "nifty/tools/for_each_coordinate.hxx"
#include "nifty/tools/array_tools.hxx"
#include "nifty/tools/memory.hxx"
#include "nifty/tools/runtime_check.hxx"

#include "xtensor/xarray.hpp"
#include "nifty/xtensor/xtensor.hxx"



namespace nifty{
namespace graph{

    template<class LABELS_PROXY, class LABELS, class NODE_MAP>
    void gridRagAccumulateLabels(
        const GridRagStacked2D<LABELS_PROXY> & graph,
        const LABELS & data,
        NODE_MAP & nodeMap,
        const int numberOfThreads = -1
    ){

        typedef LABELS_PROXY LabelsProxyType;
        typedef typename LABELS_PROXY::LabelType LabelType;
        typedef typename LabelsProxyType::BlockStorageType LabelsBlockStorage;
        typedef typename tools::BlockStorage<typename LABELS::value_type> DataBlockStorage;

        typedef array::StaticArray<int64_t, 3> Coord;
        typedef array::StaticArray<int64_t, 2> Coord2;

        const auto & labelsProxy = graph.labelsProxy();
        const auto & shape = labelsProxy.shape();
        const auto & dataShape = data.shape();

        NIFTY_CHECK_OP(dataShape[0],==,shape[0], "Shape along z does not agree")
        NIFTY_CHECK_OP(dataShape[1],==,shape[1], "Shape along y does not agree")
        NIFTY_CHECK_OP(dataShape[2],==,shape[2], "Shape along x does not agree")

        nifty::parallel::ParallelOptions pOpts(numberOfThreads);
        nifty::parallel::ThreadPool threadpool(pOpts);
        const auto nThreads = pOpts.getActualNumThreads();

        uint64_t numberOfSlices = shape[0];
        Coord2 sliceShape2({shape[1], shape[2]});
        Coord sliceShape3({1L,shape[1], shape[2]});

        std::vector<  std::unordered_map<uint64_t, uint64_t> > overlaps(graph.numberOfNodes());

        LabelsBlockStorage sliceLabelsStorage(threadpool, sliceShape3, nThreads);
        DataBlockStorage   sliceDataStorage(threadpool, sliceShape3, nThreads);

        parallel::parallel_foreach(threadpool, numberOfSlices, [&](const int tid, const int64_t sliceIndex){

            // fetch the data for the slice
            auto sliceLabelsFlat3DView = sliceLabelsStorage.getView(tid);
            auto sliceDataFlat3DView   = sliceDataStorage.getView(tid);

            const Coord blockBegin({sliceIndex,0L,0L});
            const Coord blockEnd({sliceIndex+1, sliceShape2[0], sliceShape2[1]});

            tools::readSubarray(labelsProxy, blockBegin, blockEnd, sliceLabelsFlat3DView);
            tools::readSubarray(data, blockBegin, blockEnd, sliceDataFlat3DView);

            auto sliceLabels = xtensor::squeezedView(sliceLabelsFlat3DView);
            auto sliceData = xtensor::squeezedView(sliceDataFlat3DView);

            nifty::tools::forEachCoordinate(sliceShape2,[&](const Coord2 & coord){
                const auto node = xtensor::access(sliceLabels, coord.asStdArray());
                const auto l    = xtensor::access(sliceData, coord.asStdArray());
                overlaps[node][l] += 1;
            });

        });

        parallel::parallel_foreach(threadpool, graph.numberOfNodes(), [&](const int tid, const int64_t nodeId){
            const auto & ol = overlaps[nodeId];
            // find max ol
            uint64_t maxOl = 0 ;
            uint64_t maxOlLabel = 0;
            for(auto kv : ol){
                if(kv.second > maxOl){
                    maxOl = kv.second;
                    maxOlLabel = kv.first;
                }
            }
            nodeMap[nodeId] = maxOlLabel;
        });
    }


    // TODO to lazy to port this to xtensor right now
    /*
    template<class LABELS_PROXY, class NODE_TYPE>
    inline void getSkipEdgesForNode(
        const LABELS_PROXY & labels,
        const int64_t zUp,
        const int64_t zDn,
        std::map<size_t, std::vector<NODE_TYPE>> & defectNodes,
        const marray::View<NODE_TYPE> & segDn,
        const std::vector<NODE_TYPE> & nodesDn,
        std::map<NODE_TYPE,std::vector<array::StaticArray<int64_t,2>>> & coordsToNodesDn,
        const marray::View<bool> & mask,
        std::map<int64_t,
            std::unique_ptr<marray::Marray<NODE_TYPE>>> & upperSegMap,
        std::vector<std::pair<NODE_TYPE,NODE_TYPE>> & skipEdges,
        std::vector<size_t> & skipRanges
    ){
        typedef NODE_TYPE NodeType;
        typedef array::StaticArray<int64_t, 3> Coord;
        typedef array::StaticArray<int64_t, 2> Coord2;

        skipEdges.clear();
        skipRanges.clear();

        size_t skipRange = zUp - zDn;

        // read the upper segmentation
        auto upperSegIt = upperSegMap.find(zUp);
        if(upperSegIt == upperSegMap.end()) { // this is the first time we reach this upper slice, so we need to read the labels from hdf5

            auto & shape = labels.shape();
            Coord sliceShape({1L, shape[1], shape[2]});
            upperSegMap[zUp] = tools::make_unique<marray::Marray<NodeType>>(sliceShape.begin(), sliceShape.end()); // C++ 14 ?!
            Coord sliceUpStart({zUp, 0L, 0L});
            Coord sliceUpStop({zUp+1, shape[1], shape[2]});

            std::cout << "Read Upper slice: " << zUp << std::endl;
            labels.readSubarray(sliceUpStart, sliceUpStop, *upperSegMap[zUp]);
            upperSegIt = upperSegMap.find(zUp);
        }
        marray::View<NodeType> segUp = upperSegIt->second->squeezedView();

        for(auto uDn : nodesDn) {

            const auto & coordsDn = coordsToNodesDn[uDn];

            // find intersecting nodes in upper slice
            std::vector<NodeType> connectedNodes;
            tools::uniquesWithMaskAndCoordinates<2>(segUp, mask, coordsDn, connectedNodes);

            // if any of the nodes is defected got to the next slice
            bool upperDefect = false;
            const auto & defectNodesUp = defectNodes[zUp];

            for(auto vUp : connectedNodes) {
                if(std::find(defectNodesUp.begin(), defectNodesUp.end(), vUp) != defectNodesUp.end()) {
                    upperDefect = true;
                    break;
                }
            }

            if(upperDefect){
                getSkipEdgesForNode(
                    labels,
                    zUp + 1,
                    zDn,
                    defectNodes,
                    segDn,
                    nodesDn,
                    coordsToNodesDn,
                    mask,
                    upperSegMap,
                    skipEdges,
                    skipRanges);
                break;
            }
            else {
                for(auto vUp : connectedNodes){
                    skipEdges.emplace_back(std::make_pair(uDn, vUp));
                    skipRanges.emplace_back(skipRange);
                }
            }
        }
    }

    // skip edges and stuff
    template<class LABELS_PROXY, class NODE_TYPE, class EDGE_TYPE>
    void getSkipEdgesForSlice(
        const GridRagStacked2D<LABELS_PROXY> & rag,
        const int64_t z,
        std::map<size_t,std::vector<NODE_TYPE>> & defectNodes,
        std::vector<EDGE_TYPE> & deleteEdges, // ref to outvec
        std::vector<EDGE_TYPE> & ignoreEdges, // ref to outvec
        std::vector<std::pair<NODE_TYPE,NODE_TYPE>> & skipEdges, // skip edges, ref to outvec
        std::vector<size_t> & skipRanges, // skip ranges,ref to outvec
        const bool lowerIsCompletelyDefected = false
    ){

        typedef EDGE_TYPE EdgeType;
        typedef NODE_TYPE NodeType;

        typedef array::StaticArray<int64_t, 3> Coord;
        typedef array::StaticArray<int64_t, 2> Coord2;

        auto & shape = rag.shape();
        auto & labelsProxy = rag.labelsProxy();

        auto edgeOffset = rag.numberOfInSliceEdges();

        Coord sliceShape({1L, shape[1], shape[2]});
        Coord2 sliceShape2({shape[1], shape[2]});

        // read the segmentation in the defected slice
        marray::Marray<NodeType> segZArray(sliceShape.begin(), sliceShape.end());
        Coord sliceStart({z, 0L, 0L});
        Coord sliceStop({z+1, shape[1], shape[2]});

        std::cout << "Read defected slice: " << z << std::endl;
        labelsProxy.readSubarray(sliceStart, sliceStop, segZArray);
        marray::View<NodeType> segZ = segZArray.squeezedView();

        // read the segmentation in the lower slice
        marray::Marray<NodeType> segDnArray(sliceShape.begin(), sliceShape.end());
        Coord sliceDnStart({z-1, 0L, 0L});
        Coord sliceDnStop({z, shape[1], shape[2]});

        // don't need lower segmentation for first and last slice or if the lower slice is completely defected
        if(z > 0 && z < shape[0] - 1 && !lowerIsCompletelyDefected) {
            std::cout << "Read Lower slice: " << z-1 << std::endl;
            labelsProxy.readSubarray(sliceDnStart, sliceDnStop, segDnArray);
        }
        marray::View<NodeType> segDn = segDnArray.squeezedView();

        //FIXME call by refereve is broke for some reason
        // find coordinates to nodes for defect slice and lower
        std::map<NodeType,std::vector<Coord2>> coordsToNodes;
        tools::valuesToCoordinates<2,NodeType>(segZ, coordsToNodes);

        // read segmentation for the upper slice and store pointer to it in a map
        // TODO do we need a pointer here?
        std::map<int64_t, std::unique_ptr<marray::Marray<NodeType>>> upperSegMap;
        upperSegMap[z+1] = tools::make_unique<marray::Marray<NodeType>>(sliceShape.begin(), sliceShape.end()); // C++ 14 ?!
        Coord sliceUpStart({z+1, 0L, 0L});
        Coord sliceUpStop({z+2, shape[1], shape[2]});

        // don't need upper segmentation for first and last slice or if the lower slice is completely defected
        if(z > 0 && z < shape[0] - 1 && !lowerIsCompletelyDefected) {
            std::cout << "Read Upper slice: " << z+1 << std::endl;
            labelsProxy.readSubarray(sliceUpStart, sliceUpStop, *upperSegMap[z+1]);
        }

        // defect mask
        marray::Marray<bool> mask(sliceShape2.begin(), sliceShape2.end(), false);

        const auto & defectNodesZ  = defectNodes[z];

        std::vector<Coord2> coordsUPrev;

        for(size_t i = 0; i < defectNodesZ.size(); ++i) {

            auto u = defectNodesZ[i];

            //std::cout << "Processing Node " << u << ": " << i << "/" << defectNodesZ.size() << std::endl;

            // remove edges and find edges connecting defected to non defected nodes
            for(auto vIt = rag.adjacencyBegin(u); vIt != rag.adjacencyEnd(u); ++vIt) {
                auto v = (*vIt).node();
                auto edgeId = (*vIt).edge();
                // check if this is a z-edge and if it is, remove it from the adjacency
                if(edgeId > edgeOffset)
                    deleteEdges.emplace_back(edgeId);
                else if(std::find(defectNodesZ.begin(), defectNodesZ.end(), v) == defectNodesZ.end())
                    ignoreEdges.emplace_back(edgeId);
            }

            // don't need skip edges for first and last slice
            if(z == 0 || z == shape[0] - 1)
                continue;

            // continue if the lower slice is completely defected
            if(lowerIsCompletelyDefected)
                continue;

            // reset the mask
            if(!coordsUPrev.empty()) {
                for(const auto & coord : coordsUPrev)
                    mask(coord.asStdArray()) = false;
            }
            //tools::forEachCoordinate(sliceShape2, [&](const Coord2 & coord){
            //    NIFTY_CHECK_OP(mask(coord.asStdArray()),==,false,"Not properly resetted!");
            //});

            const auto & coordsU = coordsToNodes[u];
            coordsUPrev = coordsU;

            for(const auto & coord : coordsU)
                mask(coord.asStdArray()) = true;

            // find the lower nodes overlapping with this defect for skip edges
            std::vector<NodeType> nodesDn;
            tools::uniquesWithCoordinates<2>(segDn, coordsU, nodesDn);

            // we discard defected nodes in the lower slice (if present), because they
            // were already taken care of in a previous iteration
            // erase - remove idiom
            const auto & defectNodesDn = defectNodes[z-1];
            auto isDefected = [&](NodeType nodeId)
            {
                return (std::find(defectNodesDn.begin(), defectNodesDn.end(), nodeId) != defectNodesDn.end());
            };
            nodesDn.erase( std::remove_if(nodesDn.begin(), nodesDn.end(), isDefected), nodesDn.end() );

            // we only continue doing stuff, if there are nodesDn left
            if(!nodesDn.empty()){

                std::map<NodeType,std::vector<Coord2>> coordsToNodesDn;
                tools::valuesToCoordinatesWithCoordinates<2,NodeType>(segDn, coordsU, coordsToNodesDn);

                std::vector<std::pair<NodeType,NodeType>> skipEdgesU;
                std::vector<size_t> skipRangesU;

                getSkipEdgesForNode(
                    labelsProxy,
                    z+1,
                    z-1,
                    defectNodes,
                    segDn,
                    nodesDn,
                    coordsToNodesDn,
                    mask,
                    upperSegMap,
                    skipEdgesU,
                    skipRangesU);

                // extend the vectors
                skipEdges.reserve(skipEdges.size() + skipEdgesU.size());
                skipEdges.insert(skipEdges.end(), skipEdgesU.begin(), skipEdgesU.end());
                skipRanges.reserve(skipRanges.size() + skipRangesU.size());
                skipRanges.insert(skipRanges.end(), skipRangesU.begin(), skipRangesU.end());
            }
        }
    }
    */

} // end namespace graph
} // end namespace nifty
