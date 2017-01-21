#pragma once
#ifndef NIFTY_GRAPH_RAG_GRID_RAG_FEATURES_STACKED_HXX
#define NIFTY_GRAPH_RAG_GRID_RAG_FEATURES_STACKED_HXX

//#include <iostream>
//#include <fstream>
//#include <chrono>

#include "nifty/graph/rag/grid_rag_stacked_2d.hxx"
#ifdef WITH_HDF5
#include "nifty/graph/rag/grid_rag_stacked_2d_hdf5.hxx"
#endif
#include "nifty/marray/marray.hxx"

#include "nifty/tools/for_each_coordinate.hxx"
#include "nifty/tools/array_tools.hxx"

#ifdef WITH_HDF5

#endif

namespace nifty{
namespace graph{

    template<class LABELS_PROXY, class LABELS, class NODE_MAP>
    void gridRagAccumulateLabels(
        const GridRagStacked2D<LABELS_PROXY> & graph,
        const LABELS & data,                         
        NODE_MAP &  nodeMap,
        const int numberOfThreads = -1
    ){
        
        typedef LABELS_PROXY LabelsProxyType;
        typedef typename LABELS_PROXY::LabelType LabelType;
        typedef typename LabelsProxyType::BlockStorageType LabelsBlockStorage;
        typedef typename tools::BlockStorageSelector<LABELS>::type DataBlockStorage;
        
        typedef array::StaticArray<int64_t, 3> Coord;
        typedef array::StaticArray<int64_t, 2> Coord2;

        const auto & labelsProxy = graph.labelsProxy();
        const auto & shape = labelsProxy.shape();
        
        NIFTY_CHECK_OP(data.shape(0),==,shape[0], "Shape along z does not agree")
        NIFTY_CHECK_OP(data.shape(1),==,shape[1], "Shape along y does not agree")
        NIFTY_CHECK_OP(data.shape(2),==,shape[2], "Shape along x does not agree")
        
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
            
            auto sliceLabels = sliceLabelsFlat3DView.squeezedView();
            auto sliceData = sliceDataFlat3DView.squeezedView();
            
            nifty::tools::forEachCoordinate(sliceShape2,[&](const Coord2 & coord){
                const auto node = sliceLabels( coord.asStdArray() );            
                const auto l    = sliceData( coord.asStdArray() );
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


    template<class LABELS_PROXY, class NODE_TYPE>
    inline void getSkipEdgesForNode(
        const LABELS_PROXY & labels,
        const int64_t zUp,
        const int64_t zDn,
        std::map<size_t,std::vector<NODE_TYPE>> & defectNodes, 
        const marray::View<NODE_TYPE> & segDn,
        const std::vector<NODE_TYPE> & nodesDn,
        const std::pair<array::StaticArray<int64_t, 2>,array::StaticArray<int64_t, 2>> & bb,
        const marray::View<bool> & mask,
        std::vector<std::pair<NODE_TYPE,NODE_TYPE>> & skipEdges,
        std::vector<size_t> & skipRanges
    ){
        typedef NODE_TYPE NodeType;
        typedef array::StaticArray<int64_t, 3> Coord;
        typedef array::StaticArray<int64_t, 2> Coord2;
        
        skipEdges.clear();
        skipRanges.clear();
        
        size_t skipRange = zUp - zDn;
        Coord2 bbBegin = bb.first;
        Coord2 bbEnd   = bb.second;
            
        // read the upper segmentation
        Coord beginUp({zUp, bbBegin[0],bbBegin[1]});
        Coord endUp({zUp+1, bbEnd[0],  bbEnd[1]});
        
        Coord segShape({1L, bbEnd[0] - bbBegin[0], bbEnd[1] - bbBegin[1]});
        marray::Marray<NodeType> segUpArray(segShape.begin(), segShape.end());
        labels.readSubarrayLocked(beginUp, endUp, segUpArray);
        marray::View<NodeType> segUp = segUpArray.squeezedView();
        
        for(auto uDn : nodesDn) {

            auto t0 = std::chrono::steady_clock::now();
            
            std::vector<Coord2> coordsDn;
            tools::where<2>(segDn, uDn, coordsDn);
            
            // find intersecting nodes in upper slice and 
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
                    bb,
                    mask,
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
        
        // read the segmentation
        Coord sliceShape({1L, shape[1], shape[2]});
        marray::Marray<NodeType> segZArray(sliceShape.begin(), sliceShape.end());
        Coord sliceStart({z, 0L, 0L});
        Coord sliceStop({z+1, shape[1], shape[2]});
        labelsProxy.readSubarrayLocked(sliceStart, sliceStop, segZArray);
        marray::View<NodeType> segZ = segZArray.squeezedView();

        const auto & defectNodesZ  = defectNodes[z];

        for(size_t i = 0; i < defectNodesZ.size(); ++i) {
            auto u = defectNodesZ[i];

            //std::cout << "Processing Node " << u << ": " << i << "/" << defectNodesZ.shape(0) << std::endl;
            
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

            auto t0 = std::chrono::steady_clock::now();
            
            // find the coordinate of u and the corresponding bounding box
            std::vector<Coord2> coordsU;
            auto bbU = tools::whereAndBoundingBox<2>(segZ, u, coordsU);
            Coord2 beginU = bbU.first;
            Coord2 endU   = bbU.second;

            Coord beginUDn({z-1, beginU[0], beginU[1]});
            Coord endUDn({z, endU[0], endU[1]});
            Coord bbShape({1L, endU[0] - beginU[0], endU[1] - beginU[1]});
            
            Coord2 maskShape({bbShape[1],bbShape[2]});
            // mask for this node in bounding box (2D!)
            marray::Marray<bool> mask(maskShape.begin(), maskShape.end(), false);
            
            // FIXME! THIS SHOULD NOT WORK!
            //std::cout << mask.dimension() << " " << mask.shape(1) << " " << mask.shape(2) << std::endl;
            std::vector<Coord2> coordsUMask;
            for(auto & coord : coordsU)
                coordsUMask.emplace_back( Coord2({coord[0] - beginU[0], coord[1] - beginU[1]}) );
            
            for(auto & coord : coordsUMask)
                mask(coord.asStdArray()) = true;

            // find the lower nodes for skip edges
            marray::Marray<NodeType> segDnArray(bbShape.begin(), bbShape.end());
            labelsProxy.readSubarrayLocked(beginUDn, endUDn, segDnArray);
            marray::View<NodeType> segDn = segDnArray.squeezedView();
            
            std::vector<NodeType> nodesDn;
            tools::uniquesWithCoordinates<2>(segDn, coordsUMask, nodesDn);
            
            // we discard defected nodes in the lower slice (if present), because they
            // were already taken care of in a previous iteration
            // erase - remove idiom 
            // FIXME seems this does not work properly yet
            const auto & defectNodesDn = defectNodes[z-1];
            auto isDefected = [&](NodeType nodeId)
            {
                return (std::find(defectNodesDn.begin(), defectNodesDn.end(), nodeId) != defectNodesDn.end());
            };
            //std::cout << "Nodes before erasing: " << nodesDn.size() << std::endl;
            nodesDn.erase( std::remove_if(nodesDn.begin(), nodesDn.end(), isDefected), nodesDn.end() );
            //std::cout << "Erased nodes, new nodes: " << nodesDn.size() << std::endl;
            
            //for(auto nodeDn : nodesDn)
            //    debug_out << std::to_string(nodeDn) << std::endl;

            // we only continue doing stuff, if there are nodesDn left
            if(!nodesDn.empty()){
                std::vector<std::pair<NodeType,NodeType>> skipEdgesU;
                std::vector<size_t> skipRangesU;
                
                getSkipEdgesForNode(
                    labelsProxy,
                    z+1,
                    z-1,
                    defectNodes,
                    segDn,
                    nodesDn,
                    bbU,
                    mask,
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

} // end namespace graph
} // end namespace nifty


#endif /* NIFTY_GRAPH_RAG_GRID_RAG_FEATURES_STACKED_HXX */
