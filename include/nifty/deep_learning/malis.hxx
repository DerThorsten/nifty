#pragma once

#include <algorithm>
#include <map>

#include "nifty/marray/marray.hxx"
#include "nifty/tools/for_each_coordinate.hxx"
#include "nifty/ufd/ufd.hxx"


namespace nifty{
namespace deep_learning{

// TODO implement different nhoods
template<unsigned DIM, typename DATA_TYPE, typename LABEL_TYPE>
void compute_malis_gradient(
        const marray::View<DATA_TYPE> & affinities,
        const marray::View<LABEL_TYPE> & groundtruth,
        marray::View<size_t> & positiveGradients,
        marray::View<size_t> & negativeGradients
) { // TODO which dtype / output format for the gradients ?
    
    typedef nifty::array::StaticArray<int64_t,DIM>   Coord;
    typedef nifty::array::StaticArray<int64_t,DIM+1> AffinityCoord;
    typedef LABEL_TYPE LabelType;
    typedef DATA_TYPE DataType;
    
    // check that number of affinity channels matches the dimensions
    NIFTY_CHECK_OP(affinities.shape(DIM),==,DIM,"Number of affinity channels does not match the dimension!");
    NIFTY_CHECK_OP(positiveGradients.shape(DIM),==,DIM,"Number of gradient channels must match !");
    NIFTY_CHECK_OP(negativeGradients.shape(DIM),==,DIM,"Number of gradient channels must match !");
    // check that shapes match
    for(int d = 0; d < DIM; ++d) {
        NIFTY_CHECK_OP(affinities.shape(d),==,groundtruth.shape(d),"Affinity shape does not match gt shape!");
        NIFTY_CHECK_OP(affinities.shape(d),==,positiveGradients.shape(d),"Affinity shape does not match gradients shape!");
        NIFTY_CHECK_OP(affinities.shape(d),==,negativeGradients.shape(d),"Affinity shape does not match gradients shape!");
    }

    const int numberOfNodes = groundtruth.size();
    const int numberOfEdges = affinities.size();
    Coord pixelShape;
    for(int d = 0; d < DIM; ++d)
        pixelShape[d] = groundtruth.shape(d);
    
    //
    // initialize the union find and the vector storing the overlaps 
    // of each node with each groundtruth segment
    //
    
    ufd::Ufd<LabelType> sets(numberOfNodes);
    std::vector<std::map<LabelType,size_t>> overlaps(numberOfNodes);
    int pixelIndex = 0;
    tools::forEachCoordinate(pixelShape, [&](Coord coord) {
        auto gtId = groundtruth(coord.asStdArray());
        if( gtId != 0)  
            overlaps[pixelIndex].insert( std::make_pair(gtId,1) );
        ++pixelIndex;
    });

    //
    // sort the edge indices in decreasing order of affinities
    //

    AffinityCoord affinityShape;
    for(int d = 0; d < DIM+1; ++d)
        affinityShape[d] = affinities.shape(d);
    
    // get a flattened view to the marray
    size_t flatShape[] = {affinities.size()};
    auto flatView = affinities.reshapedView(flatShape, flatShape+1);
    
    // initialize the pqueu as [0,1,2,3,...,numberOfEdges]
    std::vector<size_t> pqueue(numberOfEdges);
    std::iota(pqueue.begin(), pqueue.end(), 0);
    
    // sort pqueue in decreasing order
    std::sort(pqueue.begin(), pqueue.end(),
            [&flatView](const size_t ind1, const size_t ind2){
        return (flatView(ind1) > flatView(ind2));
    });

    //
    // run kruskals, computing gradients for all merges in the spanning tree
    //
    
    size_t edgeIndex, channel;
    LabelType setU, setV;
    size_t nPair = 0;
    size_t nodeU, nodeV;
    AffinityCoord affCoord;
    typename std::map<LabelType,size_t>::iterator itU, itV;

    // iterate over the pqueue
    for(size_t i = 0; i < pqueue.size(); ++i) {
        
        edgeIndex  = pqueue[i];
        
        // translate edge index to coordinate
        affCoord[0] = edgeIndex / affinities.strides(0) ;
        for(int d = 1; d < DIM+1; ++d) {
            affCoord[d] = (edgeIndex % affinities.strides(d-1) ) / affinities.strides(d);
        }
        channel = affCoord[DIM];

        // find the node indices associated with the edge
        nodeU = 0;
        nodeV = 0;
        for(int d = 0; d < DIM; ++d) {
            nodeU += affCoord[d] * groundtruth.strides(d);
            nodeV += affCoord[d] * groundtruth.strides(d);
        }
        
        // we increase the V node by stride of the corresponding coordinate 
        // only if this results in a valid coordinate
        if(affCoord[channel] < pixelShape[channel] - 1) {
            nodeV += groundtruth.strides(channel);
        }
        else {
            continue;
        }
        setU = sets.find( nodeU );
        setV = sets.find( nodeV );

        // only do stuff if the two segments are not merged yet
        if(setU != setV) {
            sets.merge(setU, setV);
            
            // debug output
            //std::cout << "Queue nr. " << i << " edge: " << edgeIndex << std::endl;
            //std::cout << "Weight: "   << flatView(edgeIndex) << std::endl;
            //std::cout << "AffinityCoord:" << affCoord << std::endl;
            //std::cout << "PixelNodes " << nodeU << " , " << nodeV << std::endl;
            //std::cout << "Sets "    << setU << " , " << setV << std::endl;

            // compute the number of pairs merged by this edge
            for (itU = overlaps[setU].begin(); itU != overlaps[setU].end(); ++itU) {
                for (itV = overlaps[setV].begin(); itV != overlaps[setV].end(); ++itV) {

                    // the number of pairs that are joind by this edge are given by the 
                    // number of pix associated with U times pix associated with V
                    nPair = itU->second * itV->second;
                    
                    // for positive gradient 
                    // we add nPairs if we join two nodes in the same gt segment
                    if (itU->first == itV->first) {
                        positiveGradients(affCoord.asStdArray()) += nPair;
                        //debug output
                        //std::cout << "Gt-labels: " << itU->first << " , " << itV->first << std::endl;
                        //std::cout << "Positive gradient added: " << nPair << std::endl;
                    }
                    // for negative gradient,
                    // we add nPairs if we join two nodes in different gt segments
                    else if (itU->first != itV->first) {
                        negativeGradients(affCoord.asStdArray()) += nPair;
                        //debug output
                        //std::cout << "Gt-labels: " << itU->first << " , " << itV->first << std::endl;
                        //std::cout << "Negative gradient added: " << nPair << std::endl;
                    }
                }
            }
            // debug output
            //std::cout << std::endl;
            
            // move the pixel bags of the non-representative to the representative
            if (sets.find(setU) == setV) // make setU the rep to keep and setV the rep to empty
                std::swap(setU,setV);

            itV = overlaps[setV].begin();
            while (itV != overlaps[setV].end()) {
                itU = overlaps[setU].find(itV->first);
                if (itU == overlaps[setU].end()) {
                    overlaps[setU].insert( std::make_pair(itV->first,itV->second) );
                } 
                else {
                    itU->second += itV->second;
                }
                overlaps[setV].erase(itV++);
            }
        }
    }
}






// // TODO implement different nhoods
// template<unsigned DIM, typename DATA_TYPE, typename LABEL_TYPE>
// void compute_malis_gradient(
//         const marray::View<DATA_TYPE> & affinities,
//         const marray::View<LABEL_TYPE> & groundtruth,
//         marray::View<size_t> & positiveGradients,
//         marray::View<size_t> & negativeGradients
// ) { // TODO which dtype / output format for the gradients ?
    
//     typedef nifty::array::StaticArray<int64_t,DIM>   Coord;
//     typedef nifty::array::StaticArray<int64_t,DIM+1> AffinityCoord;
//     typedef LABEL_TYPE LabelType;
//     typedef DATA_TYPE DataType;
    
//     // check that number of affinity channels matches the dimensions
//     NIFTY_CHECK_OP(affinities.shape(DIM),==,DIM,"Number of affinity channels does not match the dimension!");
//     NIFTY_CHECK_OP(positiveGradients.shape(DIM),==,DIM,"Number of gradient channels must match !");
//     NIFTY_CHECK_OP(negativeGradients.shape(DIM),==,DIM,"Number of gradient channels must match !");
//     // check that shapes match
//     for(int d = 0; d < DIM; ++d) {
//         NIFTY_CHECK_OP(affinities.shape(d),==,groundtruth.shape(d),"Affinity shape does not match gt shape!");
//         NIFTY_CHECK_OP(affinities.shape(d),==,positiveGradients.shape(d),"Affinity shape does not match gradients shape!");
//         NIFTY_CHECK_OP(affinities.shape(d),==,negativeGradients.shape(d),"Affinity shape does not match gradients shape!");
//     }

//     const int numberOfNodes = groundtruth.size();
//     const int numberOfEdges = affinities.size();
//     Coord pixelShape;
//     for(int d = 0; d < DIM; ++d)
//         pixelShape[d] = groundtruth.shape(d);
    
//     //
//     // initialize the union find and the vector storing the overlaps 
//     // of each node with each groundtruth segment
//     //
    
//     ufd::Ufd<LabelType> sets(numberOfNodes);
//     std::vector<std::map<LabelType,size_t>> overlaps(numberOfNodes);
//     int pixelIndex = 0;
//     tools::forEachCoordinate(pixelShape, [&](Coord coord) {
//         auto gtId = groundtruth(coord.asStdArray());
//         if( gtId != 0)  
//             overlaps[pixelIndex].insert( std::make_pair(gtId,1) );
//         ++pixelIndex;
//     });

//     //
//     // sort the edge indices in decreasing order of affinities
//     //

//     AffinityCoord affinityShape;
//     for(int d = 0; d < DIM+1; ++d)
//         affinityShape[d] = affinities.shape(d);
    
//     // get a flattened view to the marray
//     size_t flatShape[] = {affinities.size()};
//     auto flatView = affinities.reshapedView(flatShape, flatShape+1);
    
//     // initialize the pqueu as [0,1,2,3,...,numberOfEdges]
//     std::vector<size_t> pqueue(numberOfEdges);
//     std::iota(pqueue.begin(), pqueue.end(), 0);
    
//     // sort pqueue in decreasing order
//     std::sort(pqueue.begin(), pqueue.end(),
//             [&flatView](const size_t ind1, const size_t ind2){
//         return (flatView(ind1) > flatView(ind2));
//     });

//     //
//     // run kruskals, computing gradients for all merges in the spanning tree
//     //
    
//     size_t edgeIndex, channel;
//     LabelType setU, setV;
//     size_t nPair = 0;
//     size_t nodeU, nodeV;
//     AffinityCoord affCoord;
//     typename std::map<LabelType,size_t>::iterator itU, itV;

//     // iterate over the pqueue
//     for(size_t i = 0; i < pqueue.size(); ++i) {
        
//         edgeIndex  = pqueue[i];
        
//         // translate edge index to coordinate
//         affCoord[0] = edgeIndex / affinities.strides(0) ;
//         for(int d = 1; d < DIM+1; ++d) {
//             affCoord[d] = (edgeIndex % affinities.strides(d-1) ) / affinities.strides(d);
//         }
//         channel = affCoord[DIM];

//         // find the node indices associated with the edge
//         nodeU = 0;
//         nodeV = 0;
//         for(int d = 0; d < DIM; ++d) {
//             nodeU += affCoord[d] * groundtruth.strides(d);
//             nodeV += affCoord[d] * groundtruth.strides(d);
//         }
        
//         // we increase the V node by stride of the corresponding coordinate 
//         // only if this results in a valid coordinate
//         if(affCoord[channel] < pixelShape[channel] - 1) {
//             nodeV += groundtruth.strides(channel);
//         }
//         else {
//             continue;
//         }
//         setU = sets.find( nodeU );
//         setV = sets.find( nodeV );

//         // only do stuff if the two segments are not merged yet
//         if(setU != setV) {
//             sets.merge(setU, setV);
            
//             // debug output
//             //std::cout << "Queue nr. " << i << " edge: " << edgeIndex << std::endl;
//             //std::cout << "Weight: "   << flatView(edgeIndex) << std::endl;
//             //std::cout << "AffinityCoord:" << affCoord << std::endl;
//             //std::cout << "PixelNodes " << nodeU << " , " << nodeV << std::endl;
//             //std::cout << "Sets "    << setU << " , " << setV << std::endl;

//             // compute the number of pairs merged by this edge
//             for (itU = overlaps[setU].begin(); itU != overlaps[setU].end(); ++itU) {
//                 for (itV = overlaps[setV].begin(); itV != overlaps[setV].end(); ++itV) {

//                     // the number of pairs that are joind by this edge are given by the 
//                     // number of pix associated with U times pix associated with V
//                     nPair = itU->second * itV->second;
                    
//                     // for positive gradient 
//                     // we add nPairs if we join two nodes in the same gt segment
//                     if (itU->first == itV->first) {
//                         positiveGradients(affCoord.asStdArray()) += nPair;
//                         //debug output
//                         //std::cout << "Gt-labels: " << itU->first << " , " << itV->first << std::endl;
//                         //std::cout << "Positive gradient added: " << nPair << std::endl;
//                     }
//                     // for negative gradient,
//                     // we add nPairs if we join two nodes in different gt segments
//                     else if (itU->first != itV->first) {
//                         negativeGradients(affCoord.asStdArray()) += nPair;
//                         //debug output
//                         //std::cout << "Gt-labels: " << itU->first << " , " << itV->first << std::endl;
//                         //std::cout << "Negative gradient added: " << nPair << std::endl;
//                     }
//                 }
//             }
//             // debug output
//             //std::cout << std::endl;
            
//             // move the pixel bags of the non-representative to the representative
//             if (sets.find(setU) == setV) // make setU the rep to keep and setV the rep to empty
//                 std::swap(setU,setV);

//             itV = overlaps[setV].begin();
//             while (itV != overlaps[setV].end()) {
//                 itU = overlaps[setU].find(itV->first);
//                 if (itU == overlaps[setU].end()) {
//                     overlaps[setU].insert( std::make_pair(itV->first,itV->second) );
//                 } 
//                 else {
//                     itU->second += itV->second;
//                 }
//                 overlaps[setV].erase(itV++);
//             }
//         }
//     }
// }

} // end namespace malis
} // end namespace nifty
