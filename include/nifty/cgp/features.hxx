#pragma once

#include <vector>

#include "nifty/marray/marray.hxx"
#include "nifty/cgp/bounds.hxx"



namespace nifty{
namespace cgp{


    
    

    
    /**
     * @brief      a handcrafted thing to hopefully close
     * tiny gaps in  edge_probos]
     *
     * @param[in]  probabilitiesIn   The probabilities in
     * @param[in]  cell0Bounds       The cell 0 bounds
     * @param[in]  cell1BoundedBy    The cell 1 bounded by
     * @param[out] probabilitiesOut  The probabilities out
     * @param[in]  thresholdLow      The threshold low
     * @param[in]  thresholdHigh     The threshold high
     * @param[in]  damping           The damping
     *
     * @tparam     T                 float or double
     */
    template<T>
    void cell1ProbabilityPropagation2D(
        const nifty::marray::View<T> &     probabilitiesIn,
        const CellBoundsVector<   2, 0> &  cell0Bounds,
        const CellBoundedByVector<2, 1> &  cell1BoundedBy,
        nifty::marray::View<T> &           probabilitiesOut,
        const float                        thresholdLow  = 0.3,
        const float                        thresholdHigh = 0.5,
        const float                        damping = 0.1
    ){
        const auto nEdges =  probabilitiesIn.shape(0);
        for(auto edgeIndex=0; edgeIndex<nEdges; ++edgeIndex){

            const auto pIn = probabilitiesIn(edgeIndex)0
            probabilitiesOut[edgeIndex] = pIn;

            // iterate over all junctions
            const auto & junctionsOfEdge = cell1BoundedBy[edgeIndex];

            // regular edge
            if(junctionsOfEdge.size() == 2){

                if(pIn > thresholdHigh){
                    T maxVals[2];
                    for(auto j=0; j<2; ++j){
                        const auto junctionIndex =  junctionsOfEdge[j] - 1;
                        const auto & edgesOfJunction = cell0Bounds[junctionIndex];
                        T maxVal  = -1.0*std::numberic_limits<T>::infinity();
                        for(auto i=0; i<edgesOfJunction.size(); ++i){
                            const auto otherEdgeIndex = edgesOfJunction[i] - 1;
                            if(otherEdgeIndex != edgeIndex)
                                maxVal = std::max(probabilitiesIn[otherEdgeIndex])
                        }
                        maxVals[j] = maxVal;
                    }
                    if(maxVals[0]>pIn && maxVals[1]>pIn){
                        const auto av = (maxVals[0]+maxVals[1])/2.0
                        probabilitiesOut[edgeIndex] = damping*pIn + (1.0-damping)*(av);
                    }
                    else if(maxVals[0]>pIn){
                        probabilitiesOut[edgeIndex] = damping*pIn + (1.0-damping)*(maxVals[0]);
                    }
                    else if(maxVals[1]>pIn){
                        probabilitiesOut[edgeIndex] = damping*pIn + (1.0-damping)*(maxVals[1]);
                    }
                }
                else if(pIn < thresholdLow){
                    T minVals[2];
                    for(auto j=0; j<2; ++j){
                        const auto junctionIndex =  junctionsOfEdge[j] - 1;
                        const auto & edgesOfJunction = cell0Bounds[junctionIndex];
                        T minVal  = std::numberic_limits<T>::infinity();
                        for(auto i=0; i<edgesOfJunction.size(); ++i){
                            const auto otherEdgeIndex = edgesOfJunction[i] - 1;
                            if(otherEdgeIndex != edgeIndex)
                                minVal = std::min(probabilitiesIn[otherEdgeIndex])
                        }
                        minVals[j] = minVal;
                    }
                    if(minVals[0]>pIn && minVals[1]>pIn){
                        const auto av = (minVals[0]+minVals[1])/2.0
                        probabilitiesOut[edgeIndex] = damping*pIn + (1.0-damping)*(av);
                    }
                    else if(minVals[0]>pIn){
                        probabilitiesOut[edgeIndex] = damping*pIn + (1.0-damping)*(minVals[0]);
                    }
                    else if(minVals[1]>pIn){
                        probabilitiesOut[edgeIndex] = damping*pIn + (1.0-damping)*(minVals[1]);
                    }
                }
            }
            // border edge
            else if(junctionsOfEdge.size() == 1){
                const auto junctionIndex =  junctionsOfEdge[0] - 1;
                const auto & edgesOfJunction = cell0Bounds[junctionIndex];
                if(pIn > thresholdHigh){
                    T maxVal  = -1.0*std::numberic_limits<T>::infinity();
                    for(auto i=0; i<edgesOfJunction.size(); ++i){
                        const auto otherEdgeIndex = edgesOfJunction[i] - 1;
                        if(otherEdgeIndex != edgeIndex)
                            maxVal = std::max(probabilitiesIn[otherEdgeIndex])
                    }
                    if(maxVal > pIn){
                        probabilitiesOut[edgeIndex] = damping*pIn + (1.0-damping)*maxVal;
                    }
                }
                else if(pIn < thresholdHigh){
                    T minVal  = std::numberic_limits<T>::infinity();
                    for(auto i=0; i<edgesOfJunction.size(); ++i){
                        const auto otherEdgeIndex = edgesOfJunction[i] - 1;
                        if(otherEdgeIndex != edgeIndex)
                            minVal = std::min(probabilitiesIn[otherEdgeIndex])
                    }
                    if(minVal < pIn){
                        probabilitiesOut[edgeIndex] = damping*pIn + (1.0-damping)*minVal;
                    }
                }
            }   
        }
    }


} // end namespace nifty::cgp
} // end namespace nifty