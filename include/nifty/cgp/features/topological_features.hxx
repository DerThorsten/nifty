#pragma once

#include <vector>

#include "nifty/math/math.hxx"
#include "nifty/histogram/histogram.hxx"
#include "nifty/cgp/geometry.hxx"
#include "nifty/cgp/bounds.hxx"
#include "nifty/marray/marray.hxx"
#include "nifty/filters/gaussian_curvature.hxx"
#include "nifty/features/accumulated_features.hxx"


namespace nifty{
namespace cgp{


    class Cell1BasicTopologicalFeatures{
    public:
        Cell1BasicTopologicalFeatures(){
        }

        size_t numberOfFeatures()const{
            return 6;
        }
            

        template<class T>
        void operator()(
            const CellBoundsVector<2,0>     & cell0BoundsVector,
            const CellBoundsVector<2,1>     & cell1BoundsVector,
            const CellBoundedByVector<2,1>  & cell1BoundedByVector,
            const CellBoundedByVector<2,2>  & cell2BoundedByVector,
            nifty::marray::View<T> & features
        )const{  
            using namespace nifty::math;
            for(auto cell1Index=0; cell1Index<cell1BoundsVector.size(); ++cell1Index){

                const auto & cell1Bounds = cell1BoundsVector[cell1Index];
                const auto cell2UIndex = cell1Bounds[0]-1;
                const auto cell2VIndex = cell1Bounds[1]-1;


                auto fIndex = 0;

                auto cell2ToCell1Features = [&](const float uVal, const float vVal){
                    features(cell1Index, fIndex++) = std::min(uVal, vVal);
                    features(cell1Index, fIndex++) = std::max(uVal, vVal);
                    features(cell1Index, fIndex++) = uVal + vVal;
                    features(cell1Index, fIndex++) = std::abs(uVal-vVal);
                };

                // number of 0-cells bounding this 1-cell
                // aka how many junctions has this edge (0,1 or 2)
                const auto boundedBySize = cell1BoundedByVector[cell1Index].size();
                auto nCells0 = 0 ;
                for(auto i=0; i<boundedBySize; ++i){
                    const auto cell0Index = cell1BoundedByVector[cell1Index][i];
                    nCells0 += cell0BoundsVector[cell0Index].size();
                }
                features(cell1Index, fIndex++) = boundedBySize;
                features(cell1Index, fIndex++) = nCells0;



                // number of cells 1 which are bounding cell 2
                // aka the degree of a nodes in ordinary graph
                cell2ToCell1Features(cell2BoundedByVector[cell2UIndex].size(), 
                                     cell2BoundedByVector[cell2VIndex].size());




            }
        
        }
    private:
        std::vector<size_t> dists_;
    };



}
}