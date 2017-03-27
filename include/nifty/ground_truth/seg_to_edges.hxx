#pragma once

#include "nifty/marray/marray.hxx"
#include "nifty/tools/for_each_coordinate.hxx"
#include "nifty/tools/timer.hxx"
#include <cmath>

namespace nifty{
namespace ground_truth{



    template<   
        class T, 
        class U
    >
    void segToEdges2D(
        const marray::View<T>   & segmentation,
        marray::View<U>   & edges
    ){
        edges = U(0);
        typedef array::StaticArray<int64_t, 2> CoordType;



        CoordType shape;
        for(size_t i=0; i<2; ++i){
            shape[i] = segmentation.shape(i);
        }   


        tools::forEachCoordinate(shape,
        [&](const CoordType & coord){
            const auto lu = segmentation(coord[0],coord[1]);
            if(coord[0]-1 >=0 &&  segmentation(coord[0]-1,coord[1])!=lu ){
                edges(coord[0],coord[1]) = 1;
            }
            else if(coord[0]+1 < shape[0] &&  segmentation(coord[0]+1,coord[1])!=lu ){
                edges(coord[0],coord[1]) = 1;
            }
            else if(coord[1]-1 >=0 &&  segmentation(coord[0],coord[1]-1)!=lu ){
                edges(coord[0],coord[1]) = 1;
            }
            else if(coord[1]+1 < shape[1] &&  segmentation(coord[0],coord[1]+1)!=lu ){
                edges(coord[0],coord[1]) = 1;
            }
        });


    }




}
}
