#pragma once

#include "nifty/marray/marray.hxx"
#include "nifty/tools/for_each_coordinate.hxx"


namespace nifty{
namespace ground_truth{

    template<class T_SEG>
    void seg2dToLiftedEdges(
        const marray::View<T_SEG>             & segmentation,
        std::vector<std::array<int32_t, 2> >  & edges,
        marray::View<uint8_t> & liftedEdgesState
    ){
        for(int32_t s0=0; s0<segmentation.shape(0); ++s0)
        for(int32_t s1=0; s1<segmentation.shape(1); ++s1){

            // u node label
            const auto lu = segmentation(s0, s1);

            for(int32_t e=0; e<edges.size(); ++e){
                const auto & edge = edges[e];
                const auto  ss0   = s0 + edge[0];
                const auto  ss1   = s1 + edge[1];

                // inside edge
                if( ss0 >= int32_t(0) && int32_t(ss0<segmentation.shape(0)) &&
                    ss1 >= int32_t(0) && int32_t(ss1<segmentation.shape(1))){

                    // v node label
                    const auto lv = segmentation(ss0, ss1);

                    liftedEdgesState(s0, s1, e) = uint8_t(lu != lv);
                }
                // outside 
                else{
                    liftedEdgesState(s0, s1, e) = 0;
                }
            }
        }
    }


    template<class T_SEG>
    void seg3dToLiftedEdges(
        const marray::View<T_SEG>             & segmentation,
        std::vector<std::array<int32_t, 3> >  & edges,
        const int32_t z,
        marray::View<uint8_t> & liftedEdgesState
    ){
        for(int32_t s0=0; s0<segmentation.shape(0); ++s0)
        for(int32_t s1=0; s1<segmentation.shape(1); ++s1)
        {

            // u node label
            const auto lu = segmentation(s0, s1, z);

            for(int32_t e=0; e<edges.size(); ++e){
                const auto & edge = edges[e];

                const auto  ss0   = s0 + edge[0];
                const auto  ss1   = s1 + edge[1];
                const auto  ss2   = s1 + edge[2];

                // inside edge
                if( ss0 >= int32_t(0) && int32_t(ss0<segmentation.shape(0)) &&
                    ss1 >= int32_t(0) && int32_t(ss1<segmentation.shape(1))){

                    // v node label
                    const auto lv = segmentation(ss0, ss1, ss2);

                    liftedEdgesState(s0, s1, e) = uint8_t(lu != lv);
                }
                // outside 
                else{
                    liftedEdgesState(s0, s1, e) = 0;
                }
            }
        }
    }

}   
}