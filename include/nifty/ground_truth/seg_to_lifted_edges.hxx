#pragma once

#include "nifty/tools/for_each_coordinate.hxx"


namespace nifty{
namespace ground_truth{

    template<class T_SEG, class OUT>
    void seg2dToLiftedEdges(
        const T_SEG & segmentation,
        std::vector<std::array<int32_t, 2> >  & edges,
        OUT & liftedEdgesState
    ){
        for(int32_t s0=0; s0<segmentation.shape()[0]; ++s0)
        for(int32_t s1=0; s1<segmentation.shape()[1]; ++s1){

            // u node label
            const auto lu = segmentation(s0, s1);

            for(int32_t e=0; e<edges.size(); ++e){
                const auto & edge = edges[e];
                const auto  ss0   = s0 + edge[0];
                const auto  ss1   = s1 + edge[1];

                // inside edge
                if( ss0 >= int32_t(0) && int32_t(ss0<segmentation.shape()[0]) &&
                    ss1 >= int32_t(0) && int32_t(ss1<segmentation.shape()[1])){

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


    template<class T_SEG, class OUT>
    void seg3dToLiftedEdges(
        const T_SEG & segmentation,
        std::vector<std::array<int32_t, 3> > & edges,
        const int32_t z,
        OUT & liftedEdgesState
    ){
        for(int32_t s0=0; s0<segmentation.shape()[0]; ++s0)
        for(int32_t s1=0; s1<segmentation.shape()[1]; ++s1)
        {

            // u node label
            const auto lu = segmentation(s0, s1, z);

            for(int32_t e=0; e<edges.size(); ++e){
                const auto & edge = edges[e];

                const auto  ss0   = s0 + edge[0];
                const auto  ss1   = s1 + edge[1];
                const auto  ss2   = s1 + edge[2];

                // inside edge
                if( ss0 >= int32_t(0) && int32_t(ss0<segmentation.shape()[0]) &&
                    ss1 >= int32_t(0) && int32_t(ss1<segmentation.shape()[1])){

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



    template<class T_SEG, class OUT>
    void seg3dToCremiZ5Edges(
        const T_SEG & segmentation,
        std::vector<std::array<int32_t, 4> >  & edges,
        OUT & liftedEdgesState
    ){
        NIFTY_CHECK_OP(segmentation.shape()[2], == , 5, "");

        for(int32_t s0=0; s0<segmentation.shape()[0]; ++s0)
        for(int32_t s1=0; s1<segmentation.shape()[1]; ++s1)
        {

            for(int32_t e=0; e<edges.size(); ++e){
                const auto & edge = edges[e];

                const auto start_z = 2  + edge[0];
                const auto ss0     = s0 + edge[1];
                const auto ss1     = s1 + edge[2];
                const auto ss2     = start_z + edge[3];

                const auto lu = segmentation(s0, s1, start_z);

                // inside edge
                if( ss0 >= int32_t(0) && int32_t(ss0<segmentation.shape()[0]) &&
                    ss1 >= int32_t(0) && int32_t(ss1<segmentation.shape()[1])){

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



    template<class T_SEG, class T_DT, class T_OUT>
    void thinSegFilter(
        const T_SEG & segmentation,
        const T_DT & dt,
        T_OUT & out,
        const float sigma,
        int radius = 0
    ){
        if(radius<=0){
            radius = int(sigma*3.5+0.5);
        }
        const auto diam = 2*radius + 1;
        const auto k_size = diam*diam;
        std::vector<float> kernel(k_size);


        auto i=0;
        auto s=0.0;
        const auto sigmaSquared2 = sigma*sigma*2.0;
        for(int x0=0; x0<diam; ++x0)
        for(int x1=0; x1<diam; ++x1){

            const auto dSquared  = std::pow(x0 - radius,2) +
                                   std::pow(x1 - radius,2);
            const auto v =  std::exp(-1.0*dSquared/sigmaSquared2);
            kernel[i] = v;
            s += v;
            ++i;
        }
        for(auto & v : kernel){
            v /= s;
        }

        for(int x0=radius; x0<dt.shape()[0] - (radius); ++x0)
        for(int x1=radius; x1<dt.shape()[1] - (radius); ++x1){
            // std::cout<<"x0 "<<x0<<" x1 "<<x1<<"\n";

            auto s=0.0;
            auto w=0.0;

            const auto lu = segmentation(x0,x1);
            auto ki=0;
            for(int o0=-1*radius; o0<radius+1; ++o0){
                for(int o1=-1*radius; o1<radius+1; ++o1){
                    const int xx0 = x0 +o0;
                    const int xx1 = x1 +o1;
                    //std::cout<<"xx0 "<<xx0<<" xx1 "<<xx1<<"\n";
                    const auto lv = segmentation(xx0,xx1);
                    if(lu==lv && dt(xx0,xx1)>0){
                        // value of the kernel
                        const auto k = kernel[ki];
                        s += k*float(dt(xx0,xx1));
                        w += k;
                    }
                    ++ki;
                }
            }
            NIFTY_CHECK_OP(ki,==,kernel.size(),"");
            //std::cout<<"OS[0]"<<out.shape(0)<<" OS[1]"<<out.shape(1)<<"\n";
            out(x0, x1) = s / w;
        }


    }


    // template<class T_SEG, class T_DT, class T_LE, class T_W>
    // void seg3dToCremiZ5Edges(
    //     const T_SEG             & segmentation,
    //     const T_DT              & dt2d,
    //     std::vector<std::array<int32_t, 4> >  & edges,
    //     T_LE &               liftedEdgesState,
    //     T_W &                 weightMap
    // ){
    //     NIFTY_CHECK_OP(segmentation.shape(2), == , 5, "");
    //     NIFTY_CHECK_OP(dt2d.shape(2), == , 5, "");



    //     for(int32_t s0=0; s0<segmentation.shape(0); ++s0)
    //     for(int32_t s1=0; s1<segmentation.shape(1); ++s1)
    //     {


    //         for(int32_t e=0; e<edges.size(); ++e){
    //             const auto & edge = edges[e];

    //             const auto start_z = 2  + edge[0];
    //             const auto ss0     = s0 + edge[1];
    //             const auto ss1     = s1 + edge[2];
    //             const auto ss2     = start_z + edge[3];


    //             const auto lu = segmentation(s0, s1, start_z);

    //             // inside edge
    //             if( ss0 >= int32_t(0) && int32_t(ss0<segmentation.shape(0)) &&
    //                 ss1 >= int32_t(0) && int32_t(ss1<segmentation.shape(1))){

    //                 // v node label
    //                 const auto lv = segmentation(ss1, ss2, start_z);

    //                 liftedEdgesState(s0, s1, e) = uint8_t(lu != lv);
    //             }
    //             // outside
    //             else{
    //                 liftedEdgesState(s0, s1, e) = 0;
    //             }
    //         }
    //     }
    // }



}
}
