#pragma once
#ifndef NIFTY_TOOLS_FOR_EACH_COORD_HXX
#define NIFTY_TOOLS_FOR_EACH_COORD_HXX

#include <sstream>
#include <chrono>


namespace nifty{
namespace tools{

    
    

    template<class SHAPE_T, class F>
    void forEachCoordinateImpl(
        const std::array<SHAPE_T, 1> & shape,
        F && f,
        bool firstCoordinateMajorOrder = true
    ){
        std::array<int64_t, 1> coord;

        for(coord[0]=0; coord[0]<shape[0]; ++coord[0]){
            f(coord);
        }
    }

    template<class SHAPE_T, class F>
    void forEachCoordinateImpl(
        const std::array<SHAPE_T, 1> & shapeBegin,
        const std::array<SHAPE_T, 1> & shapeEnd,
        F && f,
        bool firstCoordinateMajorOrder = true
    ){
        std::array<int64_t, 1> coord;

        for(coord[0]=shapeBegin[0]; coord[0]<shapeEnd[0]; ++coord[0]){
            f(coord);
        }
    }


    template<class SHAPE_T, class F>
    void forEachCoordinateImpl(
        const std::array<SHAPE_T, 2> & shape,
        F && f,
        bool firstCoordinateMajorOrder = true
    ){

        std::array<int64_t, 2> coord;
        if(firstCoordinateMajorOrder){
            for(coord[0]=0; coord[0]<shape[0]; ++coord[0])
            for(coord[1]=0; coord[1]<shape[1]; ++coord[1]){
                f(coord);
            }
        }
        else{
            for(coord[1]=0; coord[1]<shape[1]; ++coord[1])
            for(coord[0]=0; coord[0]<shape[0]; ++coord[0]){
                f(coord);
            }
        }
    }

    template<class SHAPE_T, class F>
    void forEachCoordinateImpl(
        const std::array<SHAPE_T, 2> & shapeBegin,
        const std::array<SHAPE_T, 2> & shapeEnd,
        F && f,
        bool firstCoordinateMajorOrder = true
    ){

        std::array<int64_t, 2> coord;
        if(firstCoordinateMajorOrder){
            for(coord[0]=shapeBegin[0]; coord[0]<shapeEnd[0]; ++coord[0])
            for(coord[1]=shapeBegin[1]; coord[1]<shapeEnd[1]; ++coord[1]){
                f(coord);
            }
        }
        else{
            for(coord[2]=shapeBegin[1]; coord[2]<shapeEnd[2]; ++coord[2])
            for(coord[1]=shapeBegin[2]; coord[1]<shapeEnd[1]; ++coord[1]){
                f(coord);
            }
        }
    }

    template<class SHAPE_T, class F>
    void forEachCoordinateImpl(
        const std::array<SHAPE_T, 3> & shape,
        F && f,
        bool firstCoordinateMajorOrder = true
    ){

        std::array<int64_t, 3> coord;
        if(firstCoordinateMajorOrder){
            for(coord[0]=0; coord[0]<shape[0]; ++coord[0])
            for(coord[1]=0; coord[1]<shape[1]; ++coord[1])
            for(coord[2]=0; coord[2]<shape[2]; ++coord[2]){
                f(coord);
            }
        }
        else{
            for(coord[2]=0; coord[2]<shape[2]; ++coord[2])
            for(coord[1]=0; coord[1]<shape[1]; ++coord[1])
            for(coord[0]=0; coord[0]<shape[0]; ++coord[0]){
                f(coord);
            }
        }
    }

    template<class SHAPE_T, class F>
    void forEachCoordinateImpl(
        const std::array<SHAPE_T, 3> & shapeBegin,
        const std::array<SHAPE_T, 3> & shapeEnd,
        F && f,
        bool firstCoordinateMajorOrder = true
    ){

        std::array<int64_t, 3> coord;
        if(firstCoordinateMajorOrder){
            for(coord[0]=shapeBegin[0]; coord[0]<shapeEnd[0]; ++coord[0])
            for(coord[1]=shapeBegin[1]; coord[1]<shapeEnd[1]; ++coord[1])
            for(coord[2]=shapeBegin[2]; coord[2]<shapeEnd[2]; ++coord[2]){
                f(coord);
            }
        }
        else{
            for(coord[3]=shapeBegin[0]; coord[3]<shapeEnd[3]; ++coord[3])
            for(coord[2]=shapeBegin[1]; coord[2]<shapeEnd[2]; ++coord[2])
            for(coord[1]=shapeBegin[2]; coord[1]<shapeEnd[1]; ++coord[1]){
                f(coord);
            }
        }
    }


    template<class SHAPE_T, class F>
    void forEachCoordinateImpl(
        const std::array<SHAPE_T, 4> & shape,
        F && f,
        bool firstCoordinateMajorOrder = true
    ){

        std::array<int64_t, 4> coord;
        if(firstCoordinateMajorOrder){
            for(coord[0]=0; coord[0]<shape[0]; ++coord[0])
            for(coord[1]=0; coord[1]<shape[1]; ++coord[1])
            for(coord[2]=0; coord[2]<shape[2]; ++coord[2])
            for(coord[3]=0; coord[3]<shape[3]; ++coord[3]){
                f(coord);
            }
        }
        else{
            for(coord[3]=0; coord[3]<shape[3]; ++coord[3])
            for(coord[2]=0; coord[2]<shape[2]; ++coord[2])
            for(coord[1]=0; coord[1]<shape[1]; ++coord[1])
            for(coord[0]=0; coord[0]<shape[0]; ++coord[0]){
                f(coord);
            }
        }
    }

    template<class SHAPE_T, class F>
    void forEachCoordinateImpl(
        const std::array<SHAPE_T, 4> & shapeBegin,
        const std::array<SHAPE_T, 4> & shapeEnd,
        F && f,
        bool firstCoordinateMajorOrder = true
    ){

        std::array<int64_t, 4> coord;
        if(firstCoordinateMajorOrder){
            for(coord[0]=shapeBegin[0]; coord[0]<shapeEnd[0]; ++coord[0])
            for(coord[1]=shapeBegin[1]; coord[1]<shapeEnd[1]; ++coord[1])
            for(coord[2]=shapeBegin[2]; coord[2]<shapeEnd[2]; ++coord[2])
            for(coord[3]=shapeBegin[3]; coord[3]<shapeEnd[3]; ++coord[3]){
                f(coord);
            }
        }
        else{
            for(coord[3]=shapeBegin[0]; coord[3]<shapeEnd[3]; ++coord[3])
            for(coord[2]=shapeBegin[1]; coord[2]<shapeEnd[2]; ++coord[2])
            for(coord[1]=shapeBegin[2]; coord[1]<shapeEnd[1]; ++coord[1])
            for(coord[0]=shapeBegin[3]; coord[0]<shapeEnd[0]; ++coord[0]){
                f(coord);
            }
        }
    }



    template<class SHAPE_T, size_t DIMENSIONS, class F>
    void forEachCoordinate(
        const std::array<SHAPE_T, DIMENSIONS> & shape,
        F && f,
        bool firstCoordinateMajorOrder = true
    ){
        forEachCoordinateImpl(shape, f, firstCoordinateMajorOrder);
    }
    template<class SHAPE_T, size_t DIMENSIONS, class F>
    void forEachCoordinate(
        const std::array<SHAPE_T, DIMENSIONS> & shapeBegin,
        const std::array<SHAPE_T, DIMENSIONS> & shapeEnd,
        F && f,
        bool firstCoordinateMajorOrder = true
    ){
        forEachCoordinateImpl(shapeBegin, shapeEnd, f, firstCoordinateMajorOrder);
    }

} // end namespace nifty::tools
} // end namespace nifty

#endif /*NIFTY_TOOLS_FOR_EACH_COORD_HXX*/
