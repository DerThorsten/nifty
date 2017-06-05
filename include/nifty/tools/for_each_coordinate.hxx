#pragma once

#include <sstream>
#include <chrono>
#include <array>

#include "nifty/parallel/threadpool.hxx"
#include "nifty/array/arithmetic_array.hxx"

namespace nifty{
namespace tools{

    
    

    template<class SHAPE_T, class F>
    void forEachCoordinateImpl(
        const array::StaticArray<SHAPE_T, 1> & shape,
        F && f,
        bool firstCoordinateMajorOrder = true
    ){
        array::StaticArray<int64_t, 1> coord;
        for(coord[0]=0; coord[0]<shape[0]; ++coord[0]){
            f(coord);
        }
    }

    template<class SHAPE_T, class F>
    void forEachCoordinateImpl(
        const array::StaticArray<SHAPE_T, 1> & shapeBegin,
        const array::StaticArray<SHAPE_T, 1> & shapeEnd,
        F && f,
        bool firstCoordinateMajorOrder = true
    ){
        array::StaticArray<int64_t, 1> coord;
        for(coord[0]=shapeBegin[0]; coord[0]<shapeEnd[0]; ++coord[0]){
            f(coord);
        }
    }

    template<class SHAPE_T, class F>
    void forEachCoordinateImpl(
        const array::StaticArray<SHAPE_T, 2> & shape,
        F && f,
        bool firstCoordinateMajorOrder = true
    ){
        array::StaticArray<int64_t, 2> coord;
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
        const array::StaticArray<SHAPE_T, 2> & shapeBegin,
        const array::StaticArray<SHAPE_T, 2> & shapeEnd,
        F && f,
        bool firstCoordinateMajorOrder = true
    ){
        array::StaticArray<int64_t, 2> coord;
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
        const array::StaticArray<SHAPE_T, 3> & shape,
        F && f,
        bool firstCoordinateMajorOrder = true
    ){
        array::StaticArray<int64_t, 3> coord;
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
        const array::StaticArray<SHAPE_T, 3> & shapeBegin,
        const array::StaticArray<SHAPE_T, 3> & shapeEnd,
        F && f,
        bool firstCoordinateMajorOrder = true
    ){

        array::StaticArray<int64_t, 3> coord;
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
        const array::StaticArray<SHAPE_T, 4> & shape,
        F && f,
        bool firstCoordinateMajorOrder = true
    ){

        array::StaticArray<int64_t, 4> coord;
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
        const array::StaticArray<SHAPE_T, 4> & shapeBegin,
        const array::StaticArray<SHAPE_T, 4> & shapeEnd,
        F && f,
        bool firstCoordinateMajorOrder = true
    ){
        array::StaticArray<int64_t, 4> coord;
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
        const array::StaticArray<SHAPE_T, DIMENSIONS> & shape,
        F && f,
        bool firstCoordinateMajorOrder = true
    ){
        forEachCoordinateImpl(shape, f, firstCoordinateMajorOrder);
    }

    template<class SHAPE_T, size_t DIMENSIONS, class F>
    void forEachCoordinate(
        const array::StaticArray<SHAPE_T, DIMENSIONS> & shapeBegin,
        const array::StaticArray<SHAPE_T, DIMENSIONS> & shapeEnd,
        F && f,
        bool firstCoordinateMajorOrder = true
    ){
        forEachCoordinateImpl(shapeBegin, shapeEnd, f, firstCoordinateMajorOrder);
    }




    template<class SHAPE_T, size_t DIM, class F>
    void parallelForEachCoordinate(
        nifty::parallel::ThreadPool & threadpool,                   
        const array::StaticArray<SHAPE_T, DIM> & shape,
        F && f,
        bool firstCoordinateMajorOrder = true
    ){
        static_assert(DIM<=5,"currently dimension must be smaller or equal to 5");
        typedef array::StaticArray<int64_t, DIM> Coord;
        if(firstCoordinateMajorOrder){
            const auto nItems = shape[0];
            parallel_foreach(threadpool,nItems,[&](const int tid, const int parallelCord){
                Coord currentCord;
                currentCord[0] = parallelCord;
                if(DIM == 1){
                    f(tid, currentCord);
                }
                else if(DIM == 2){
                    for(currentCord[1]=0; currentCord[1]<shape[1]; ++currentCord[1]){
                        f(tid, currentCord);
                    }
                }
                else if(DIM == 3){
                    for(currentCord[1]=0; currentCord[1]<shape[1]; ++currentCord[1])
                    for(currentCord[2]=0; currentCord[2]<shape[2]; ++currentCord[2]){
                        f(tid, currentCord);
                    }
                }
                else if(DIM == 4){
                    for(currentCord[1]=0; currentCord[1]<shape[1]; ++currentCord[1])
                    for(currentCord[2]=0; currentCord[2]<shape[2]; ++currentCord[2])
                    for(currentCord[3]=0; currentCord[3]<shape[3]; ++currentCord[3]){
                        f(tid, currentCord);
                    }
                }
                else if(DIM == 5){
                    for(currentCord[1]=0; currentCord[1]<shape[1]; ++currentCord[1])
                    for(currentCord[2]=0; currentCord[2]<shape[2]; ++currentCord[2])
                    for(currentCord[3]=0; currentCord[3]<shape[3]; ++currentCord[3])
                    for(currentCord[4]=0; currentCord[4]<shape[4]; ++currentCord[4]){
                        f(tid, currentCord);
                    }
                }
            });
        }
        else{
            const auto nItems = shape[DIM-1];
            parallel_foreach(threadpool,nItems,[&](const int tid, const int parallelCord){
                Coord currentCord;
                currentCord[DIM-1] = parallelCord;
                if(DIM == 1){
                    f(tid, currentCord);
                }
                else if(DIM ==2){
                    for(currentCord[0]=0; currentCord[0]<shape[0]; ++currentCord[0]){
                        f(tid, currentCord);
                    }
                }
                else if(DIM ==3){
                    for(currentCord[1]=0; currentCord[1]<shape[1]; ++currentCord[1])
                    for(currentCord[0]=0; currentCord[0]<shape[0]; ++currentCord[0]){
                        f(tid, currentCord);
                    }
                }
                else if(DIM == 4){
                    for(currentCord[2]=0; currentCord[2]<shape[2]; ++currentCord[2])
                    for(currentCord[1]=0; currentCord[1]<shape[1]; ++currentCord[1])
                    for(currentCord[0]=0; currentCord[0]<shape[0]; ++currentCord[0]){
                        f(tid, currentCord);
                    }
                }
                else if(DIM == 5){
                    for(currentCord[3]=0; currentCord[3]<shape[3]; ++currentCord[3])
                    for(currentCord[2]=0; currentCord[2]<shape[2]; ++currentCord[2])
                    for(currentCord[1]=0; currentCord[1]<shape[1]; ++currentCord[1])
                    for(currentCord[0]=0; currentCord[0]<shape[0]; ++currentCord[0]){
                        f(tid, currentCord);
                    }
                }
            });
        }
    }

} // end namespace nifty::tools
} // end namespace nifty

