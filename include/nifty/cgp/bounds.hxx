#pragma once

#include <vector>

#include "nifty/marray/marray.hxx"
#include "nifty/cgp/topological_grid.hxx"
#include "nifty/array/arithmetic_array.hxx"
#include "nifty/container/boost_flat_set.hxx"


namespace nifty{
namespace cgp{


    template<size_t DIM, size_t CELL_TYPE>
    class CellBounds;

    template<>
    class CellBounds<2, 0>{
    public:
        CellBounds(const uint32_t a = 0, const uint32_t b = 0,
                   const uint32_t c = 0, const uint32_t d = 0):
        data_{uint32_t(0), uint32_t(0), uint32_t(0),uint32_t(0) }{

            nifty::container::BoostFlatSet<uint32_t> s;
            if(a){
                s.insert(a);
            }
            if(b){
                s.insert(b);
            }
            if(c){
                s.insert(c);
            }
            if(d){
                s.insert(d);
            }
            std::copy(s.begin(), s.end(), data_);              
        }
        uint32_t size()const{
            return data_[3] == 0 ? 3:4;
        }
        const uint32_t & operator[](const unsigned int i)const{
            return data_[i];
        }
    private:
        uint32_t data_[4];
    };


    template<>
    class CellBounds<2, 1>{
    public:
        CellBounds(const uint32_t a = 0, const uint32_t b = 0){
           data_[0] = std::min(a, b);
           data_[1] = std::max(a, b);
        }
        uint32_t size()const{
            return 2;
        }
        const uint32_t &  operator[](const unsigned int i)const{
            return data_[i];
        }
    private:
        uint32_t data_[2];
    };




    template<size_t DIM>
    class Bounds;

    template<>
    class Bounds<2>{
    public:

        typedef array::StaticArray<uint32_t, 2> CoordinateType;
        typedef array::StaticArray<int64_t, 2>  SignedCoordinateType;

        typedef TopologicalGrid<2> TopologicalGridType;

        Bounds(const TopologicalGridType & tGrid);

        template<size_t CELL_TYPE>
        const std::vector< CellBounds<2, CELL_TYPE> > &
        bounds()const{
            return std::get<CELL_TYPE>(bounds_);
        }

    private:
        
        std::tuple<
            std::vector< CellBounds<2, 0> >,
            std::vector< CellBounds<2, 1> >
        > bounds_;
    };



    inline Bounds<2>::Bounds(const TopologicalGridType & tGrid){

        std::get<0>(bounds_).resize(tGrid.numberOfCells()[0]);
        std::get<1>(bounds_).resize(tGrid.numberOfCells()[1]);

        nifty::tools::forEachCoordinate(tGrid.topologicalGridShape(), [&](
            const SignedCoordinateType & tCoord
        ){
      

            auto even0 = tCoord[0] % 2 == 0 ;
            auto even1 = tCoord[1] % 2 == 0 ;

            //std::cout<<"tCoord "<<tCoord[0]<<" "<<tCoord[1]<<"\n";

            if(!even0 && !even1){
                const auto cell0Label = tGrid(tCoord);
                if(cell0Label){



                    std::get<0>(bounds_)[cell0Label-1] = CellBounds<2,0>(
                       tGrid(tCoord[0]-1, tCoord[1]),
                       tGrid(tCoord[0]+1, tCoord[1]),
                       tGrid(tCoord[0]  , tCoord[1]+1),
                       tGrid(tCoord[0]  , tCoord[1]-1)
                    );
                }
            }
            else if(!(even0 && even1)){
                const auto cell1Label = tGrid(tCoord);
                if(cell1Label){

                    const auto tCoord_ = CoordinateType({tCoord[0],tCoord[1]});
                    uint32_t cell2LabelA, cell2LabelB;
                    if(!even0){
                        cell2LabelA=tGrid( tCoord[0]-1, tCoord[1] );
                        cell2LabelB=tGrid( tCoord[0]+1, tCoord[1] );
                    }
                    else{
                        cell2LabelA=tGrid( tCoord[0], tCoord[1]-1);
                        cell2LabelB=tGrid( tCoord[0], tCoord[1]+1);
                    } 
                    //std::cout<<"cell1Label "<<cell1Label<<"\n";
                    //std::cout<<"size "<<std::get<1>(bounds_).size()<<"\n";

                    std::get<1>(bounds_)[cell1Label-1] = CellBounds<2,1>(cell2LabelA,cell2LabelB);
                }
            }
        });
    }










} // namespace nifty::cgp
} // namespace nifty


