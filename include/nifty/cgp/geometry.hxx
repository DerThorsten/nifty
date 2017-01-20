#pragma once

#include <vector>

#include "nifty/marray/marray.hxx"
#include "nifty/cgp/topological_grid.hxx"
#include "nifty/array/arithmetic_array.hxx"
#include "nifty/container/boost_flat_set.hxx"

namespace nifty{
namespace cgp{



    
    template<size_t DIM, size_t CELL_TYPE>
    class CellGeometry;


    template<size_t DIM>
    class CellGeometry<DIM, 0> : public std::array<
        array::StaticArray<uint32_t, 2>, 
        1 
    >
    {
    public:
        typedef std::integral_constant<size_t, DIM> DimensionType;
        typedef nifty::array::StaticArray<uint32_t, 2> CoordinateType;
        typedef nifty::array::StaticArray<float, DIM> FloatCoordinateType;
        typedef std::array<CoordinateType, 1> BaseType;
        using BaseType::BaseType;

        FloatCoordinateType centerOfMass()const{
            FloatCoordinateType fcoord;
            const auto & icoord = this->operator[](0);
            std::copy(icoord.begin(), icoord.end(), fcoord.begin());
            return fcoord;
        }
    };

    template<size_t DIM, size_t CELL_TYPE>
    class CellGeometry : public std::vector< array::StaticArray<uint32_t, DIM> > {
    public:
        typedef std::integral_constant<size_t, DIM> DimensionType;
        typedef nifty::array::StaticArray<uint32_t, DIM> CoordinateType;
        typedef nifty::array::StaticArray<float, DIM> FloatCoordinateType;
        typedef std::vector<CoordinateType> BaseType;
        using BaseType::BaseType;

        FloatCoordinateType centerOfMass()const{
            CoordinateType s(uint32_t(0));
            for(const auto & c : *this){
                s += c;
            }
            FloatCoordinateType center;
            std::copy(s.begin(), s.end(), center.begin());
            center /= float(this->size());
            return center;
        }
    };


    template<size_t DIM>
    class Geometry;


    template<>
    class Geometry<2>{
    public:

        typedef array::StaticArray<uint32_t, 2> CoordinateType;
        typedef array::StaticArray<int64_t, 2>  SignedCoordinateType;

        typedef TopologicalGrid<2> TopologicalGridType;

        Geometry(const TopologicalGridType & tGrid, const bool=false);

        template<size_t CELL_TYPE>
        const std::vector< CellGeometry<2, CELL_TYPE> > &
        geometry()const{
            return std::get<CELL_TYPE>(geometry_);
        }

    private:
        
        std::tuple<
            std::vector< CellGeometry<2, 0> >,
            std::vector< CellGeometry<2, 1> >,
            std::vector< CellGeometry<2, 2> >
        > geometry_;
    };



    inline Geometry<2>::Geometry(const TopologicalGridType & tGrid, const bool fill){


        std::get<0>(geometry_).resize(tGrid.numberOfCells()[0]);
        std::get<1>(geometry_).resize(tGrid.numberOfCells()[1]);
        std::get<2>(geometry_).resize(tGrid.numberOfCells()[2]);

        if(!fill){
            nifty::tools::forEachCoordinate(tGrid.topologicalGridShape(), [&](
                const SignedCoordinateType & tCoord
            ){
          

                auto even0 = tCoord[0] % 2 == 0 ;
                auto even1 = tCoord[1] % 2 == 0 ;

                const auto cellLabel = tGrid(tCoord);
                CoordinateType tCoordCasted({uint32_t(tCoord[0]), uint32_t(tCoord[1])});

                if(even0 && even1){
                    std::get<2>(geometry_)[cellLabel-1].push_back(tCoordCasted);
                }
                else if(!even0 && !even1){
                    if(cellLabel){
                        std::get<0>(geometry_)[cellLabel-1][0] = tCoordCasted;
                    }
                }
                else{
                    if(cellLabel){
                       std::get<1>(geometry_)[cellLabel-1].push_back(tCoordCasted); 
                    }
                }
            });
        }
        else{

            // pass 1 
            nifty::tools::forEachCoordinate(tGrid.topologicalGridShape(), [&](
                const SignedCoordinateType & tCoord
            ){


                // compute even odd
                auto even0 = tCoord[0] % 2 == 0 ;
                auto even1 = tCoord[1] % 2 == 0 ;

                const auto cellLabel = tGrid(tCoord);
                CoordinateType tCoordCasted({uint32_t(tCoord[0]), uint32_t(tCoord[1])});

                if(even0 && even1){
                    std::get<2>(geometry_)[cellLabel-1].push_back(tCoordCasted);
                }
                // junction (0-Cell)
                else if(!even0 && !even1){
                    if(cellLabel == 0){
                        
                        // check if to relabel inactive cell-0 as   cell-1 or cell-2
                        //   d 
                        // a * c
                        //   b 
                        const auto a = tGrid(tCoord[0] - 1, tCoord[1]    );
                        const auto b = tGrid(tCoord[0]    , tCoord[1] + 1);
                        const auto c = tGrid(tCoord[0] + 1, tCoord[1]    );
                        const auto d = tGrid(tCoord[0]    , tCoord[1] - 1);

                        if(a && ( a==b  || a==c  || a==d ) ){
                            // relabel inactive cell-0 as cell-1
                            std::get<1>(geometry_)[a-1].push_back(tCoordCasted);

                        }
                        else if(b && ( b==c  || b==d) ){
                            std::get<1>(geometry_)[b-1].push_back(tCoordCasted);
                        }
                        else if(c && ( c==d  ) ){
                            // relabel inactive cell-0 as cell-1
                            std::get<1>(geometry_)[c-1].push_back(tCoordCasted);
                        }
                        else{
                            // relabel inactive cell-0 as cell-2
                            const auto cell2Label = tGrid( tCoord[0]-1, tCoord[1]-1);
                            std::get<2>(geometry_)[cell2Label-1].push_back(tCoordCasted);
                        }
                    }
                    else{
                        
                        std::get<0>(geometry_)[cellLabel-1][0] = tCoordCasted;
                    }

                }
                // boundary (1-Cell)
                else{
                    if(cellLabel == 0){
                        // relabel inactive cell-1 as cell-2
                        if(!even0){
                            const auto cell2Label = tGrid( tCoord[0]-1, tCoord[1] );
                            std::get<2>(geometry_)[cell2Label-1].push_back(tCoordCasted);
                        } 
                        else{
                            const auto cell2Label = tGrid( tCoord[0], tCoord[1]-1 );
                            std::get<2>(geometry_)[cell2Label-1].push_back(tCoordCasted);
                        }
                    }
                    else{   
                        std::get<1>(geometry_)[cellLabel-1].push_back(tCoordCasted); 
                    }
                }

                //std::cout<<"\n";
            });


        }
    }





} // namespace nifty::cgp
} // namespace nifty


