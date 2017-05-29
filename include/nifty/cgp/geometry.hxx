#pragma once

#include <vector>
#include <deque>

#include "nifty/marray/marray.hxx"
#include "nifty/cgp/topological_grid.hxx"
#include "nifty/array/arithmetic_array.hxx"
#include "nifty/container/boost_flat_set.hxx"

namespace nifty{
namespace cgp{


    template<size_t DIM>
    class Geometry;

    
    template<size_t DIM, size_t CELL_TYPE>
    class CellGeometry;

    // zero cell should be the same for all dimensions,
    // a single dot
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


    // Edges in 2D
    template<>
    class CellGeometry<2, 1> : public std::vector< array::StaticArray<uint32_t, 2> > {
    public:
        friend class Geometry<2>;
        const static size_t DIM = 2;
        typedef std::integral_constant<size_t, DIM> DimensionType;
        typedef nifty::array::StaticArray<uint32_t, DIM> CoordinateType;
        typedef nifty::array::StaticArray<float, DIM> FloatCoordinateType;
        typedef std::vector<CoordinateType> BaseType;
        using BaseType::BaseType;

        CellGeometry()
        :   BaseType(),
            isSorted_(false){
        }

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


    private:
        bool isSorted_;
    };


    // default cell
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


    template<size_t DIM, size_t CELL_TYPE>
    class CellGeometryVector  :
        public std::vector<CellGeometry<DIM, CELL_TYPE> > 
    {

    };

    template<size_t DIM>
    class Geometry;




    template<>
    class Geometry<2>{
    public:

        typedef array::StaticArray<uint32_t, 2> CoordinateType;
        typedef array::StaticArray<int64_t, 2>  SignedCoordinateType;

        typedef TopologicalGrid<2> TopologicalGridType;

        Geometry(const TopologicalGridType & tGrid, const bool fill=false, const bool sort1Cells=true);

        template<size_t CELL_TYPE>
        const CellGeometryVector<2,CELL_TYPE> &
        geometry()const{
            return std::get<CELL_TYPE>(geometry_);
        }

    private:
        
        std::tuple<
            CellGeometryVector<2, 0> ,
            CellGeometryVector<2, 1> ,
            CellGeometryVector<2, 2> 
        > geometry_;

        
    };



    inline Geometry<2>::Geometry(const TopologicalGridType & tGrid, const bool fill, const bool sort1Cells){


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

              
            });
        }

        // the complexity of the current implementation is o(n*2) which is
        // a waste! 
        // but i guess in practice for 2D this might not be harmful
        if(sort1Cells){
           
            auto & cells = std::get<1>(geometry_);

            for(uint32_t cell1Index=0; cell1Index<tGrid.numberOfCells()[1]; ++cell1Index){
                auto & geo     = cells[cell1Index];
                geo.isSorted_ = true;
                auto nUsed = 0;
                std::vector<bool> used(geo.size(), false);
                std::deque<CoordinateType>  sorted;
                sorted.push_back(geo.front());
                used[0] = true;
                ++nUsed;


                auto isMatch = [fill](const CoordinateType & a, const CoordinateType b){
                    if(fill){
                        // if we used filled coordinates,
                        // only one coordinate should differ by one
                        // => if so, we know it's a match
                        
                        const auto dx = std::abs(int(a[0]) -int(b[0]));
                        const auto dy = std::abs(int(a[1]) -int(b[1]));
                        return dx + dy == 1;
                    }
                    else{
                        // more complicated
                        //    horizontal   
                        //      |   |
                        //    - * - * - 
                        //      |   |
                        //      
                        if(a[0]%2 == 0){
                            const int offsets[6][2] = {
                                {-2, 0},
                                { 2, 0},
                                {-1,-1},
                                { 1,-1},
                                {-1, 1},
                                { 1, 1}                           
                            };
                            for(auto oi=0; oi<6; ++oi){
                                if(b[0]+offsets[oi][0] == a[0] && b[1]+offsets[oi][1]){
                                    return true;
                                }
                            }
                        }   
                        //     vertical   
                        //         | 
                        //       - * -
                        //         |
                        //       - * -
                        //         |
                        //         
                        if(a[0]%2 == 1){
                            const int offsets[6][2] = {
                                { 0,-2},
                                { 0, 2},
                                {-1,-1},
                                {-1, 1},
                                { 1,-1},
                                { 1, 1}                         
                            };
                            for(auto oi=0; oi<6; ++oi){
                                if(b[0]+offsets[oi][0] == a[0] && b[1]+offsets[oi][1]){
                                    return true;
                                }
                            }
                        }
                        return false;
                    }
                };
                while(nUsed != geo.size()){
                    auto added = false;
                    for(auto c=0; c<geo.size(); ++c){

                        if(!used[c]){
                            auto & coord = geo[c];
                            // check if coord matches begin or end
                            if(isMatch(sorted.front(), coord)){
                                sorted.push_front(coord);
                                used[c] = true;
                                ++nUsed;
                                added = true;
                                continue;
                            }
                            else if(isMatch(sorted.back(), coord)){
                                sorted.push_back(coord);
                                used[c] = true;
                                ++nUsed;
                                added = true;
                                continue;
                            }
                        }
                       
                    }
                    NIFTY_CHECK(added,"internal error, please contacts developers");

                }

                // here it is sorted
                //geo.clear();

                std::copy(sorted.begin(), sorted.end(), geo.begin());
            }
        }
    }





} // namespace nifty::cgp
} // namespace nifty


