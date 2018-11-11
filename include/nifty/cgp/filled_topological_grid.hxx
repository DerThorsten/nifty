#pragma once

#include <vector>

#include "nifty/xtensor/xtensor.hxx"
#include "nifty/array/arithmetic_array.hxx"
#include "nifty/tools/for_each_coordinate.hxx"
#include "nifty/container/boost_flat_set.hxx"
#include "nifty/cgp/topological_grid.hxx"

namespace nifty{
namespace cgp{




    template<size_t DIM>
    class FilledTopologicalGrid;


    /**
     * @brief      Class for cartesian grid partitioning
     * for 2d images
     *
     * @tparam     INDEX_TYPE  { description }
     */
    template<>
    class FilledTopologicalGrid<2>{
    public:
        typedef array::StaticArray<int64_t, 2 > CoordinateType;
        typedef array::StaticArray<uint32_t, 3 > NumberOfCellsType;
        typedef TopologicalGrid<2> TopologicalGridType;

        FilledTopologicalGrid(const TopologicalGridType &);


        const NumberOfCellsType & numberOfCells() const{
            return numberOfCells_;
        }

        const nifty::marray::Marray<uint32_t> & array()const{
            return ftGrid_;
        }

        const CoordinateType & shape()const{
            return shape_;
        }

        const CoordinateType & topologicalGridShape()const{
            return tShape_;
        }
        const NumberOfCellsType & cellTypeOffset()const{
            return cellTypeOffset_;
        }
    private:

        CoordinateType shape_;
        CoordinateType tShape_;
        NumberOfCellsType numberOfCells_;

        nifty::marray::Marray<uint32_t> ftGrid_;
        NumberOfCellsType cellTypeOffset_;
    };


    inline FilledTopologicalGrid<2>::FilledTopologicalGrid(const TopologicalGridType & tGrid)
    :
        shape_(tGrid.shape()),
        tShape_(tGrid.topologicalGridShape()),
        numberOfCells_(tGrid.numberOfCells()),
        ftGrid_(tGrid.array()),
        cellTypeOffset_{tGrid.numberOfCells()[1] + tGrid.numberOfCells()[2],tGrid.numberOfCells()[2],uint32_t(0)}
    {

        // pass 1
        nifty::tools::forEachCoordinate(tShape_, [&](
            const CoordinateType & tCoord
        ){


            // compute even odd
            auto even0 = tCoord[0] % 2 == 0 ;
            auto even1 = tCoord[1] % 2 == 0 ;

            if(even0 && even1){}
            // junction (0-Cell)
            else if(!even0 && !even1){

                const auto cell0Label = tGrid(tCoord);
                if(cell0Label == 0){
                    // check if to relabel inactive cell-0 as   cell-1 or cell-2
                    //   d
                    // a * c
                    //   b
                    const auto a = tGrid(tCoord[0] - 1, tCoord[1]    );
                    const auto b = tGrid(tCoord[0]    , tCoord[1] + 1);
                    const auto c = tGrid(tCoord[0] + 1, tCoord[1]    );
                    const auto d = tGrid(tCoord[0]    , tCoord[1] - 1);

                    if(a && ( a==b  || a==d  || a==c ) ){
                        // relabel inactive cell-0 as cell-1
                        ftGrid_(tCoord.asStdArray()) = a + cellTypeOffset_[1];
                    }
                    else if(b && ( b==d  || b==c ) ){
                        // relabel inactive cell-0 as cell-1
                        ftGrid_(tCoord.asStdArray()) = b + cellTypeOffset_[1];
                    }
                    else if(c && ( c==d  ) ){
                        // relabel inactive cell-0 as cell-1
                        ftGrid_(tCoord.asStdArray()) = c + cellTypeOffset_[1];
                    }
                    else{
                        // relabel inactive cell-0 as cell-2
                        ftGrid_(tCoord.asStdArray()) = tGrid(tCoord[0]-1, tCoord[1]-1);
                    }
                }
                else{
                    // if cell 0 is active
                    ftGrid_(tCoord.asStdArray()) += cellTypeOffset_[0];
                }

            }
            // boundary (1-Cell)
            else{
                const auto cell1Label = tGrid(tCoord);
                // relabel inactive cell-1 as cell-2
                if(cell1Label == 0){
                    if(!even0){
                        ftGrid_(tCoord.asStdArray()) = tGrid( tCoord[0]-1, tCoord[1] );
                    }
                    else{
                        ftGrid_(tCoord.asStdArray()) = tGrid( tCoord[0], tCoord[1]-1 );
                    }
                }
                else{
                    ftGrid_(tCoord.asStdArray()) += cellTypeOffset_[1];
                }
            }

            //std::cout<<"\n";
        });
    }

} // namespace nifty::cgp
} // namespace nifty


