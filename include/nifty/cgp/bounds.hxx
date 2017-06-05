#pragma once

#include <vector>

#include "nifty/marray/marray.hxx"
#include "nifty/cgp/topological_grid.hxx"
#include "nifty/array/arithmetic_array.hxx"
#include "nifty/container/boost_flat_set.hxx"


namespace nifty{
namespace cgp{

    template<size_t DIM>
    class Bounds;

    template<size_t DIM, size_t CELL_TYPE>
    class CellBoundedByVector;


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

    template<size_t DIM, size_t CELL_TYPE>
    class CellBoundedBy;


    // the junctions of the boundaries
    template<>
    class CellBoundedBy<2, 1>{
        friend class CellBoundedByVector<2,1>;
    public:
        CellBoundedBy()
        : data_{0,0}{

        }
        uint32_t size()const{
            if(data_[0] == 0 ){
                return 0;
            }
            else if(data_[1] == 0){
                return 1;
            }
            else{
                return 2;
            }
        }
        const uint32_t & operator[](const size_t i)const{
            return data_[i];
        }
    private:
        uint32_t data_[2];
    };

    // the edges of the regions
    template<>
    class CellBoundedBy<2, 2>{
        friend class CellBoundedByVector<2,2>;
    public:
        CellBoundedBy()
        : data_(){
        }
        uint32_t size()const{
            return data_.size();
        }
        const uint32_t & operator[](const size_t i)const{
            return data_[i];
        }
    private:
        std::vector<uint32_t> data_;
    };


    template<size_t DIM, size_t CELL_TYPE>
    class CellBoundsVector : public std::vector<CellBounds<DIM, CELL_TYPE>>{
    public:

        friend class Bounds<DIM>;

        typedef array::StaticArray<uint32_t, DIM +1 > NumberOfCellsType;
        typedef std::vector<CellBounds<DIM, CELL_TYPE>> BaseType; 
        using BaseType::BaseType;

        const NumberOfCellsType & numberOfCells() const{
            return numberOfCells_;
        }
    private:
        NumberOfCellsType numberOfCells_;
    };

    template<size_t DIM, size_t CELL_TYPE>
    class CellBoundedByVector;



    // the junctions of the boundaries
    template< >
    class CellBoundedByVector<2,1> : public std::vector<CellBoundedBy<2, 1> >{
    public:
       typedef std::vector<CellBoundedBy<2, 1>> BaseType; 

        // construct from junctions of boundaries
        CellBoundedByVector(
            const CellBoundsVector<2, 0> & cell0Bounds
        )
        :   BaseType(cell0Bounds.numberOfCells()[1]){

            for(auto cell0Index=0; cell0Index<cell0Bounds.numberOfCells()[0]; ++cell0Index){
                const auto & bounds = cell0Bounds[cell0Index];
                for(auto i=0; i<bounds.size(); ++i){
                    const auto cell1Label = bounds[i];

                    auto & boundedBy = this->operator[](cell1Label-1);
                    const auto s = boundedBy.size();
                    boundedBy.data_[s] = cell0Index + 1;
                }
            }
        }
    };

    // the edges of the regions
    template< >
    class CellBoundedByVector<2,2> : public std::vector<CellBoundedBy<2, 2> >{
    public:
       typedef std::vector<CellBoundedBy<2, 2>> BaseType; 

        // construct from regions of boundaries
        CellBoundedByVector(
            const CellBoundsVector<2, 1> & cell1Bounds
        )
        :   BaseType(cell1Bounds.numberOfCells()[2]){

            for(auto cell1Index=0; cell1Index<cell1Bounds.numberOfCells()[1]; ++cell1Index){
                const auto & bounds = cell1Bounds[cell1Index];
                for(auto i=0; i<bounds.size(); ++i){
                    const auto cell2Label = bounds[i];
                    auto & boundedBy = this->operator[](cell2Label-1);
                    boundedBy.data_.push_back(cell1Index);
                }
            }
        }
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
        const CellBoundsVector<2, CELL_TYPE> &
        bounds()const{
            return std::get<CELL_TYPE>(bounds_);
        }

    private:
        
        std::tuple<
            CellBoundsVector<2, 0>,
            CellBoundsVector<2, 1>
        > bounds_;
    };



    inline Bounds<2>::Bounds(const TopologicalGridType & tGrid){

        std::get<0>(bounds_).resize(tGrid.numberOfCells()[0]);
        std::get<1>(bounds_).resize(tGrid.numberOfCells()[1]);

        std::get<0>(bounds_).numberOfCells_  = tGrid.numberOfCells();
        std::get<1>(bounds_).numberOfCells_  = tGrid.numberOfCells();            ;

        nifty::tools::forEachCoordinate(tGrid.topologicalGridShape(), [&](
            const SignedCoordinateType & tCoord
        ){
      

            auto even0 = tCoord[0] % 2 == 0 ;
            auto even1 = tCoord[1] % 2 == 0 ;


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

                    std::get<1>(bounds_)[cell1Label-1] = CellBounds<2,1>(cell2LabelA,cell2LabelB);
                }
            }
        });


        // sort the bounded by relation of 2 cells
        // such that one can iterate over them in order??


    }

} // namespace nifty::cgp
} // namespace nifty


