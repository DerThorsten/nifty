#pragma once

#include <vector>
#include <unordered_map>
#include <map>

#include "nifty/array/arithmetic_array.hxx"
#include "nifty/tools/for_each_coordinate.hxx"
#include "nifty/container/boost_flat_set.hxx"
#include "nifty/ufd/ufd.hxx"
#include "nifty/tools/timer.hxx"
#include "nifty/xtensor/xtensor.hxx"

namespace nifty{
namespace cgp{

    template<size_t DIM>
    class TopologicalGrid;


    /**
     * @brief      Class for cartesian grid partitioning
     * for 2d images
     *
     * @tparam     INDEX_TYPE  { description }
     */
    template<>
    class TopologicalGrid<2>{
    private:
        typedef array::StaticArray<int64_t, 2> CoordinateType;
        typedef array::StaticArray<uint32_t, 3> NumberOfCellsType;

    public:

        template<class LABELS>
        TopologicalGrid(const xt::xexpression<LABELS> & labelsExp);


        uint32_t operator()(const CoordinateType & coord)const{
            return xtensor::read(tGrid_, coord.asStdArray());
        }

        uint32_t operator()(const uint32_t x0, const uint32_t x1)const{
            return tGrid_(x0, x1);
        }

        const NumberOfCellsType & numberOfCells() const{
            return numberOfCells_;
        }

        const xt::xtensor<uint32_t, 2> & array()const{
            return tGrid_;
        }

        const CoordinateType & shape()const{
            return shape_;
        }

        const CoordinateType & topologicalGridShape()const{
            return tShape_;
        }

    private:

        CoordinateType shape_;
        CoordinateType tShape_;
        NumberOfCellsType numberOfCells_;

        xt::xtensor<uint32_t, 2> tGrid_;
    };


    template<class LABELS>
    inline TopologicalGrid<2>::TopologicalGrid(
        const xt::xexpression<LABELS> & labelsExp
    ) :
        shape_({{labelsExp.derived_cast().shape()[0],
                 labelsExp.derived_cast().shape()[1]}}),
        tShape_({{2*labelsExp.derived_cast().shape()[0] - 1,
                  2*labelsExp.derived_cast().shape()[1] - 1}}),
        tGrid_({2*labels.derived_cast().shape()[0]-1,
                2*labels.derived_cast().shape()[1]-1}, 0)
    {

        NIFTY_CHECK_OP(labels.dimension(),==,2,"wrong dimensions");

        uint32_t jLabel = 1, bLabel = 1, maxNodeLabel = 0;
        // pass 1
        nifty::tools::forEachCoordinate(tShape_, [&](
            const CoordinateType & tCoord
        ){
            // compute even odd
            auto even0 = tCoord[0] % 2 == 0 ;
            auto even1 = tCoord[1] % 2 == 0 ;

            if(even0 && even1){
                //std::cout<<" N \n";
                const auto l = labels(tCoord[0]/2, tCoord[1]/2);
                maxNodeLabel = std::max(maxNodeLabel, static_cast<uint32_t>(l));
                tGrid_(tCoord[0], tCoord[1]) = l;
            }
            // junction (0-Cell)
            else if(!even0 && !even1){
                const auto a = labels((tCoord[0]-1)/2,(tCoord[1]-1)/2);
                const auto b = labels((tCoord[0]+1)/2,(tCoord[1]-1)/2);
                const auto c = labels((tCoord[0]-1)/2,(tCoord[1]+1)/2);
                const auto d = labels((tCoord[0]+1)/2,(tCoord[1]+1)/2);

                const auto nEdge = (a==b ? 0:1) + (c==d ? 0:1) +
                                   (a==c ? 0:1) + (b==d ? 0:1);
                tGrid_(tCoord[0], tCoord[1]) = nEdge;
            }
            // boundary (1-Cell)
            else{
                //std::cout<<" E \n";
                T l0,l1;
                // A|B
                // vertical  boundary
                if(!even0){
                    l0=labels( (tCoord[0]-1)/2, tCoord[1]/2 );
                    l1=labels( (tCoord[0]+1)/2, tCoord[1]/2 );
                }
                // horizontal boundary
                else{
                    l0=labels( tCoord[0]/2, (tCoord[1]-1)/2);
                    l1=labels( tCoord[0]/2, (tCoord[1]+1)/2);
                }
                // active boundary ?
                if(l0!=l1){
                    //std::cout<<" active \n";
                    tGrid_(tCoord[0],tCoord[1])=bLabel;
                    ++bLabel;
                }
                //else
                //    tGrid_(tCoord[0],tCoord[1])=0;
            }

            //std::cout<<"\n";
        });

        nifty::ufd::Ufd<uint32_t> edgeUfd(bLabel-1);

        nifty::tools::forEachCoordinate(tShape_, [&](
            const CoordinateType & tCoord
        ){
            // compute even odd
            auto even0 = tCoord[0] % 2 == 0 ;
            auto even1 = tCoord[1] % 2 == 0 ;

            if(!even0 && !even1){

                const auto nEdge = tGrid_(tCoord[0], tCoord[1]);
                if(nEdge < 2){
                    tGrid_(tCoord[0], tCoord[1]) = 0;
                }
                else if(nEdge == 2){

                    auto mergeIfActive = [&](const uint32_t e0, const uint32_t e1){
                        if(e1){
                            edgeUfd.merge(e0-1, e1-1);
                        }
                    };
                    const auto a = tGrid_(tCoord[0] - 1, tCoord[1]    );
                    const auto b = tGrid_(tCoord[0]    , tCoord[1] + 1);
                    const auto c = tGrid_(tCoord[0] + 1, tCoord[1]    );
                    const auto d = tGrid_(tCoord[0]    , tCoord[1] - 1);
                    if(a){
                        mergeIfActive(a, b);
                        mergeIfActive(a, c);
                        mergeIfActive(a, d);
                    }
                    else if(b){
                        mergeIfActive(b, c);
                        mergeIfActive(b, d);
                    }
                    else if(c){
                        mergeIfActive(c, d);
                    }
                    tGrid_(tCoord[0],tCoord[1]) = 0;
                }
                else if(nEdge == 3){
                    tGrid_(tCoord[0],tCoord[1]) =jLabel;
                    ++jLabel;
                }
                else if(nEdge == 4){
                    // A | B
                    // - * -
                    // C | D
                    const auto nA = tGrid_(tCoord[0]-1, tCoord[1]-1);
                    const auto nB = tGrid_(tCoord[0]+1, tCoord[1]-1);
                    const auto nC = tGrid_(tCoord[0]-1, tCoord[1]+1);
                    const auto nD = tGrid_(tCoord[0]+1, tCoord[1]+1);
                    //   d
                    // a * c
                    //   b
                    const auto a = tGrid_(tCoord[0] - 1, tCoord[1]    );
                    const auto b = tGrid_(tCoord[0]    , tCoord[1] + 1);
                    const auto c = tGrid_(tCoord[0] + 1, tCoord[1]    );
                    const auto d = tGrid_(tCoord[0]    , tCoord[1] - 1);


                    if(nA == nD){
                        edgeUfd.merge(a-1, b-1);
                        edgeUfd.merge(c-1, d-1);
                        tGrid_(tCoord[0],tCoord[1]) = 0;
                    }
                    else if(nB == nC){
                        edgeUfd.merge(a-1, d-1);
                        edgeUfd.merge(b-1, c-1);
                        tGrid_(tCoord[0],tCoord[1]) = 0;
                    }
                    else{
                        tGrid_(tCoord[0],tCoord[1]) = jLabel;
                        ++jLabel;
                    }
                }
            }
        });

        numberOfCells_[0] = jLabel - 1;
        numberOfCells_[1] = edgeUfd.numberOfSets();
        numberOfCells_[2] = maxNodeLabel;

        // pass 3,make dense
        std::unordered_map<uint32_t, uint32_t> rmap;
        edgeUfd.representativeLabeling(rmap);

        nifty::tools::forEachCoordinate(tShape_, [&](
            const CoordinateType & tCoord
        ){
            auto even0 = tCoord[0] % 2 == 0 ;
            auto even1 = tCoord[1] % 2 == 0 ;

            if((even0 && !even1) || (!even0 && even1)){
                auto & l = tGrid_(tCoord[0], tCoord[1]);
                if(l){
                    l = rmap[edgeUfd.find(l-1)] + 1;
                }
            }
        });
    }

} // namespace nifty::cgp
} // namespace nifty
