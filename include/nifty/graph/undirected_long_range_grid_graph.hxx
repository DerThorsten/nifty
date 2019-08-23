#pragma once

#include "xtensor/xexpression.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xmath.hpp"

#include "nifty/graph/undirected_list_graph.hxx"
#include "nifty/array/arithmetic_array.hxx"
#include "nifty/tools/for_each_coordinate.hxx"

namespace nifty{
namespace graph{


    template<std::size_t DIM>
    class UndirectedLongRangeGridGraph;

    ///\cond
    namespace detail_graph{

        template<std::size_t DIM>
        class UndirectedLongRangeGridGraphAssign;

        template<>
        class UndirectedLongRangeGridGraphAssign<2>{
        public:
            template<class G>
            static void assign(
               G & graph
            ){
                const auto & shape = graph.shape();
                const auto & offsets = graph.offsets();
                uint64_t u=0;
                for(int p0=0; p0< graph.shape()[0]; ++p0)
                for(int p1=0; p1< graph.shape()[1]; ++p1){
                    for(int io=0; io<offsets.size(); ++io){
                        const int q0 = p0 + offsets[io][0];
                        const int q1 = p1 + offsets[io][1];
                        if(q0>=0 && q0<shape[0] && q1>=0 && q1<shape[1]){
                            const auto v = q0*shape[1] + q1;
                            const auto e = graph.insertEdge(u, v);
                        }
                    }
                    ++u;
                }
            }



        };

        template<>
        class UndirectedLongRangeGridGraphAssign<3>{
        public:
            template<class G>
            static void assign(
                G & graph
            ){
                const auto & shape = graph.shape();
                const auto & offsets = graph.offsets();
                uint64_t u=0;
                for(int p0=0; p0<shape[0]; ++p0)
                for(int p1=0; p1<shape[1]; ++p1)
                for(int p2=0; p2<shape[2]; ++p2){
                    for(int io=0; io<offsets.size(); ++io){
                        const int q0 = p0 + offsets[io][0];
                        const int q1 = p1 + offsets[io][1];
                        const int q2 = p2 + offsets[io][2];
                        if(q0>=0 && q0<shape[0] && q1>=0 && q1<shape[1] && q2>=0 && q2<shape[2]){
                            const auto v = q0*shape[1]*shape[2] + q1*shape[2] + q2;
                            const auto e = graph.insertEdge(u, v);
                        }
                    }
                    ++u;
                }
            }
        };
    }
    ///\endcond

    template<std::size_t DIM>
    class UndirectedLongRangeGridGraph
    :   public UndirectedGraph<>
    {
    private:
        typedef detail_graph::UndirectedLongRangeGridGraphAssign<DIM> HelperType;
    public:

        typedef array::StaticArray<int64_t, DIM>    ShapeType;
        typedef array::StaticArray<int64_t, DIM>    StridesType;
        typedef array::StaticArray<int64_t, DIM>    CoordinateType;
        typedef array::StaticArray<int64_t, DIM>    OffsetType;

        typedef std::vector<OffsetType>     OffsetVector;

            
        UndirectedLongRangeGridGraph(
            const ShapeType &    shape,
            const OffsetVector & offsets
        )
        :   UndirectedGraph<>(),
            shape_(shape),
            offsets_(offsets)
        {
            NIFTY_CHECK(DIM==2 || DIM==3,"wrong dimension");

            uint64_t nNodes = shape_[0];
            for(int d=1; d<DIM; ++d){
                nNodes *= shape_[d];
            }
            this->assign(nNodes);

            strides_.back() = 1;
            for(int d=int(DIM)-2; d>=0; --d){
                strides_[d] = shape_[d+1] * strides_[d+1];
            }
            HelperType::assign(*this);
        }



        auto edgeOffsetIndex(
        )const{
            typename xt::xtensor<int32_t, 1>::shape_type retshape;
            retshape[0] = this->numberOfEdges();
            xt::xtensor<int32_t, 1> ret(retshape); 
            uint64_t u = 0;
            nifty::tools::forEachCoordinate( shape_,[&](const auto & coordP){
            auto offsetIndex = 0;
            for(const auto & offset : offsets_){
                    const auto coordQ = offset + coordP;
                    if(coordQ.allInsideShape(shape_)){
                        const auto v = this->coordinateToNode(coordQ);
                        const auto e = this->findEdge(u,v);
                        ret[e] = offsetIndex;
                    }
                    ++offsetIndex;
                }
                ++u;
            });
            
            return ret;
        }

        template<class D>
        auto nodeFeatureDiffereces(
            const xt::xexpression<D> & nodeFeaturesExpression
        )const{
            typedef typename D::value_type value_type;
            const auto & nodeFeatures = nodeFeaturesExpression.derived_cast();
            for(auto d=0; d<DIM; ++d){
                NIFTY_CHECK_OP(shape_[d],==,nodeFeatures.shape()[d], "input has wrong shape");
            }
            const auto nFeatures = nodeFeatures.shape()[DIM];

            typename xt::xtensor<value_type, 1>::shape_type retshape;
            retshape[0] = this->numberOfEdges();
            xt::xtensor<value_type, 1> ret(retshape);

            if(DIM == 2){
                uint64_t u = 0;
                nifty::tools::forEachCoordinate( shape_,[&](const auto & coordP){
                    const auto valP = xt::view(nodeFeatures, coordP[0],coordP[1], xt::all());
                    for(const auto & offset : offsets_){
                        const auto coordQ = offset + coordP;
                        if(coordQ.allInsideShape(shape_)){

                            const auto valQ = xt::view(nodeFeatures, coordQ[0],coordQ[1], xt::all());
                            const auto v = this->coordinateToNode(coordQ);
                            const auto e = this->findEdge(u,v);
                            NIFTY_CHECK_OP(e,>=,0,"");
                            ret[e] = xt::sum(xt::pow(valP-valQ, 2))();
                        }
                    }
                    ++u;
                });
            }
            if(DIM == 3){
                uint64_t u = 0;
                nifty::tools::forEachCoordinate( shape_,[&](const auto & coordP){
                    const auto valP = xt::view(nodeFeatures, coordP[0], coordP[1], coordP[2], xt::all());
                    for(const auto & offset : offsets_){
                        const auto coordQ = offset + coordP;
                        if(coordQ.allInsideShape(shape_)){
                            const auto valQ = xt::view(nodeFeatures, coordQ[0], coordQ[1], coordQ[2], xt::all());
                            const auto v = this->coordinateToNode(coordQ);
                            const auto e = this->findEdge(u,v);
                            NIFTY_CHECK_OP(e,>=,0,"");
                            ret[e] = xt::sum(xt::pow(valP-valQ, 2))();
                        }
                    }
                    ++u;
                });
            }
            return ret;
        }

        template<class D>
        auto nodeFeatureDiffereces2(
            const xt::xexpression<D> & nodeFeaturesExpression
        )const{
            typedef typename D::value_type value_type;
            const auto & nodeFeatures = nodeFeaturesExpression.derived_cast();
            for(auto d=0; d<DIM; ++d){
                NIFTY_CHECK_OP(shape_[d],==,nodeFeatures.shape()[d], "input has wrong shape");
            }
            const auto nFeatures = nodeFeatures.shape()[DIM];

            typename xt::xtensor<value_type, 1>::shape_type retshape;
            retshape[0] = this->numberOfEdges();
            xt::xtensor<value_type, 1> ret(retshape);

            if(DIM == 2){
                uint64_t u = 0;
                nifty::tools::forEachCoordinate( shape_,[&](const auto & coordP){
                    
                    auto offsetIndex=0;
                    for(const auto & offset : offsets_){
                        const auto coordQ = offset + coordP;
                        if(coordQ.allInsideShape(shape_)){

                            const auto valP = xt::view(nodeFeatures, coordP[0],coordP[1],offsetIndex, xt::all());
                            const auto valQ = xt::view(nodeFeatures, coordQ[0],coordQ[1],offsetIndex, xt::all());
                            const auto v = this->coordinateToNode(coordQ);
                            const auto e = this->findEdge(u,v);
                            NIFTY_CHECK_OP(e,>=,0,"");
                            ret[e] = xt::sum(xt::pow(valP-valQ, 2))();
                        }
                        ++offsetIndex;
                    }
                    ++u;
                });
            }
            if(DIM == 3){
                uint64_t u = 0;
                nifty::tools::forEachCoordinate( shape_,[&](const auto & coordP){
                    

                    auto offsetIndex=0;
                    for(const auto & offset : offsets_){
                        const auto coordQ = offset + coordP;
                        if(coordQ.allInsideShape(shape_)){
                            const auto valP = xt::view(nodeFeatures, coordP[0], coordP[1], coordP[2],offsetIndex, xt::all());
                            const auto valQ = xt::view(nodeFeatures, coordQ[0], coordQ[1], coordQ[2],offsetIndex, xt::all());
                            const auto v = this->coordinateToNode(coordQ);
                            const auto e = this->findEdge(u,v);
                            NIFTY_CHECK_OP(e,>=,0,"");
                            ret[e] = xt::sum(xt::pow(valP-valQ, 2))();
                        }
                        offsetIndex+=1;
                    }
                    ++u;
                });
            }
            return ret;
        }


        template<class D>
        auto edgeValues(
            const xt::xexpression<D> & valuesExpression
        )const{

            typedef typename D::value_type value_type;
            const auto & values = valuesExpression.derived_cast();

            for(auto d=0; d<DIM; ++d){
                NIFTY_CHECK_OP(shape_[d],==,values.shape()[d], "input has wrong shape");
            }
            NIFTY_CHECK_OP(offsets_.size(),==,values.shape()[DIM], "input has wrong shape");


            typename xt::xtensor<value_type, 1>::shape_type retshape;
            retshape[0] = this->numberOfEdges();
            xt::xtensor<value_type, 1> ret(retshape);


            uint64_t u = 0;
            nifty::tools::forEachCoordinate( shape_,[&](const auto & coordP){
                auto offsetIndex = 0;
                for(const auto & offset : offsets_){
                    const auto coordQ = offset + coordP;
                    if(coordQ.allInsideShape(shape_)){

                        const auto v = this->coordinateToNode(coordQ);
                        const auto e = this->findEdge(u,v);

                        if(DIM == 2){
                            const auto val = values(coordP[0],coordP[1], offsetIndex);
                            ret[e] = val;
                        }
                        else{
                            const auto val = values(coordP[0],coordP[1],coordP[2], offsetIndex);
                            ret[e] = val;
                        }
                    }
                    ++offsetIndex;
                }
                ++u;
            });
            
            return ret;

        }

        // template<class NODE_COORDINATE>
        // void nodeToCoordinate(
        //     const uint64_t node,
        //     NODE_COORDINATE & coordinate
        // )const{
            
        // }

        template<class NODE_COORDINATE>
        uint64_t coordinateToNode(
            const NODE_COORDINATE & coordinate
        )const{
            uint64_t n = 0;
            for(auto d=0; d<DIM; ++d){
                n +=strides_[d]*coordinate[d];
            }
            return n;
        }

        const auto & shape()const{
            return shape_;
        }
        const auto & offsets()const{
            return offsets_;
        }
    private:
        ShapeType shape_;
        StridesType strides_;
        OffsetVector offsets_;

    };
}
}