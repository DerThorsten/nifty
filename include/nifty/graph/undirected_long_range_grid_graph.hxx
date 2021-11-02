#pragma once

#include "xtensor/xexpression.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xmath.hpp"

#include "nifty/graph/undirected_list_graph.hxx"
#include "nifty/array/arithmetic_array.hxx"
#include "nifty/tools/for_each_coordinate.hxx"

#include <cstdlib>

namespace nifty{
namespace graph{


    template<std::size_t DIM>
    class UndirectedLongRangeGridGraph;

    ///\cond
    namespace detail_graph{

        template<std::size_t DIM>
        class UndirectedLongRangeGridGraphAssign{
        public:
            template<class G, class D>
            static void assign(
                G & graph,
                const xt::xexpression<D> & edgeMaskExp,
                const bool hasMaskedEdges

            ){
                const auto & edgeMask = edgeMaskExp.derived_cast();
                const auto & shape = graph.shape();
                const auto & offsets = graph.offsets();

                nifty::tools::forEachCoordinate(
                    shape,
                    [&](const auto & coordP){
                        const auto u = graph.coordinateToNode(coordP);
                        for(int io=0; io<offsets.size(); ++io){
                            const auto offset = offsets[io];
                            const auto coordQ = offset + coordP;

                            // Check if both coordinates are in the volume:
                            if(coordQ.allInsideShape(shape)){
                                const auto v = graph.coordinateToNode(coordQ);
                                bool isValidEdge = true;

                                if (hasMaskedEdges)
                                {
                                    // Check if the edge should be added according to the mask:
                                    array::StaticArray<int64_t, DIM + 1> maskIndex;
                                    std::copy(std::begin(coordP), std::end(coordP), std::begin(maskIndex));
                                    maskIndex[DIM] = io;
                                    isValidEdge = edgeMask[maskIndex];
                                }

                                if (isValidEdge) {
                                    // Insert new edge in graph:
                                    graph.insertEdge(u, v);
                                }
                            }
                        }
                    }
                );
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

        template<class D>
        UndirectedLongRangeGridGraph(
            const ShapeType &    shape,
            const OffsetVector & offsets,
            const xt::xexpression<D> & edgeMaskExp,
            const bool hasMaskedEdges
        )
        :   UndirectedGraph<>(),
            shape_(shape),
            offsets_(offsets),
            hasMaskedEdges_(hasMaskedEdges)
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
            HelperType::assign(*this, edgeMaskExp, hasMaskedEdges);
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
                        if (!hasMaskedEdges_) {
                            NIFTY_CHECK_OP(e,>=,0,"");
                            ret[e] = offsetIndex;
                        } else if (e>=0) {
                            ret[e] = offsetIndex;
                        }
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
                    const auto valP = xt::view(nodeFeatures, coordP[0], coordP[1], xt::all());
                    for(const auto & offset : offsets_){
                        const auto coordQ = offset + coordP;
                        if(coordQ.allInsideShape(shape_)){

                            const auto valQ = xt::view(nodeFeatures, coordQ[0], coordQ[1], xt::all());
                            const auto v = this->coordinateToNode(coordQ);
                            const auto e = this->findEdge(u,v);

                            if (!hasMaskedEdges_) {
                                NIFTY_CHECK_OP(e,>=,0,"");
                                ret[e] = xt::sum(xt::pow(valP - valQ, 2))();
                            } else if (e>=0) {
                                ret[e] = xt::sum(xt::pow(valP - valQ, 2))();
                            }
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
                            if (!hasMaskedEdges_) {
                                NIFTY_CHECK_OP(e,>=,0,"");
                                ret[e] = xt::sum(xt::pow(valP-valQ, 2))();
                            } else if (e>=0) {
                                ret[e] = xt::sum(xt::pow(valP-valQ, 2))();
                            }
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
                            if (!hasMaskedEdges_) {
                                NIFTY_CHECK_OP(e,>=,0,"");
                                ret[e] = xt::sum(xt::pow(valP-valQ, 2))();
                            } else if (e>=0) {
                                ret[e] = xt::sum(xt::pow(valP-valQ, 2))();
                            }
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
                            if (!hasMaskedEdges_) {
                                NIFTY_CHECK_OP(e,>=,0,"");
                                ret[e] = xt::sum(xt::pow(valP-valQ, 2))();
                            } else if (e>=0) {
                                ret[e] = xt::sum(xt::pow(valP-valQ, 2))();
                            }
                        }
                        offsetIndex+=1;
                    }
                    ++u;
                });
            }
            return ret;
        }

        template<class D>
        auto nodeValues(
                const xt::xexpression<D> & valuesExpression
        )const {

            typedef typename D::value_type value_type;
            const auto &values = valuesExpression.derived_cast();

            for (auto d = 0; d < DIM; ++d) {
                NIFTY_CHECK_OP(shape_[d], == , values.shape()[d], "input has wrong shape");
            }


            typename xt::xtensor<value_type, 1>::shape_type retshape;
            retshape[0] = this->numberOfNodes();
            xt::xtensor<value_type, 1> ret(retshape);


            nifty::tools::forEachCoordinate(shape_, [&](const auto &coordP) {
                const auto u = this->coordinateToNode(coordP);
                const auto val = values[coordP];
                ret[u] = val;
            });

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

                        array::StaticArray<int64_t, DIM + 1> valIndex;
                        std::copy(std::begin(coordP), std::end(coordP), std::begin(valIndex));
                        valIndex[DIM] = offsetIndex;
                        const auto val = values[valIndex];

                        if (!hasMaskedEdges_) {
                            NIFTY_CHECK_OP(e,>=,0,"");
                            ret[e] = val;
                        } else if (e>=0) {
                            ret[e] = val;
                        }
                    }
                    ++offsetIndex;
                }
                ++u;
            });
            return ret;
        }

        // Maps Edge IDs to an image tensor (4D). If an edge does not exist, then value -1 is returned.
        auto projectEdgesIDToPixels(
        )const{
            typename xt::xtensor<int64_t, DIM+1>::shape_type retshape;
            for(auto d=0; d<DIM; ++d){
                retshape[d] = shape_[d];
            }
            retshape[DIM] = offsets_.size();
            xt::xtensor<int64_t, DIM+1> ret(retshape);

            uint64_t u = 0;
            nifty::tools::forEachCoordinate( shape_,[&](const auto & coordP){
                auto offsetIndex = 0;
                for(const auto & offset : offsets_){
                    const auto coordQ = offset + coordP;
                    int64_t e = -1;

                    if(coordQ.allInsideShape(shape_)){
                        const auto v = this->coordinateToNode(coordQ);
                        e = this->findEdge(u,v);
                        if (!hasMaskedEdges_) {
                            NIFTY_CHECK_OP(e, >=, 0, "");
                        }
                    }

                    array::StaticArray<int64_t, DIM + 1> retIndex;
                    std::copy(std::begin(coordP), std::end(coordP), std::begin(retIndex));
                    retIndex[DIM] = offsetIndex;

                    ret[retIndex] = e;

                    ++offsetIndex;
                }
                ++u;
            });

            return ret;

        }

        auto projectNodesIDToPixels(
        )const{
            typename xt::xtensor<uint64_t, DIM>::shape_type retshape;
            for(auto d=0; d<DIM; ++d){
                retshape[d] = shape_[d];
            }
            xt::xtensor<uint64_t, DIM> ret(retshape);


            nifty::tools::forEachCoordinate( shape_,[&](const auto & coordP){
                const auto u = this->coordinateToNode(coordP);
                ret[coordP] = u;
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
        bool hasMaskedEdges_;

    };
}
}
