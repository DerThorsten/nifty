#pragma once

#include <cstddef>
#include <boost/iterator/counting_iterator.hpp>
#include <boost/iterator/transform_iterator.hpp>
#include <boost/iterator/iterator_facade.hpp>

#include "andres/graph/grid-graph.hxx"

#include "nifty/tools/runtime_check.hxx"
#include "nifty/tools/for_each_coordinate.hxx"
#include "nifty/graph/undirected_graph_base.hxx"
#include "nifty/graph/detail/adjacency.hxx"
#include "nifty/graph/graph_tags.hxx"
#include "nifty/parallel/threadpool.hxx"
#include "nifty/array/arithmetic_array.hxx"


namespace nifty{
namespace graph{




namespace detail_graph{

















    template<std::size_t DIM, bool SIMPLE_NH>
    class UndirectedGridGraphIter{
    public:
        typedef andres::graph::GridGraph<DIM> AGridGraph;
        typedef typename AGridGraph::AdjacencyIterator AGridGraphAdjacencyIter;
        typedef UndirectedAdjacency<int64_t,int64_t,int64_t,int64_t> NodeAdjacency;

        struct UnaryFunction{
            typedef NodeAdjacency value_type;
            template<class ADJ>
            NodeAdjacency operator()(const ADJ & adjacency)const{
                return NodeAdjacency(adjacency.vertex(), adjacency.vertex());
            }
        };

        typedef boost::transform_iterator<
            UnaryFunction,
            typename AGridGraph::AdjacencyIterator,
            NodeAdjacency,
            NodeAdjacency
        > OldAdjacencyIter;



        class AdjacencyIter
        : public boost::iterator_facade<
            AdjacencyIter,
            NodeAdjacency,
            std::random_access_iterator_tag,
            const NodeAdjacency &
        >
        {
        public:
            AdjacencyIter(const AGridGraphAdjacencyIter & iter)
            :   iter_(iter),
                adjacency_(){
            }
            bool equal(const AdjacencyIter & other)const{
                return iter_ == other.iter_;
            }
            void increment(){
                ++iter_;
            }
            void dencrement(){
                --iter_;
            }
            void advance(const std::size_t n){
                iter_+=n;
            }
            std::ptrdiff_t distance_to(const AdjacencyIter & other)const{
                return std::distance(iter_, other.iter_);
            }
            const NodeAdjacency & dereference()const{
                adjacency_ = NodeAdjacency(iter_->vertex(), iter_->edge());
                return adjacency_;
            }
        private:
            mutable AGridGraphAdjacencyIter iter_;
            mutable NodeAdjacency adjacency_;
        };


        class NodeIter : public boost::counting_iterator<int64_t>{
            using boost::counting_iterator<int64_t>::counting_iterator;
            using boost::counting_iterator<int64_t>::operator=;
        };

        class EdgeIter : public boost::counting_iterator<int64_t>{
            using boost::counting_iterator<int64_t>::counting_iterator;
            using boost::counting_iterator<int64_t>::operator=;
        };
    };



};


template<std::size_t DIM, bool SIMPLE_NH>
class UndirectedGridGraph;



template<std::size_t DIM>
class UndirectedGridGraph<DIM,true> : public
    UndirectedGraphBase<
        UndirectedGridGraph<DIM, true>,
        typename detail_graph::UndirectedGridGraphIter<DIM,true>::NodeIter,
        typename detail_graph::UndirectedGridGraphIter<DIM,true>::EdgeIter,
        typename detail_graph::UndirectedGridGraphIter<DIM,true>::AdjacencyIter
    >
{
private:
    typedef andres::graph::GridGraph<DIM> AndresGridGraphType;
    typedef typename AndresGridGraphType::VertexCoordinate AndresVertexCoordinate;
public:
    typedef nifty::array::StaticArray<int64_t, DIM> ShapeType;
    typedef nifty::array::StaticArray<int64_t, DIM> CoordinateType;

    typedef typename detail_graph::UndirectedGridGraphIter<DIM,true>::NodeIter      NodeIter;
    typedef typename detail_graph::UndirectedGridGraphIter<DIM,true>::EdgeIter      EdgeIter;
    typedef typename detail_graph::UndirectedGridGraphIter<DIM,true>::AdjacencyIter AdjacencyIter;


    typedef ContiguousTag EdgeIdTag;
    typedef ContiguousTag NodeIdTag;

    typedef SortedTag EdgeIdOrderTag;
    typedef SortedTag NodeIdOrderTag;



    UndirectedGridGraph()
    : gridGraph_(){
    }

    template<class T>
    UndirectedGridGraph(const nifty::array::StaticArray<T, DIM> & shape)
    : gridGraph_(){

        AndresVertexCoordinate ashape;
        std::copy(shape.rbegin(), shape.rend(), ashape.begin());
        gridGraph_.assign(ashape);

    }

    template<class T>
    void assign(const nifty::array::StaticArray<T, DIM> & shape){

        AndresVertexCoordinate ashape;
        std::copy(shape.rbegin(), shape.rend(), ashape.begin());
        gridGraph_.assign(ashape);

    }


    //void assign(const uint64_t numberOfNodes = 0, const uint64_t reserveNumberOfEdges = 0);



    // MUST IMPL INTERFACE
    int64_t u(const int64_t e)const{
        return gridGraph_.vertexOfEdge(e,0);
    }
    int64_t v(const int64_t e)const{
        return gridGraph_.vertexOfEdge(e,1);
    }

    int64_t findEdge(const int64_t u, const int64_t v)const{
        const auto r = gridGraph_.findEdge(u,v);
        if(r.first)
            return r.second;
        else
            return -1;
    }
    int64_t nodeIdUpperBound() const{
         return numberOfNodes() == 0 ? 0 : numberOfNodes()-1;
    }
    int64_t edgeIdUpperBound() const{
        return numberOfEdges() == 0 ? 0 : numberOfEdges()-1;
    }

    uint64_t numberOfEdges() const{
        return gridGraph_.numberOfEdges();
    }
    uint64_t numberOfNodes() const{
        return gridGraph_.numberOfVertices();
    }

    NodeIter nodesBegin()const{
        return NodeIter(0);
    }
    NodeIter nodesEnd()const{
        return NodeIter(this->numberOfNodes());
    }
    EdgeIter edgesBegin()const{
        return EdgeIter(0);
    }
    EdgeIter edgesEnd()const{
        return EdgeIter(this->numberOfEdges());
    }

    AdjacencyIter adjacencyBegin(const int64_t node)const{
        return AdjacencyIter(gridGraph_.adjacenciesFromVertexBegin(node));
    }
    AdjacencyIter adjacencyEnd(const int64_t node)const{
        return AdjacencyIter(gridGraph_.adjacenciesFromVertexEnd(node));
    }
    AdjacencyIter adjacencyOutBegin(const int64_t node)const{
        return AdjacencyIter(gridGraph_.adjacenciesFromVertexBegin(node));
    }
     AdjacencyIter adjacencyOutEnd(const int64_t node)const{
        return AdjacencyIter(gridGraph_.adjacenciesFromVertexEnd(node));
    }


    // optional (with default impl in base)
    //std::pair<int64_t,int64_t> uv(const int64_t e)const;

    template<class F>
    void forEachEdge(F && f)const{
        for(uint64_t edge=0; edge< numberOfEdges(); ++edge){
            f(edge);
        }
    }

    template<class F>
    void forEachNode(F && f)const{
        for(uint64_t node=0; node< numberOfNodes(); ++node){
            f(node);
        }
    }


    // serialization de-serialization

    uint64_t serializationSize() const{
        return DIM + 1;
    }

    template<class ITER>
    void serialize(ITER iter) const{
        for(auto d=0; d<DIM; ++d){
            *iter = gridGraph_.shape(d);
            ++iter;
        }
        // simple nh?
        *iter = true;
        ++iter;
    }

    template<class ITER>
    void deserialize(ITER iter);


    /**
     * @brief convert an image with DIM dimension to an edge map
     * @details convert an image with DIM dimension to an edge map
     * by applying a binary functor to the values of a node map at
     * the endpoints of an edge.
     *
     * @param       image the  input image
     * @param       binaryFunctor a binary functor
     * @param[out]  the result edge map
     *
     * @return [description]
     */
    template<class IMAGE, class BINARY_FUNCTOR, class EDGE_MAP>
    void imageToEdgeMap(
        const IMAGE & image,
        BINARY_FUNCTOR binaryFunctor,
        EDGE_MAP & edgeMap
    )const{
        for(const auto edge : this->edges()){
            const auto uv = this->uv(edge);
            CoordinateType cU,cV;
            nodeToCoordinate(uv.first,  cU);
            nodeToCoordinate(uv.second, cV);
            const auto uVal = image(cU.asStdArray());
            const auto vVal = image(cU.asStdArray());
            edgeMap[edge] = binaryFunctor(uVal, vVal);
        }
    }


    // TODO parallelize
    /**
     * @brief convert an affinity map with DIM+1 dimension to an edge map
     * @details convert an affinity map with DIM+1 dimension to an edge map
     * by assigning the affinity values to corresponding affinity values
     *
     * @param       image the input affinities
     * @param[out]  the result edge map
     *
     * @return [description]
     */
    template<class AFFINITIES, class EDGE_MAP>
    void affinitiesToEdgeMap(const AFFINITIES & affinities,
                             EDGE_MAP & edgeMap) const {
        NIFTY_CHECK_OP(affinities.shape(0), ==, DIM, "wrong number of affinity channels")
        for(auto d=1; d<DIM+1; ++d){
            NIFTY_CHECK_OP(shape(d-1), ==, affinities.shape(d), "wrong shape")
        }

        typedef nifty::array::StaticArray<int64_t, DIM+1> AffinityCoordType;

        CoordinateType cU, cV;
        for(const auto edge : this->edges()){

            const auto uv = this->uv(edge);
            nodeToCoordinate(uv.first,  cU);
            nodeToCoordinate(uv.second, cV);

            // find the correct affinity edge
            AffinityCoordType affCoord;
            for(size_t d = 0; d < DIM; ++d) {
                auto diff = cU[d] - cV[d];
                if(diff == 0) {
                    affCoord[d + 1] = cU[d];
                }
                else {
                    // TODO max for different direction convention
                    affCoord[d + 1] = std::min(cU[d], cV[d]);
                    affCoord[0] = d;
                }
            }

            edgeMap[edge] = affinities(affCoord.asStdArray());
        }
    }


    template<class AFFINITIES, class EDGE_MAP, class ITER>
    void longRangeAffinitiesToLiftedEdges(const AFFINITIES & affinities,
                                          EDGE_MAP & edgeMap,
                                          ITER offsetsBegin,
                                          const int numberOfThreads=-1) const {
        //
        typedef nifty::array::StaticArray<int64_t, DIM+1> AffinityCoordType;
        for(auto d=1; d<DIM+1; ++d){
            NIFTY_CHECK_OP(shape(d-1), ==, affinities.shape(d), "wrong shape")
        }
        size_t affLen = affinities.shape(0);
        std::vector<std::vector<int>> offsets(offsetsBegin, offsetsBegin + affLen);

        AffinityCoordType affShape;
        affShape[0] = affLen;
        for(unsigned d = 0; d < DIM; ++d) {
            affShape[d + 1] = shape(d);
        }

        // TODO need to pre-allocate output or merge for parralel
        // parallel::ThreadPool tp(numberOfThreads);
        // tools::parallelForEachCoordinate(tp, affShape, [&](const int tId, const AffinityCoordType affCoord) {
        tools::forEachCoordinate(affShape, [&](const AffinityCoordType & affCoord) {
            const auto & offset = offsets[affCoord[0]];
            CoordinateType cU, cV;
            bool outOfRange = false;

            for(unsigned d = 0; d < DIM; ++d) {
                cU[d] = affCoord[d + 1];
                cV[d] = affCoord[d + 1] + offset[d];
                // range check
                if(cV[d] >= shape(d) || cV[d] < 0) {
                    outOfRange = true;
                    break;
                }
            }

            if(outOfRange) {
                return;
            }

            const size_t u = coordianteToNode(cU);
            const size_t v = coordianteToNode(cV);
            const size_t u_ = std::min(u, v);
            const size_t v_ = std::max(u, v);

            // debiggomg
            // if(u_ > numberOfNodes()) {
            //     std::cout << "u is out of range " << u_ << " " << numberOfNodes() << std::endl;
            //     throw std::runtime_error("Out of node range");
            // }
            // if(v_ > numberOfNodes()) {
            //     std::cout << "v is out of range " << v_ << " " << numberOfNodes() << std::endl;
            //     throw std::runtime_error("Out of node range");
            // }
            // TODO use xtensor::read for xt
            edgeMap.emplace(std::make_pair(u_, v_), affinities(affCoord.asStdArray()));
        });
    }


    /**
     * @brief convert an image with DIM dimension to an edge map
     * @details convert an image with DIM dimension to an edge map
     * by taking the values of the image at the
     * interpixel coordinates.
     * The shape of the image must be 2*shape-1
     *
     *
     * @param       image the  input image
     * @param       binaryFunctor a binary functor
     * @param[out]  the result edge map
     *
     * @return [description]
     */
    template<class IMAGE, class EDGE_MAP>
    void imageToInterpixelEdgeMap(
        const IMAGE & image,
        EDGE_MAP & edgeMap
    )const{

        for(auto d=0; d<DIM; ++d){
            NIFTY_CHECK_OP(shape(d)*2-1, ==, image.shape(d),
                "wrong shape foer image to interpixel edge map")
        }

        for(const auto edge : this->edges()){
            const auto uv = this->uv(edge);
            CoordinateType cU,cV;
            nodeToCoordinate(uv.first,  cU);
            nodeToCoordinate(uv.second, cV);
            const auto uVal = image(cU.asStdArray());
            cU += cV;
            edgeMap[edge] = image(cU.asStdArray());
        }
    }


    uint64_t shape(const std::size_t d)const{
        return gridGraph_.shape(DIM-1-d);
    }

    // COORDINATE RELATED
    CoordinateType nodeToCoordinate(const uint64_t node)const{
        CoordinateType ret;
        nodeToCoordinate(node, ret);
        return ret;
    }

    template<class NODE_COORDINATE>
    void nodeToCoordinate(
        const uint64_t node,
        NODE_COORDINATE & coordinate
    )const{
        AndresVertexCoordinate aCoordinate;
        gridGraph_.vertex(node, aCoordinate);
        for(auto d=0; d<DIM; ++d){
            coordinate[d] = aCoordinate[DIM-1-d];
        }
    }

    template<class NODE_COORDINATE>
    uint64_t coordianteToNode(const NODE_COORDINATE & coordinate)const{
        AndresVertexCoordinate aCoordinate;
        for(auto d=0; d<DIM; ++d){
            aCoordinate[DIM-1-d] = coordinate[d];
        }
        return gridGraph_.vertex(aCoordinate);
    }


private:
    andres::graph::GridGraph<DIM> gridGraph_;


};



} // namespace nifty::graph
} // namespace nifty
