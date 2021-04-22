#pragma once

#include <cstddef>
#include <random>
#include <boost/iterator/counting_iterator.hpp>
#include <boost/iterator/transform_iterator.hpp>
#include <boost/iterator/iterator_facade.hpp>

#include "nifty/graph/detail/andres/grid-graph.hxx"

#include "nifty/tools/runtime_check.hxx"
#include "nifty/tools/for_each_coordinate.hxx"
#include "nifty/graph/undirected_graph_base.hxx"
#include "nifty/graph/detail/adjacency.hxx"
#include "nifty/graph/graph_tags.hxx"
#include "nifty/parallel/threadpool.hxx"
#include "nifty/array/arithmetic_array.hxx"
#include "nifty/xtensor/xtensor.hxx"


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

    //
    // distance functions along channel dimension
    //

    template<unsigned DIM, class IMAGE>
    double l2Distance(
        const IMAGE & image,
        const unsigned nChannels,
        std::array<int64_t, DIM+1> coordU,
        std::array<int64_t, DIM+1> coordV
    ) {
        double val = 0;
        for(unsigned c = 0; c < nChannels; ++c) {
            coordU[0] = c;
            coordV[0] = c;
            const auto uVal = xtensor::read(image, coordU);
            const auto vVal = xtensor::read(image, coordV);
            val += (uVal - vVal) * (uVal - vVal);
        }
        return std::sqrt(val);
    }

    template<unsigned DIM, class IMAGE>
    double l1Distance(
        const IMAGE & image,
        const unsigned nChannels,
        std::array<int64_t, DIM+1> coordU,
        std::array<int64_t, DIM+1> coordV
    ) {
        double val = 0;
        for(unsigned c = 0; c < nChannels; ++c) {
            coordU[0] = c;
            coordV[0] = c;
            const auto uVal = xtensor::read(image, coordU);
            const auto vVal = xtensor::read(image, coordV);
            val += std::abs(uVal - vVal);
        }
        return val;
    }

    template<unsigned DIM, class IMAGE>
    double cosineDistance(
        const IMAGE & image,
        const unsigned nChannels,
        std::array<int64_t, DIM+1> coordU,
        std::array<int64_t, DIM+1> coordV
    ) {
        double val = 0;
        double normU = 0;
        double normV = 0;
        for(unsigned c = 0; c < nChannels; ++c) {
            coordU[0] = c;
            coordV[0] = c;
            const auto uVal = xtensor::read(image, coordU);
            const auto vVal = xtensor::read(image, coordV);
            val += uVal * vVal;
            normU += uVal * uVal;
            normV += vVal * vVal;
        }
        normU = std::sqrt(normU);
        normV = std::sqrt(normV);
        val = 1. - (val / normU / normV);
        return val;
    }


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
            CoordinateType cU, cV;
            nodeToCoordinate(uv.first,  cU);
            nodeToCoordinate(uv.second, cV);
            const auto uVal = xtensor::read(image, cU.asStdArray());
            const auto vVal = xtensor::read(image, cV.asStdArray());

            edgeMap[edge] = binaryFunctor(uVal, vVal);
        }
    }


    /**
     * @brief convert an image with DIM + 1 dimension to an edge map
     * @details convert an image with DIM + 1 dimension to an edge map
     * by computing the distance between the values of a node map at
     * the endpoints of an edge.
     *
     * @param       image the  input image
     * @param       distance   the distance (l1, l2 or cosine)
     * @param[out]  the result edge map
     *
     * @return [description]
     */
    template<class IMAGE, class EDGE_MAP>
    void imageWithChannelsToEdgeMap(
        const IMAGE & image,
        const std::string & distance,
        EDGE_MAP & edgeMap
    ) const {
        if(distance == "l1") {
            imageWithChannelsToEdgeMapImpl(image, edgeMap, detail_graph::l1Distance<DIM, IMAGE>);
        } else if(distance == "l2") {
            imageWithChannelsToEdgeMapImpl(image, edgeMap, detail_graph::l2Distance<DIM, IMAGE>);
        } else if(distance == "cosine") {
            imageWithChannelsToEdgeMapImpl(image, edgeMap, detail_graph::cosineDistance<DIM, IMAGE>);
        } else {
            throw std::runtime_error("Invalid distance.");
        }
    }

    template<class IMAGE, class EDGES, class EDGE_MAP>
    std::size_t imageWithChannelsToEdgeMapWithOffsets(
        const IMAGE & image,
        const std::string & distance,
        const std::vector<std::vector<int>> & offsets,
        EDGES & edges,
        EDGE_MAP & edgeMap
    ) const {

        std::size_t edgeId = 0;
        auto sampler = [](nifty::array::StaticArray<int64_t, DIM> & coord){return true;};

        if(distance == "l1") {
            edgeId = imageWithChannelsToEdgeMapWithOffsetsImpl(image, offsets,
                                                               detail_graph::l1Distance<DIM, IMAGE>,
                                                               sampler,
                                                               edges, edgeMap);
        } else if(distance == "l2") {
            edgeId = imageWithChannelsToEdgeMapWithOffsetsImpl(image, offsets,
                                                               detail_graph::l2Distance<DIM, IMAGE>,
                                                               sampler,
                                                               edges, edgeMap);
        } else if(distance == "cosine") {
            edgeId = imageWithChannelsToEdgeMapWithOffsetsImpl(image, offsets,
                                                               detail_graph::cosineDistance<DIM, IMAGE>,
                                                               sampler,
                                                               edges, edgeMap);
        } else {
            throw std::runtime_error("Invalid distance.");
        }

        return edgeId;
    }

    template<class IMAGE, class EDGES, class EDGE_MAP>
    std::size_t imageWithChannelsToEdgeMapWithOffsets(
        const IMAGE & image,
        const std::string & distance,
        const std::vector<std::vector<int>> & offsets,
        const std::vector<int> & strides,
        EDGES & edges,
        EDGE_MAP & edgeMap
    ) const {

        std::size_t edgeId = 0;

        auto sampler = [&strides](nifty::array::StaticArray<int64_t, DIM> & coord){
            bool inStride = true;
            for(unsigned d = 0; d < DIM; ++d) {
                if(coord[d] % strides[d] != 0) {
                    inStride = false;
                    break;
                }
            }
            return inStride;
        };

        if(distance == "l1") {
            edgeId = imageWithChannelsToEdgeMapWithOffsetsImpl(image, offsets,
                                                               detail_graph::l1Distance<DIM, IMAGE>,
                                                               sampler,
                                                               edges, edgeMap);
        } else if(distance == "l2") {
            edgeId = imageWithChannelsToEdgeMapWithOffsetsImpl(image, offsets,
                                                               detail_graph::l2Distance<DIM, IMAGE>,
                                                               sampler,
                                                               edges, edgeMap);
        } else if(distance == "cosine") {
            edgeId = imageWithChannelsToEdgeMapWithOffsetsImpl(image, offsets,
                                                               detail_graph::cosineDistance<DIM, IMAGE>,
                                                               sampler,
                                                               edges, edgeMap);
        } else {
            throw std::runtime_error("Invalid distance.");
        }

        return edgeId;
    }

    template<class IMAGE, class EDGES, class EDGE_MAP>
    std::size_t imageWithChannelsToEdgeMapWithOffsets(
        const IMAGE & image,
        const std::string & distance,
        const std::vector<std::vector<int>> & offsets,
        const double sampleProbability,
        EDGES & edges,
        EDGE_MAP & edgeMap
    ) const {

        std::size_t edgeId = 0;

        std::default_random_engine gen;
        std::uniform_real_distribution<double> distr(0., 1.);
        auto draw = std::bind(distr, gen);

        auto sampler = [&draw, sampleProbability](nifty::array::StaticArray<int64_t, DIM> & coord){
            return draw() < sampleProbability;
        };

        if(distance == "l1") {
            edgeId = imageWithChannelsToEdgeMapWithOffsetsImpl(image, offsets,
                                                               detail_graph::l1Distance<DIM, IMAGE>,
                                                               sampler,
                                                               edges, edgeMap);
        } else if(distance == "l2") {
            edgeId = imageWithChannelsToEdgeMapWithOffsetsImpl(image, offsets,
                                                               detail_graph::l2Distance<DIM, IMAGE>,
                                                               sampler,
                                                               edges, edgeMap);
        } else if(distance == "cosine") {
            edgeId = imageWithChannelsToEdgeMapWithOffsetsImpl(image, offsets,
                                                               detail_graph::cosineDistance<DIM, IMAGE>,
                                                               sampler,
                                                               edges, edgeMap);
        } else {
            throw std::runtime_error("Invalid distance.");
        }

        return edgeId;
    }

    /**
     * @brief convert an affinity map with DIM+1 dimension to an edge map
     * @details convert an affinity map with DIM+1 dimension to an edge map
     * by assigning the affinity values to corresponding affinity values
     *
     * @param       image the input affinities
     * @param       whether affinities encode connectivity yo lower or upper pixel
     * @param[out]  the result edge map
     *
     * @return [description]
     */
    template<class AFFINITIES, class EDGE_MAP>
    void affinitiesToEdgeMap(const AFFINITIES & affinities,
                             EDGE_MAP & edgeMap,
                             const bool toLower=true) const {
        NIFTY_CHECK_OP(affinities.shape()[0], ==, DIM, "wrong number of affinity channels")
        for(auto d=1; d<DIM+1; ++d){
            NIFTY_CHECK_OP(shape(d-1), ==, affinities.shape()[d], "wrong shape")
        }

        typedef nifty::array::StaticArray<int64_t, DIM+1> AffinityCoordType;

        CoordinateType cU, cV;
        for(const auto edge : this->edges()){

            const auto uv = this->uv(edge);
            nodeToCoordinate(uv.first,  cU);
            nodeToCoordinate(uv.second, cV);

            // find the correct affinity edge
            AffinityCoordType affCoord;
            for(std::size_t d = 0; d < DIM; ++d) {
                auto diff = cU[d] - cV[d];
                if(diff == 0) {
                    affCoord[d + 1] = cU[d];
                }
                else {
                    affCoord[d + 1] = toLower ? (cU[d] < cV[d] ? cU[d] : cV[d]) : (cU[d] < cV[d] ? cV[d] : cU[d]);
                    affCoord[0] = d;
                }
            }

            edgeMap(edge) = xtensor::read(affinities, affCoord.asStdArray());
        }
    }


    template<class AFFINITIES, class EDGES, class EDGE_MAP>
    std::size_t affinitiesToEdgeMapWithOffsets(const AFFINITIES & affinities,
                                               const std::vector<std::vector<int>> & offsets,
                                               EDGES & edges,
                                               EDGE_MAP & edgeMap) const {
        auto sampler = [](nifty::array::StaticArray<int64_t, DIM> & coord){return true;};
        const std::size_t edgeId = affinitiesToEdgeMapWithOffsetsImpl(
            affinities, offsets,
            edges, edgeMap, sampler
        );
        return edgeId;
    }


    template<class AFFINITIES, class EDGES, class EDGE_MAP>
    std::size_t affinitiesToEdgeMapWithOffsets(const AFFINITIES & affinities,
                                               const std::vector<std::vector<int>> & offsets,
                                               const std::vector<int> & strides,
                                               EDGES & edges,
                                               EDGE_MAP & edgeMap) const {

        auto sampler = [&strides](nifty::array::StaticArray<int64_t, DIM> & coord){
            bool inStride = true;
            for(unsigned d = 0; d < DIM; ++d) {
                if(coord[d] % strides[d] != 0) {
                    inStride = false;
                    break;
                }
            }
            return inStride;
        };

        const std::size_t edgeId = affinitiesToEdgeMapWithOffsetsImpl(
            affinities, offsets,
            edges, edgeMap, sampler
        );
        return edgeId;
    }


    template<class AFFINITIES, class EDGES, class EDGE_MAP>
    std::size_t affinitiesToEdgeMapWithOffsets(const AFFINITIES & affinities,
                                               const std::vector<std::vector<int>> & offsets,
                                               const double sampleProbability,
                                               EDGES & edges,
                                               EDGE_MAP & edgeMap) const {

        std::default_random_engine gen;
        std::uniform_real_distribution<double> distr(0., 1.);
        auto draw = std::bind(distr, gen);

        auto sampler = [&draw, sampleProbability](nifty::array::StaticArray<int64_t, DIM> & coord){
            return draw() < sampleProbability;
        };

        const std::size_t edgeId = affinitiesToEdgeMapWithOffsetsImpl(
            affinities, offsets,
            edges, edgeMap, sampler
        );
        return edgeId;
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
            NIFTY_CHECK_OP(shape(d)*2 - 1, ==, image.shape()[d],
                "wrong shape foer image to interpixel edge map")
        }

        for(const auto edge : this->edges()){
            const auto uv = this->uv(edge);
            CoordinateType cU,cV;
            nodeToCoordinate(uv.first,  cU);
            nodeToCoordinate(uv.second, cV);
            // FIXME I don't understand what is going on here ?!
            // But doesn't look right
            const auto uVal = xtensor::read(image, cU.asStdArray());
            cU += cV;
            edgeMap(edge) = xtensor::read(image, cU.asStdArray());
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
    uint64_t coordinateToNode(const NODE_COORDINATE & coordinate)const{
        AndresVertexCoordinate aCoordinate;
        for(auto d=0; d<DIM; ++d){
            aCoordinate[DIM-1-d] = coordinate[d];
        }
        return gridGraph_.vertex(aCoordinate);
    }


private:

    //
    // implementation of imageWithChannelsToEdgeMap
    //

    template<class IMAGE, class EDGE_MAP, class F>
    void imageWithChannelsToEdgeMapImpl(
        const IMAGE & image,
        EDGE_MAP & edgeMap,
        F & dist
    ) const {
        const int nChannels = image.shape()[0];
        std::array<int64_t, DIM+1> coordU;
        std::array<int64_t, DIM+1> coordV;
        for(const auto edge : this->edges()){
            const auto uv = this->uv(edge);
            CoordinateType cU, cV;
            nodeToCoordinate(uv.first,  cU);
            nodeToCoordinate(uv.second, cV);
            for(unsigned d = 0; d < DIM; ++d) {
                coordU[d + 1] = cU[d];
                coordV[d + 1] = cV[d];
            }
            edgeMap(edge) = dist(image, nChannels, coordU, coordV);
        }
    }

    //
    // implementations of imageToEdgeMapWithChannelsWithOffsets
    //

    template<class IMAGE, class DIST, class SAMPLER,
             class EDGES, class EDGE_MAP>
    std::size_t imageWithChannelsToEdgeMapWithOffsetsImpl(
        const IMAGE & image,
        const std::vector<std::vector<int>> & offsets,
        DIST & dist,
        SAMPLER & sampler,
        EDGES & edges,
        EDGE_MAP & edgeMap
    ) const {
        typedef typename EDGE_MAP::value_type edgeValType;
        typedef nifty::array::StaticArray<int64_t, DIM> CoordType;
        typedef std::array<int64_t, DIM+1> CoordWithChannelType;

        const unsigned nChannels = image.shape()[0];
        const std::size_t nOffsets = offsets.size();

        CoordType coordU;
        CoordType coordV;
        CoordWithChannelType coordUC;
        CoordWithChannelType coordVC;

        for(auto d=1; d<DIM+1; ++d){
            NIFTY_CHECK_OP(shape(d-1), ==, image.shape()[d], "wrong shape")
        }

        CoordType shape_;
        for(unsigned d = 0; d < DIM; ++d) {
            shape_[d] = shape(d);
        }

        std::size_t edgeId = 0;
        tools::forEachCoordinate(shape_, [&](const CoordType & coord) {
            for(int offsetId = 0; offsetId < nOffsets; ++offsetId) {
                const auto & offset = offsets[offsetId];

                // initialise the coordinates w/o and w/ channel
                for(unsigned d = 0; d < DIM; ++d) {
                    coordU[d] = coord[d];
                    coordV[d] = coord[d] + offset[d];
                    // range check
                    if(coordV[d] >= shape(d) || coordV[d] < 0) {
                        return;
                    }
                    coordUC[d+1] = coordU[d];
                    coordVC[d+1] = coordV[d];
                }

                if(!sampler(coordU)) {
                    return;
                }

                const std::size_t u = coordinateToNode(coordU);
                const std::size_t v = coordinateToNode(coordV);

                edgeMap(edgeId) = dist(image, nChannels, coordUC, coordVC);
                edges(edgeId, 0) = std::min(u, v);
                edges(edgeId, 1) = std::max(u, v);
                ++edgeId;
            }

        });
        return edgeId;
    }

    //
    // implementation of affinitiesToEdgeMapWithOffsets
    //

    template<class AFFINITIES, class EDGES, class EDGE_MAP, class F>
    std::size_t affinitiesToEdgeMapWithOffsetsImpl(const AFFINITIES & affinities,
                                                   const std::vector<std::vector<int>> & offsets,
                                                   EDGES & edges,
                                                   EDGE_MAP & edgeMap,
                                                   F & sampler) const {
        typedef nifty::array::StaticArray<int64_t, DIM+1> AffinityCoordType;
        for(auto d=1; d<DIM+1; ++d){
            NIFTY_CHECK_OP(shape(d-1), ==, affinities.shape()[d], "wrong shape")
        }
        std::size_t affLen = affinities.shape()[0];

        AffinityCoordType affShape;
        affShape[0] = affLen;
        for(unsigned d = 0; d < DIM; ++d) {
            affShape[d + 1] = shape(d);
        }

        std::size_t edgeId = 0;
        tools::forEachCoordinate(affShape, [&](const AffinityCoordType & affCoord) {
            const auto & offset = offsets[affCoord[0]];
            CoordinateType cU, cV;

            for(unsigned d = 0; d < DIM; ++d) {
                cU[d] = affCoord[d + 1];
                cV[d] = affCoord[d + 1] + offset[d];
                // range check
                if(cV[d] >= shape(d) || cV[d] < 0) {
                    return;
                }
            }

            // check if we keep this edge
            if(!sampler(cU)) {
                return;
            }

            const std::size_t u = coordinateToNode(cU);
            const std::size_t v = coordinateToNode(cV);

            edgeMap(edgeId) = xtensor::read(affinities, affCoord.asStdArray());
            edges(edgeId, 0) = std::min(u, v);
            edges(edgeId, 1) = std::max(u, v);
            ++edgeId;

        });
        return edgeId;
    }


private:
    andres::graph::GridGraph<DIM> gridGraph_;
};



} // namespace nifty::graph
} // namespace nifty
