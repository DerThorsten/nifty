#pragma once

#include "nifty/xtensor/xtensor.hxx"

namespace nifty{
namespace graph{
namespace graph_maps{




template<class G, class T>
struct NodeMap : public std::vector<T>{
    NodeMap( const G & g, const T & val)
    :   std::vector<T>( g.nodeIdUpperBound()+1, val){
    }
    NodeMap( const G & g)
    :   std::vector<T>( g.nodeIdUpperBound()+1){
    }
    NodeMap( )
    :   std::vector<T>( ){
    }

    // graph has been modified
    void insertedNodes(const uint64_t nodeId, const T & insertValue = T()){
        if(nodeId == this->size()){
            this->push_back(insertValue);
        }
        else if(nodeId > this->size()){
            this->resize(nodeId + 1, insertValue);
        }
    }
};



/**
 * @brief Multiband node map
 * @details Sometimes we need to hold not a single scalar,
 * but a fixed length vector for each node.
 * The return type of operator[] is a tiny proxy
 * object holding the vector.
 *
 *
 * @tparam G GraphType4
 * @tparam T ValueType
 */
template<class G, class T>
struct MultibandNodeMap
{

public:
    class Proxy{
    public:
        Proxy(T * ptr, const std::size_t size)
        :   ptr_(ptr),
            size_(size){
        }
        const T & operator[](const std::size_t i)const{
            return ptr_[i];
        }
        T & operator[](const std::size_t i){
            return ptr_[i];
        }
    private:
        T * ptr_;
        std::size_t size_;
    };

    class ConstProxy{
    public:
        ConstProxy(const T * ptr, const std::size_t size)
        :   ptr_(ptr),
            size_(size){
        }
        const T & operator[](const std::size_t i)const{
            return ptr_[i];
        }
        const T & operator[](const std::size_t i){
            return ptr_[i];
        }
    private:
        const T * ptr_;
        std::size_t size_;
    };


    MultibandNodeMap( const G & g, const std::size_t nChannels)
    :   nChannels_(nChannels),
        data_((g.nodeIdUpperBound()+1)*nChannels){
    }
    MultibandNodeMap( const G & g, const std::size_t nChannels, const T & val)
    :   nChannels_(nChannels),
        data_((g.nodeIdUpperBound()+1)*nChannels, val){
    }

    Proxy operator[](const uint64_t nodeIndex){
        return Proxy(data_.data() + nodeIndex*nChannels_, nChannels_);
    }
    ConstProxy operator[](const uint64_t nodeIndex)const{
        return ConstProxy(data_.data() + nodeIndex*nChannels_, nChannels_);
    }
    const std::size_t numberOfChannels()const{
        return nChannels_;
    }
private:
    std::vector<T> data_;
    std::size_t nChannels_;
};



template<class ARRAY>
struct MultibandArrayViewNodeMap
{

public:
    typedef typename ARRAY::value_type value_type;
    typedef typename ARRAY::reference reference;
    typedef typename ARRAY::const_reference const_reference;

    class Proxy{
    public:
        Proxy(ARRAY & array, const uint64_t node)
        :   array_(&array),
            node_(node){
        }
        const_reference operator[](const std::size_t i)const{
            return array_->operator()(node_, i);
        }
        reference operator[](const std::size_t i){
            return array_->operator()(node_, i);
        }
        std::size_t size()const{
            return array_->shape(1);
        }
    private:
        uint64_t node_;
        ARRAY * array_;
    };

    class ConstProxy{
    public:
        ConstProxy(const ARRAY & array, const uint64_t node)
        :   array_(&array),
            node_(node){
        }
        const_reference operator[](const std::size_t i)const{
            return array_->operator()(node_, i);
        }
        const_reference operator[](const std::size_t i){
            return array_->operator()(node_, i);
        }
        std::size_t size()const{
            return array_->shape(1);
        }
    private:
        uint64_t node_;
        const ARRAY * array_;
    };


    MultibandArrayViewNodeMap(const ARRAY & array)
    :   nChannels_(array.shape()[1]),
        array_(array){
    }


    Proxy operator[](const uint64_t nodeIndex){
        return Proxy(array_, nodeIndex);
    }

    ConstProxy operator[](const uint64_t nodeIndex)const{
        return ConstProxy(array_, nodeIndex);
    }

    const std::size_t numberOfChannels()const{
        return nChannels_;
    }
private:
    const ARRAY & array_;
    std::size_t nChannels_;
};




template<class G, class T>
struct EdgeMap : public std::vector<T>{
    EdgeMap( const G & g, const T & val)
    :   std::vector<T>( g.edgeIdUpperBound()+1, val){
    }

    EdgeMap( const G & g)
    :   std::vector<T>( g.edgeIdUpperBound()+1){
    }

    EdgeMap( )
    :   std::vector<T>( ){
    }

    // graph has been modified
    void insertedEdges(const uint64_t edgeId, const T & insertValue = T()){
        if(edgeId == this->size()){
            this->push_back(insertValue);
        }
        else if(edgeId > this->size()){
            this->resize(edgeId + 1, insertValue);
        }
    }
};


} // namespace nifty::graph::graph_maps
} // namespace nifty::graph
} // namespace nifty

