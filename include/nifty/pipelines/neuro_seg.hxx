
#pragma once

#include <vector>
#include <queue>
#include <algorithm> // max
#include <map>

#include "nifty/marray/marray.hxx"
#include "nifty/tools/for_each_coordinate.hxx"
#include "nifty/tools/blocking.hxx"

#include <unordered_map>
#include <map>

#include "vigra/accumulator.hxx"


namespace nifty{
namespace pipelines{
namespace neuro_seg{




        struct Edge{
            Edge(const uint32_t uu, const uint32_t vv)
            :   u(std::min(uu,vv)),
                v(std::max(uu,vv)){

            }

            bool operator < (const Edge & other)const{
                if(u < other.u)
                    return true;
                else if(u > other.u)
                    return false;
                else
                    return v < other.v;
            }

            bool operator == (const Edge & other)const{
                return u == other.u && v == other.v;
            }
            uint32_t u,v;
        };

        struct EdgeHash {
        public:
          std::size_t operator()(const Edge &e) const
          {
            return std::hash<uint32_t>()(e.u) ^ std::hash<uint32_t>()(e.v);
          }
        };




        class PerEdgeData{
        public:
            


            
            PerEdgeData()
            :   count_(0){

            }

            void merge(const PerEdgeData & other){

            }

            void accumulate(){
                ++count_;
            }

        private:
            uint32_t count_;
        };

        struct PerNodeData{
        };


        class BlockData{
        public:
            typedef tools::Blocking<3> BlockingType;    
            typedef array::StaticArray<int64_t,3> CoordType;

            BlockData(
                BlockingType & blocking,
                const size_t blockIndex
            )
            :   blocking_(blocking),
                blockIndex_(blockIndex)
            {
                

            }
  
            void accumulate( 
                marray::View<uint32_t>  & labels
                //,
                //,array::View<uint8_t>  &  rawData,
                //marray::View<float> &     pmaps
            ){
                const auto coreBlock = blocking_.getBlock(blockIndex_);
                const auto shape = coreBlock.shape();


                nifty::tools::forEachCoordinate(shape,[&](const CoordType & coord){
                    const auto lU = labels(coord.asStdArray());
                    for(size_t axis=0; axis<3; ++axis){
                        auto coord2 = coord;
                        ++coord2[axis];
                        if(coord2[axis] < labels.shape(axis)){
                            const auto lV = labels(coord2.asStdArray());
                            if(lU != lV){
                                auto iter = edges_.emplace(Edge(lU,lV), PerEdgeData()).first;
                                iter->second.accumulate();
                            }
                        }
                    }
                });
            }


            void merge(const BlockData & other){


                for(auto & kv : other.edges_){
                    const auto p = edges_.insert(kv);
                    const auto iter = p.first;
                    const auto added = p.second;
                    if(!added){
                        iter->second.merge(kv.second);
                    }
                }
            }

        private:
            BlockingType blocking_;
            size_t blockIndex_;
            std::map<Edge, PerEdgeData> edges_;
            //std::unordered_map<Edge, PerEdgeData, EdgeHash> edges_;


        };
    


}
}
}