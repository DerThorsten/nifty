
#pragma once

#include <vector>
#include <queue>
#include <algorithm> // max
#include <map>

#include "nifty/tools/runtime_check.hxx"
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


        template<class T,class U>
        inline T 
        replaceVals(const T & val, const U & replaceVal){
            if(std::isfinite(val))
                return val;
            else
                return replaceVal;
        }



        struct Accumulator{

            typedef vigra::TinyVector<int, 3> VigraCoordType;
            typedef array::StaticArray<int64_t,3> CoordType;
            typedef vigra::acc::UserRangeHistogram<40>            SomeHistogram;   //binCount set at compile time
            typedef vigra::acc::StandardQuantiles<SomeHistogram > Quantiles;

            typedef vigra::acc::AccumulatorChain<
                float, 
                vigra::acc::Select<
                    vigra::acc::Mean, 
                    vigra::acc::Variance, 
                    vigra::acc::Skewness, 
                    vigra::acc::Kurtosis, 
                    SomeHistogram,
                    Quantiles
                > 
            > DataAccChain;


            typedef vigra::acc::AccumulatorChain<
                VigraCoordType, 
                vigra::acc::Select<
                    vigra::acc::Mean
                > 
            > CoordAccChain;


            Accumulator(
                const uint8_t numberOfChannels
            )
            :   numberOfChannels_(numberOfChannels),
                accChain_(numberOfChannels)
            {
                vigra::HistogramOptions hOpts;
                hOpts = hOpts.setMinMax(0.0, 1.0); 

                for(auto c=0; c<numberOfChannels_; ++c){
                    accChain_[c].setHistogramOptions(hOpts);
                }

            }

        

            void merge(const Accumulator & other){
                for(auto c=0; c<numberOfChannels_; ++c){
                    accChain_[c].merge(other.accChain_[c]);
                }
            }

            void accumulatePass(const CoordType & coord, const float * dataPtr, const size_t pass){
                for(auto c=0; c<numberOfChannels_; ++c){
                    accChain_[c].updatePassN(dataPtr[c], pass);
                }
                if(pass <= coordAccChain_.passesRequired()){
                    VigraCoordType vc;
                    for(auto c=0; c<3; ++c)
                        vc[c] = coord[c];
                    coordAccChain_.updatePassN(vc ,pass);
                }
            }   



            uint8_t numberOfChannels_;
            std::vector<DataAccChain> accChain_;
            CoordAccChain coordAccChain_;
        };




        class BlockData{
        public:
            typedef tools::Blocking<3> BlockingType;    
            typedef array::StaticArray<int64_t,3> CoordType;

            typedef vigra::acc::UserRangeHistogram<40>            SomeHistogram;   //binCount set at compile time
            typedef vigra::acc::StandardQuantiles<SomeHistogram > Quantiles;

            BlockData(
                BlockingType & blocking,
                const size_t blockIndex,
                const size_t numberOfChannels, 
                const size_t numberOfBins
            )
            :   blocking_(blocking),
                blockIndex_(blockIndex),
                numberOfChannels_(numberOfChannels),
                numberOfBins_(numberOfBins)
            {
                

            }
  
            void accumulate( 
                marray::View<uint32_t>  & labels,
                marray::View<float> &     data
            ){

                NIFTY_CHECK_OP(data.shape(3),==,numberOfChannels_,"data has wrong number of Channels");


                const auto coreBlock = blocking_.getBlock(blockIndex_);
                const auto shape = coreBlock.shape();
                const auto & blockBegin = coreBlock.begin();

                for(auto p=1; p<=2; ++p){
                    nifty::tools::forEachCoordinate(shape,[&](const CoordType & coord){

                        const auto lU = labels(coord.asStdArray());
                        const auto pU = &data(coord[0],coord[1],coord[2],0);

                        {
                            //auto iter = nodes_.emplace(lU, Accumulator(numberOfChannels_)).first;
                            //iter->second.accumulatePass(coord + blockBegin, pU, p);
                        }

                        for(size_t axis=0; axis<3; ++axis){
                            auto coord2 = coord;
                            ++coord2[axis];
                            if(coord2[axis] < labels.shape(axis)){
                                const auto lV = labels(coord2.asStdArray());

                                if(lU != lV){

                                    const auto pV = &data(coord2[0],coord2[1],coord2[2],0);

                                    auto iter = edges_.emplace(Edge(lU,lV), Accumulator(numberOfChannels_)).first;
                                    iter->second.accumulatePass(coord +blockBegin, pU, p);
                                    iter->second.accumulatePass(coord2+blockBegin, pV, p);
                                }
                            }
                        }
                    });
                }
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

                //for(auto & kv : other.nodes_){
                //    const auto p = nodes_.insert(kv);
                //    const auto iter = p.first;
                //    const auto added = p.second;
                //    if(!added){
                //        iter->second.merge(kv.second);
                //    }
                //}
            }

        public:

            size_t numberOfFeatures()const{
                return numberOfChannels_*11;
            }


            void extractFeatures(
                const Edge & edge,
                marray::View<float> & out
            ){

                auto fRes = edges_.find(edge);
                NIFTY_CHECK(fRes!=edges_.end(),"");

                auto & acc = fRes->second;

                auto fIndex = 0;


                for(auto c=0; c<int(numberOfChannels_); ++c){

                    const auto & chain = acc.accChain_[c];
                    const auto mean = vigra::acc::get<vigra::acc::Mean>(chain);
                    const auto quantiles = vigra::acc::get<Quantiles>(chain);

                    out(fIndex+0) = replaceVals(mean,     0.0);
                    out(fIndex+1) = replaceVals(vigra::acc::get<vigra::acc::Variance>(chain), 0.0);
                    out(fIndex+2) = replaceVals(vigra::acc::get<vigra::acc::Skewness>(chain), 0.0);
                    out(fIndex+3) = replaceVals(vigra::acc::get<vigra::acc::Kurtosis>(chain), 0.0);
                    for(auto qi=0; qi<7; ++qi){
                        out(fIndex+4+qi) = replaceVals(quantiles[qi], mean);
                    }

                    fIndex += 11;
                }
            }




            void extractFeatures(
                marray::View<float> & out
            ){
                auto c = 0;
                for(const auto kv: edges_){
                    const auto edge = kv.first;
                    auto outSub = out.boundView(0, c);
                    extractFeatures(edge, outSub);
                    ++c;
                }
            }


            size_t numberOfEdges()const{
                return edges_.size();
            }

            void uvIds(marray::View<uint32_t> uv)const{
                auto c=0;
                for(const auto & kv: edges_){
                    const auto & edge = kv.first;
                    uv(c, 0) = edge.u;
                    uv(c, 1) = edge.v;
                    ++c;
                }
            }


        private:
            BlockingType blocking_;
            size_t blockIndex_;
            uint8_t numberOfChannels_;
            uint8_t numberOfBins_;
            std::map<Edge,     Accumulator> edges_;
            std::map<uint32_t, Accumulator> nodes_;
            //std::unordered_map<Edge, PerEdgeData, EdgeHash> edges_;





        };
    
      
}
}
}