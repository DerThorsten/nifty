
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
#include <cmath>

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

                    vigra::acc::Count,
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
                            auto iter = nodes_.emplace(lU, Accumulator(numberOfChannels_)).first;
                            iter->second.accumulatePass(coord + blockBegin, pU, p);
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

                for(auto & kv : other.nodes_){
                    const auto p = nodes_.insert(kv);
                    const auto iter = p.first;
                    const auto added = p.second;
                    if(!added){
                        iter->second.merge(kv.second);
                    }
                }
            }

        public:

            size_t numberOfFeatures()const{
                return numberOfChannels_*(11*9) + 22;
            }


            void extractFeatures(
                const Edge & edge,
                marray::View<float> & out
            ){

                using namespace vigra::acc;
                auto fRes = edges_.find(edge);

                NIFTY_CHECK(fRes!=edges_.end(),"");

                auto & accEdge = fRes->second;
                const auto & accU = nodes_.find(edge.u)->second;
                const auto & accV = nodes_.find(edge.v)->second;

                

                auto fIndex = 0;


                auto wardness = [](const double cU, const double cV, const double beta){

                    const auto rU = std::pow(1.0/cU, beta); 
                    const auto rV = std::pow(1.0/cV, beta); 

                    return 1.0/(rU + rV);

                };

                // geometric data
                {
                    const auto & chainE = accEdge.coordAccChain_;
                    const auto & chainU = accU.coordAccChain_;
                    const auto & chainV = accV.coordAccChain_;

                    const double countE = get<Count>(chainE);
                    const double countU = get<Count>(chainU);
                    const double countV = get<Count>(chainV);

                    const auto centerE = get<Mean>(chainE);
                    const auto centerU = get<Mean>(chainU);
                    const auto centerV = get<Mean>(chainV);

                    // standart
                    out(fIndex + 0) = countE;
                    out(fIndex + 1) = std::abs(countU - countV);
                    out(fIndex + 2) = std::min(countU,  countV);
                    out(fIndex + 3) = std::max(countU,  countV);
                    out(fIndex + 4) =         (countU + countV);

                    // distance
                    const auto dUV = vigra::norm(centerU - centerV);
                    const auto dEU = vigra::norm(centerE - centerU);
                    const auto dEV = vigra::norm(centerE - centerV);


                    out(fIndex + 5) = dUV;
                    out(fIndex + 6) = std::min(dEU, dEV);
                    out(fIndex + 7) = std::max(dEU, dEV);
                    out(fIndex + 8) =         (dEU + dEV);

                    out(fIndex + 9)  = dUV - std::min(dEU, dEV);
                    out(fIndex + 10) = dUV - std::max(dEU, dEV);
                    out(fIndex + 11) = dUV -         (dEU + dEV);

                    out(fIndex + 12)  = dUV / (std::min(dEU, dEV)  + 0.01);
                    out(fIndex + 13) = dUV / (std::max(dEU, dEV)  + 0.01);
                    out(fIndex + 14) = dUV / (        (dEU + dEV) + 0.01);



                    // 'wardness'
                    out(fIndex + 15) = wardness(countU, countV, 0.1);
                    out(fIndex + 16) = wardness(countU, countV, 0.2);
                    out(fIndex + 17) = wardness(countU, countV, 0.4);
                    out(fIndex + 18) = wardness(countU, countV, 0.8);

                    // 'edge' / 'node' relation
                    const double nCountE = std::sqrt(countE);
                    const double nCountU = std::cbrt(countU);
                    const double nCountV = std::cbrt(countV);

                    out(fIndex + 19) = nCountE / std::min(nCountU,  nCountV);
                    out(fIndex + 20) = nCountE / std::max(nCountU,  nCountV);
                    out(fIndex + 21) = nCountE /         (nCountU + nCountV);
                }



                // channel data
                {
                    array::StaticArray<float, 11> fE,fU,fV;
                    for(auto c=0; c<int(numberOfChannels_); ++c){

                        const auto & chainE = accEdge.accChain_[c];
                        const auto & chainU = accU.accChain_[c];
                        const auto & chainV = accV.accChain_[c];


                        // get values
                        this->extract(chainE, fE);
                        this->extract(chainE, fU);
                        this->extract(chainE, fV);


                        for(auto i=0; i<11; ++i){

                            const auto dEU = std::abs(fE[i] - fU[i]);
                            const auto dEV = std::abs(fE[i] - fV[i]);


                            out(fIndex + 0) =         (fE[i]);          
                            out(fIndex + 1) = std::abs(fU[i] - fV[i]);  
                            out(fIndex + 2) = std::min(fU[i] , fV[i]);  
                            out(fIndex + 3) = std::max(fU[i] , fV[i]);
                            out(fIndex + 4) =         (fU[i] + fV[i]);
                            out(fIndex + 5) = std::abs(dEU - dEV);  
                            out(fIndex + 6) = std::min(dEU , dEV);  
                            out(fIndex + 7) = std::max(dEU , dEV);
                            out(fIndex + 8) =         (dEU + dEV);

                            fIndex += 9;

                        }
                    }
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


            // helper for channel acc
            template<class CHAIN, class ARRAY>
            void extract(const CHAIN & chain, ARRAY & out){

                using namespace vigra::acc;

                const auto mean = get<Mean>(chain);
                const auto quantiles = get<Quantiles>(chain);

                out[0] = replaceVals(mean,     0.0);
                out[1] = replaceVals(get<Variance>(chain), 0.0);
                out[2] = replaceVals(get<Skewness>(chain), 0.0);
                out[3] = replaceVals(get<Kurtosis>(chain), 0.0);
                for(auto qi=0; qi<7; ++qi){
                    out[4+qi] = replaceVals(quantiles[qi], mean);
                }
            }


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