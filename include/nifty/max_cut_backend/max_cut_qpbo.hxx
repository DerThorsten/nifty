#pragma once

#include <limits>
#include <string>
#include <sstream>



#include "QPBO.h"



namespace nifty {
namespace max_cut_backend{

    
    template<class T, class QPBO_VALUE_TYPE=float>
    class MaxCutQpbo{
    public:
        typedef T               ValueType;
        typedef QPBO_VALUE_TYPE QpboValueType;
        
        struct  Setting
        {
            bool guaranteeNoParallelEdges {false};
        };

        MaxCutQpbo(const uint64_t nNodes=0, const uint64_t nEdges=0)
        :  qpbo_(nNodes, nEdges){

            if(nNodes>0){
                qpbo_.AddNode(nNodes);
                // we should use the max degree node
                //qpbo_.AddUnaryTerm(0, -1000.0, 0.0);
            }
        }

        void reset(){
            qpbo_.Reset();
        }

        void assign(const uint64_t nNodes, const uint64_t nEdges=0){
            
            qpbo_.Reset();

            if(nNodes>0)
                qpbo_.AddNode(nNodes);
            
            if(nEdges>0)
                qpbo_.SetMaxEdgeNum(nEdges);

            //if(nNodes>0)
            //    qpbo_.AddUnaryTerm(0, -1000.0, 0.0);
        }


        void addEdge(const uint64_t u, const uint64_t v, const ValueType w){
            const auto w_ = w;// -1.0*w;
            qpbo_.AddPairwiseTerm(u,v,0.0,w_,w_,0.0);
        }

        double optimize(){
            
            if(!settings_.guaranteeNoParallelEdges){
                qpbo_.MergeParallelEdges();
            }

            qpbo_.Solve();
            qpbo_.Improve();

            auto val = qpbo_.ComputeTwiceEnergy(0)/2.0;
            //if(this->label(0) == 0){
            //    val += 1000.0;
            //}
            return val;
        }
        uint8_t label(const uint64_t node){
            const auto l = qpbo_.GetLabel(node);

            if(l==0)
                return 0;
            else if(l==1)
                return 1;
            else{
                //std::cout<<"damn..\n";
                return 0;
            }
        }

        static std::string name(){
            return std::string("MaxCutQpbo");
        }

    private:
        Setting settings_;
        QPBO<QpboValueType> qpbo_;
    };




    template<class T>
    class MaxCutQpbo<T,int>{
    private:
        struct Edge{
            Edge(const uint64_t uu=0, const uint64_t vv=0, const T ww=0)
            :   u(uu),
                v(vv),
                w(ww){
            }
            uint64_t u;
            uint64_t v;
            T w;
        };
    public:
        typedef T               ValueType;
        typedef int QpboValueType;
        
        struct  Setting
        {
            bool guaranteeNoParallelEdges {false};
        };

        MaxCutQpbo(const uint64_t nNodes=0, const uint64_t nEdges=0)
        :   settings_(),
            qpbo_(nNodes, nEdges),
            edges_(),
            minW_(std::numeric_limits<T>::infinity()),
            maxW_(-1.0*std::numeric_limits<T>::infinity())
        {
            edges_.reserve(nEdges);
            if(nNodes>0){
                qpbo_.AddNode(nNodes);
                // we should use the max degree node
                //qpbo_.AddUnaryTerm(0, -1000.0, 0.0);
            }
        }

        void reset(){
            qpbo_.Reset();
            edges_.clear();

            minW_ =      std::numeric_limits<T>::infinity();
            maxW_ = -1.0*std::numeric_limits<T>::infinity();

        }

        void assign(const uint64_t nNodes, const uint64_t nEdges=0){
            edges_.reserve(nEdges);
            qpbo_.Reset();

            if(nNodes>0)
                qpbo_.AddNode(nNodes);
            
            if(nEdges>0)
                qpbo_.SetMaxEdgeNum(nEdges);

            //if(nNodes>0)
            //    qpbo_.AddUnaryTerm(0, -1000.0, 0.0);
        }


        void addEdge(const uint64_t u, const uint64_t v, const ValueType w){
            edges_.emplace_back(u, v, w);
            minW_ = std::min(minW_, w);
            maxW_ = std::min(maxW_, w);

            
        }

        double optimize(){

            const auto mW = std::max(std::abs(minW_),std::abs(maxW_));
            const auto fac = 100.0 / mW;

            for(const auto  edge : edges_){
                const auto w = QpboValueType((edge.w * fac) + 0.5);
                qpbo_.AddPairwiseTerm(edge.u,edge.v,0.0,w,w,0.0);
            }

            
            if(!settings_.guaranteeNoParallelEdges){
                qpbo_.MergeParallelEdges();
            }

            qpbo_.Solve();
            qpbo_.Improve();

            double energy = 0.0;
            for(const auto  edge : edges_){
                const auto  u = edge.u;
                const auto  v = edge.v;
                if(label(u)!=label(v))
                    energy += edge.w;
            }
            return energy;
        }
        uint8_t label(const uint64_t node){
            const auto l = qpbo_.GetLabel(node);

            if(l==0)
                return 0;
            else if(l==1)
                return 1;
            else{
                //std::cout<<"damn..\n";
                return 0;
            }
        }

        static std::string name(){
            return std::string("MaxCutQpbo");
        }

    private:
        Setting settings_;
        QPBO<QpboValueType> qpbo_;
        std::vector<Edge> edges_;
        T minW_;
        T maxW_;
    };








}
}