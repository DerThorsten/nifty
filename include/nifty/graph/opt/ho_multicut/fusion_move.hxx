#pragma once

#include <mutex>          // std::mutex
#include <memory>
#include <unordered_set>

#include "nifty/tools/runtime_check.hxx"
#include "nifty/ufd/ufd.hxx"
#include "nifty/graph/opt/ho_multicut/ho_multicut_base.hxx"
#include "nifty/graph/opt/common/solver_factory.hxx"
#include "nifty/graph/opt/ho_multicut/ho_multicut_objective.hxx"
#include "nifty/graph/undirected_list_graph.hxx"

namespace nifty{
namespace graph{
namespace opt{
namespace ho_multicut{



    namespace detail_fm
    {

    }

    template<class OBJECTIVE>
    class FusionMove{
    public:
        typedef OBJECTIVE ObjectiveType;
        typedef typename ObjectiveType::GraphType GraphType;
        typedef typename GraphType:: template NodeMap<uint64_t> NodeLabelsType;
        

        typedef UndirectedGraph<> FmGraph;
        typedef HoMulticutObjective<FmGraph, double> FmObjective;
        typedef HoMulticutBase<FmObjective> FmHoMcBase;
        typedef nifty::graph::opt::common::SolverFactoryBase<FmHoMcBase> FmHoMcFactoryBase;
        typedef HoMulticutEmptyVisitor<FmObjective> FmEmptyVisitor;
        typedef typename  FmHoMcBase::NodeLabelsType FmNodeLabelsType;

        struct SettingsType{
            std::shared_ptr<FmHoMcFactoryBase> hoMcFactory;
        };

        FusionMove(const ObjectiveType & objective, const SettingsType & settings = SettingsType())
        :   objective_(objective),
            graph_(objective.graph()),
            settings_(settings),
            ufd_(objective.graph().nodeIdUpperBound()+1),
            nodeToDense_(objective.graph())
        {
            if(!bool(settings_.hoMcFactory)){
                throw std::runtime_error("hoMcFactory may not be empty");
            }
        }

        template<class NODE_MAP>
        void fuse(
            std::initializer_list<const NODE_MAP *> proposals,
            NODE_MAP * result
        ){
            std::vector<const NODE_MAP *> p(proposals);
            fuse(p, result);
        }


        template<class NODE_MAP >
        void fuse(
            const std::vector< const NODE_MAP *> & proposals,
            NODE_MAP * result 
        ){
            //std::cout<<"reset ufd\n";
            ufd_.reset();


            for(const auto edge : graph_.edges()){
                // merge two nodes iff all proposals agree to merge
                bool merge = true;
                const auto uv = graph_.uv(edge);
                const auto u = uv.first;
                const auto v = uv.second;

                for(auto p=0; p<proposals.size(); ++p){

                    if(proposals[p]->operator[](u) != proposals[p]->operator[](v) ){
                        merge = false;
                        break;
                    }
                }
                if(merge)
                    ufd_.merge(u, v);
            }

            //std::cout<<"fuse impl\n";
            this->fuseImpl(result);

            //std::cout<<"fuse impl done\n";
            // evaluate if the result
            // is indeed better than each proposal
            // Iff the result is not better we
            // use the best proposal as a result
            auto eMin = std::numeric_limits<double>::infinity();
            auto eMinIndex = 0;
            for(auto i=0; i<proposals.size(); ++i){
                const auto p = proposals[i];
                const auto e = objective_.evalNodeLabels(*p);
                std::cout<<"p"<<i<<" "<<e<<"\n";
                if(e < eMin){
                    eMin = e;
                    eMinIndex = i;
                }
            }
            const auto eResult = objective_.evalNodeLabels(*result);
            std::cout<<"r"<<"  "<<eResult<<"\n";            
            if(eMin < eResult){
                for(auto node : graph_.nodes()){
                    result->operator[](node) = proposals[eMinIndex]->operator[](node);
                }
            }
            const auto eResult2 = objective_.evalNodeLabels(*result);
            std::cout<<"r2"<<" "<<eResult2<<"\n"; 
        }

    private:
        template<class NODE_MAP>
        void fuseImpl(NODE_MAP * result){

            // dense relabeling
            //std::cout<<"make dense\n";
            std::unordered_set<uint64_t> relabelingSet;
            for(const auto node: graph_.nodes()){
                relabelingSet.insert(ufd_.find(node));
            }
            auto denseLabel = 0;
            for(auto sparse: relabelingSet){
                nodeToDense_[sparse] = denseLabel;
                ++denseLabel;
            }
            const auto numberOfNodes = relabelingSet.size();
            
            //std::cout<<"fm graph\n";
            // build the graph
            FmGraph fmGraph(numberOfNodes);
            
            for(auto edge : graph_.edges()){
                const auto uv = graph_.uv(edge);
                const auto u = uv.first;
                const auto v = uv.second;
                const auto lu = nodeToDense_[ufd_.find(u)];
                const auto lv = nodeToDense_[ufd_.find(v)];
                NIFTY_CHECK_OP(lu,<,numberOfNodes,"");
                NIFTY_CHECK_OP(lv,<,numberOfNodes,"");
                if(lu != lv){
                    fmGraph.insertEdge(lu, lv);
                }
            }

            const auto fmEdges = fmGraph.numberOfEdges();

            if(fmEdges == 0){
                for(const auto node : graph_.nodes()){
                    result->operator[](node)  = ufd_.find(node);
                }
            }
            else{



                NIFTY_CHECK_OP(fmGraph.numberOfEdges(),>,0,"");
                FmObjective fmObjective(fmGraph);
                auto & fmWeights = fmObjective.weights();
                for(auto edge : graph_.edges()){
                    const auto uv = graph_.uv(edge);
                    const auto u = uv.first;
                    const auto v = uv.second;
                    const auto lu = nodeToDense_[ufd_.find(u)];
                    const auto lv = nodeToDense_[ufd_.find(v)];
                    NIFTY_CHECK_OP(lu,<,fmGraph.numberOfNodes(),"");
                    NIFTY_CHECK_OP(lv,<,fmGraph.numberOfNodes(),"");
                    if(lu != lv){
                        auto e = fmGraph.findEdge(lu, lv);
                        NIFTY_CHECK_OP(e,!=,-1,"");
                        fmWeights[e] += objective_.weights()[edge];
                    }
                }


                this->addHigherOrderFactors(fmObjective);







                //std::cout<<"fm solve\n";
                // solve that thin
                auto solverPtr = settings_.hoMcFactory->create(fmObjective);
                FmNodeLabelsType fmLabels(fmGraph);
                FmEmptyVisitor fmVisitor;
                //std::cout<<"opt\n";
                solverPtr->optimize(fmLabels, &fmVisitor);
                //std::cout<<"del ptr\n";
                delete solverPtr;

                //std::cout<<"fm get res\n";
                for(auto edge : graph_.edges()){
                    const auto uv = graph_.uv(edge);
                    const auto u = uv.first;
                    const auto v = uv.second;
                    const auto lu = nodeToDense_[ufd_.find(u)];
                    const auto lv = nodeToDense_[ufd_.find(v)];
                    if(lu != lv){
                        if(fmLabels[lu] == fmLabels[lv]){
                            ufd_.merge(u, v);
                        }
                    }
                }
                for(const auto node : graph_.nodes()){
                    result->operator[](node)  = ufd_.find(node);
                }
            }
        }


        void addHigherOrderFactors(
            FmObjective & fmObjective
        )
        {
            typedef std::vector<uint64_t> EdgeIdKey; 

            const auto& fmGraph = fmObjective.graph();
            auto& fmWeights = fmObjective.weights();

            std::vector<uint8_t>  edgeState;
            EdgeIdKey fmEdges;
            

            std::map<EdgeIdKey, xt::xarray<float> > edgeIdToVt;

            for(const auto & fac : objective_.higherOrderFactors())
            {   
                // input
                const auto& edgeIds = fac.edgeIds();
                const auto& vt = fac.valueTable();
                const auto& arity = fac.arity();

                // state of edges 
                // and edge mapping

                auto n_fixed = 0;
                auto n_free = 0;
                edgeState.resize(arity);
                fmEdges.resize(arity);

                for(std::size_t i=0; i<arity; ++i)
                {
                    const auto edge = edgeIds[i];
                    const auto uv = graph_.uv(edge);
                    const auto lu = nodeToDense_[ufd_.find(uv.first)];
                    const auto lv = nodeToDense_[ufd_.find(uv.second)];
                    const auto isCut = (lu != lv);
                    edgeState[i] = isCut;
                    if(isCut)
                    {
                        fmEdges[i] = fmGraph.findEdge(lu, lv);
                        ++n_free;
                    }
                    else
                    {
                        //0fmEdges[i] = -1;
                        ++n_fixed;
                    }
                }
                //std::cout<<"free/fixed "<<n_free<<"/"<<n_fixed<<"\n";

                if(arity == 2)
                {

                    if(n_fixed == 0)
                    {
                        // factors stays the same but could there might be a factor like this already
                        auto it = edgeIdToVt.find(fmEdges);
                        if(it == edgeIdToVt.end())
                        {
                            edgeIdToVt.insert(std::make_pair(fmEdges, vt));
                        }
                        else
                        {
                            auto& existingVt = it->second;
                            existingVt += vt;
                        }
                        
                    }
                    else if(n_fixed == 1)
                    {   
                        // first is alive
                        if(edgeState[0] == 1)
                        {
                            NIFTY_CHECK_OP(edgeState[1], ==, 0, "internal error");
                            const auto e0 = vt(0,0);
                            const auto e1 = vt(1,0);  
                            const auto w = (e1 - e0);
                            fmWeights[fmEdges[0]] += w;
                        }
                        // second alive
                        else
                        {
                            NIFTY_CHECK_OP(edgeState[0], ==, 0, "internal error");
                            NIFTY_CHECK_OP(edgeState[1], ==, 1, "internal error");
                            const auto e0 = vt(0,0);
                            const auto e1 = vt(0,1);  
                            const auto w = (e1 - e0);
                            fmWeights[fmEdges[1]] += w;
                        }   
                    }
                }
            }

            // add all ho factors
            for(const auto& kv : edgeIdToVt)
            {
                const auto edgeIds = kv.first;
                const auto vt = kv.second;
                fmObjective.addHigherOrderFactor(vt, edgeIds);
            }


            std::cout<<"reduced: "<<objective_.higherOrderFactors().size()<<" "<<fmObjective.higherOrderFactors().size()<<"\n";
        }

        const ObjectiveType & objective_;
        const GraphType & graph_;
        SettingsType settings_;
        nifty::ufd::Ufd< > ufd_;
        NodeLabelsType nodeToDense_;
    };





} // namespace nifty::graph::opt::multicut
} // namespace nifty::graph::opt
} // namespace nifty::graph
} // namespace nifty

