#pragma once

#include "nifty/graph/undirected_list_graph.hxx"
#include "nifty/tools/runtime_check.hxx"
#include "nifty/graph/components.hxx"


#include "nifty/ilp_backend/ilp_backend.hxx"
#include "nifty/graph/detail/contiguous_indices.hxx"
#include "nifty/graph/detail/node_labels_to_edge_labels_iterator.hxx"

#include "nifty/tools/runtime_check.hxx"
#include "nifty/graph/opt/common/solver_factory.hxx"

#include "nifty/graph/opt/ho_multicut/fusion_move.hxx"
#include "nifty/graph/opt/ho_multicut/ho_multicut_base.hxx"
#include "nifty/graph/opt/ho_multicut/ho_multicut_objective.hxx"


#include "nifty/graph/opt/multicut/multicut_base.hxx"
#include "nifty/graph/opt/multicut/multicut_objective.hxx"

#include <xtensor/xarray.hpp>
#include <xtensor/xstrided_view.hpp>

#include "zgm/zfunctions.hpp"
#include "zgm/dgm/zqpbo.hpp"
#include "zgm/dgm/zgraphcut_maxflow.hpp"
#include "zgm/dgm/zad3.hpp"

#include <xtensor/xexpression.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xfixed.hpp>

namespace nifty{
namespace graph{
namespace opt{
namespace ho_multicut{




    /*!
    @brief dual decompostion solver
    @ingroup group_multicut_solver
    @tparam OBJECTIVE The multicut objective (e.g. MulticutObjective)
    @tparam CRF_SOLVER The ILP solver backend (e.g. ilp_backend::Cplex, ilp_backend::Glpk, ilp_backend::Gurobi)
    */
    template<class OBJECTIVE>
    class HoMulticutDualDecomposition : public HoMulticutBase<OBJECTIVE>
    {
    public: 

        typedef OBJECTIVE ObjectiveType;
        typedef typename ObjectiveType::GraphType GraphType;
        typedef typename ObjectiveType::WeightType WeightType;
        


        // the fusion move
        // =======================
        typedef nifty::graph::opt::ho_multicut::FusionMove<ObjectiveType> FusionMoveType;
        typedef typename FusionMoveType::SettingsType                     FusionMoveSettingsType;
        typedef typename FusionMoveType::FmHoMcFactoryBase                FmHoMcFactoryBase;


        // submodel multicut
        // =====================
        typedef GraphType                                                                   SubmodelMcGraph;
        typedef nifty::graph::opt::multicut::MulticutObjective<SubmodelMcGraph, WeightType> SubmodelMcObjective;
        typedef nifty::graph::opt::multicut::MulticutBase<SubmodelMcObjective>              SubmodelMcBaseType;
        typedef nifty::graph::opt::common::SolverFactoryBase<SubmodelMcBaseType>            SubmodelMcFactoryBase;
        typedef typename SubmodelMcBaseType::NodeLabelsType                                 SubmodelMcNodeLabels;


        typedef typename  GraphType:: template EdgeMap<float>                               FloatEdgeMap;
        typedef typename  GraphType:: template EdgeMap<uint8_t>                             UInt8EdgeMap;






        
        typedef std::is_same<typename GraphType::EdgeIdTag,  ContiguousTag> GraphHasContiguousEdgeIds;

        static_assert( GraphHasContiguousEdgeIds::value,
                  "HoMulticutDualDecomposition assumes that the edge id-s are dense "
        );
        // static_assert( GraphHasSortedEdgeIds::value,
        //           "HoMulticutDualDecomposition assumes that the edge id-s are dense "
        // );


        /// \brief Base Type / parent class
        typedef HoMulticutBase<OBJECTIVE> BaseType;

        /// Visitor base class
        typedef typename BaseType::VisitorBaseType VisitorBaseType;
        
        typedef typename BaseType::NodeLabelsType NodeLabelsType;


    private:

        typedef typename BaseType::VisitorProxyType VisitorProxyType;
        typedef ComponentsUfd<GraphType> Components;
        typedef detail_graph::EdgeIndicesToContiguousEdgeIndices<GraphType> DenseIds;

    public:

        /**
         * @brief Settings for HoMulticutDualDecomposition solver.
         * @details The settings for HoMulticutDualDecomposition
         * are not very critical and the default
         * settings should be changed seldomly.
         */
        struct SettingsType{
            enum class crf_solver_type
            {
                graphcut,
                qpbo,
                ad3
            };  

            std::shared_ptr<SubmodelMcFactoryBase> submodelMcFactory;
            size_t numberOfIterations{0};   
            float stepSize{0.1};
            crf_solver_type crf_solver{crf_solver_type::graphcut};
            float absoluteGap{0.0};

            FusionMoveSettingsType fusionMoveSetting;

        };

        virtual ~HoMulticutDualDecomposition(){
        }

        HoMulticutDualDecomposition(const ObjectiveType & objective, const SettingsType & settings = SettingsType());


        virtual void optimize(NodeLabelsType & nodeLabels, VisitorBaseType * visitor);
        virtual const ObjectiveType & objective() const;


        virtual const NodeLabelsType & currentBestNodeLabels( ){
            return *currentBest_;
        }

        virtual std::string name()const{
            return std::string("HoMulticutDualDecomposition");
        }

        
    private:
        double solve_submodel_crf();
        double solve_submodel_mc();
        double get_upper();

        const ObjectiveType & objective_;
        const GraphType & graph_;

        Components components_;

        // for all so far existing graphs EdgeIndicesToContiguousEdgeIndices
        // is a zero overhead function which just returns the edge itself
        // since all so far existing graphs have contiguous edge ids
        DenseIds denseIds_;

        SettingsType settings_;
        NodeLabelsType * currentBest_;

        UInt8EdgeMap sol_crf_;
        SubmodelMcNodeLabels node_labels_mc_;
        SubmodelMcNodeLabels node_labels_crf_;
        SubmodelMcNodeLabels node_labels_fm_;
        SubmodelMcNodeLabels node_labels_fm2_;
        FloatEdgeMap lambdas_;
        SubmodelMcObjective submodeMcObjective_;
        float best_ub_;
        float best_lb_;


        // fusion move
        FusionMoveType fusionMove_;

    };

    
    template<class OBJECTIVE>
    HoMulticutDualDecomposition<OBJECTIVE>::
    HoMulticutDualDecomposition(
        const ObjectiveType & objective, 
        const SettingsType & settings
    )
    :   objective_(objective),
        graph_(objective.graph()),
        components_(objective.graph()),
        denseIds_(objective.graph()),
        settings_(settings),
        node_labels_mc_(objective.graph(), 0),
        node_labels_crf_(objective.graph(), 0),
        node_labels_fm_(objective.graph(), 0),
        node_labels_fm2_(objective.graph(), 0),
        sol_crf_(objective.graph(), 0),
        lambdas_(objective.graph(), 0.0),
        submodeMcObjective_(objective.graph()),
        best_ub_(     std::numeric_limits<float>::infinity()),
        best_lb_(-1.0*std::numeric_limits<float>::infinity()),
        fusionMove_(objective, settings.fusionMoveSetting)
    {

        if(!settings_.submodelMcFactory)
        {
            throw std::runtime_error("submodelMcFactory shall not be empty");
        }
    }

    template<class OBJECTIVE>
    void HoMulticutDualDecomposition<OBJECTIVE>::
    optimize(
        NodeLabelsType & nodeLabels,  VisitorBaseType * visitor
    ){  

        //std::cout<<"nStartConstraints "<<addedConstraints_<<"\n";
        VisitorProxyType visitorProxy(visitor);

        //visitorProxy.addLogNames({"violatedConstraints"});

        currentBest_ = &nodeLabels;
        
        visitorProxy.begin(this);


        for(std::size_t i=0; i<settings_.numberOfIterations; ++i)
        {
            // build submodel mc
            // optimize submodel mc



           
            const auto val_mc = this->solve_submodel_mc();
            const auto val_crf = this->solve_submodel_crf();
        
            const float ub = objective_.evalNodeLabels(node_labels_mc_);
            const float lb = val_mc + val_crf;        

            if(lb > best_lb_)
                best_lb_ = lb;
    
            auto lr = settings_.stepSize/float(1+i);
            // compute gradient
            for(auto e : graph_.edges())
            {
                
                const auto uv = graph_.uv(e);
                const auto is_cut_mc = (node_labels_mc_[uv.first] != node_labels_mc_[uv.second]);
                lambdas_[e] += lr * (float(is_cut_mc) - float( sol_crf_[e]));

            }

            // decode upper bound
            // for(auto e : graph_.edges())
            // {
            //     submodelMcWeights[e] = (0.9*mcWeights[e] + 0.1*lambdas_[e]);
            //     //submodelMcWeights[e] = lambdas_[e];
            // }

            // {
            //     auto submodelMcSolver = settings_.submodelMcFactory->create(submodeMcObjective_);
            //     submodelMcSolver->optimize(node_labels_mc_, nullptr);
            //     delete submodelMcSolver;
            // }
            this->get_upper();

            const auto  gap = best_ub_ - best_lb_;
            if(gap < settings_.absoluteGap){
                break;
            }
            if((i+1) % 1 == 0)
                std::cout<<"   best lb ub" <<best_lb_<<"  "<<best_ub_<<"    gap "<<best_ub_-best_lb_<<"\n";




           
        }


        visitorProxy.end(this);
    }

    template<class OBJECTIVE>
    double HoMulticutDualDecomposition<OBJECTIVE>::solve_submodel_crf(
    )
    {
        const auto & mcWeights = objective_.weights();
        auto & submodelMcWeights = submodeMcObjective_.weights();


        // build submodel crf
        // optimize submodel crf
        typedef double value_type;
        typedef std::vector<uint8_t> variable_space_type;
        
        typedef xt::xtensorf<value_type, xt::xshape<2>>      unary_function_type;
        typedef xt::xtensorf<value_type, xt::xshape<2, 2>>   pairwise_function_type;
        variable_space_type variable_space(graph_.numberOfEdges(), 2);


        auto f = [&](auto & solver){
            for(auto e : graph_.edges())
            {
                unary_function_type unary_function = {0.0,  mcWeights[e]/2.0 - lambdas_[e]};
                solver.add_factor({e}, unary_function);
            }
            for(const auto & fac : objective_.higherOrderFactors())
            {
                const auto& vis = fac.edgeIds();
                const auto& f = fac.valueTable();
                solver.add_factor(vis, f); 
            }
            solver.optimize();
            auto states = solver.states();
            for(auto edge : graph_.edges())
            {
                sol_crf_[edge] = states[edge];
            }
            double val = 0.0;

            // calculate value 
            for(auto e : graph_.edges())
            {
                if(sol_crf_[e])
                    val += mcWeights[e]/2.0 - lambdas_[e];
            }

            std::vector<uint8_t>  fac_state;
            for(const auto & fac : objective_.higherOrderFactors())
            {
                const auto& vis = fac.edgeIds();
                fac_state.resize(vis.size());
                auto c=0;
                for(auto v : vis)
                {
                    fac_state[c] = sol_crf_[vis[c]];
                    ++c;
                }

                val += fac.valueTable()[fac_state];
                
            }

            return val;
            //return solver.value();
        };

        if(  settings_.crf_solver == SettingsType::crf_solver_type::qpbo)
        {
            typedef zgm::dgm::zqpbo_inplace<variable_space_type, value_type> solver_type;
            typedef typename solver_type::settings_type  settings_type;
            settings_type settings;
            solver_type solver(variable_space, settings);      
            return f(solver);
        }
        else if(settings_.crf_solver == SettingsType::crf_solver_type::graphcut)
        {
            typedef zgm::dgm::zgraphcut_maxflow_inplace<variable_space_type, value_type> solver_type;
            typedef typename solver_type::settings_type  settings_type;
            settings_type settings;
            solver_type solver(variable_space, settings);      
            return f(solver);
        }
        else if(settings_.crf_solver == SettingsType::crf_solver_type::ad3)
        {
            typedef zgm::dgm::zad3_inplace<variable_space_type> solver_type;
            typedef typename solver_type::settings_type  settings_type;
            settings_type settings;
            solver_type solver(variable_space, settings);      
            return f(solver);
        }
        else
        {
            throw std::runtime_error("wrong crf solver");
        }

    }

    template<class OBJECTIVE>
    double HoMulticutDualDecomposition<OBJECTIVE>::solve_submodel_mc(
    )
    {
        const auto & mcWeights = objective_.weights();
        auto & submodelMcWeights = submodeMcObjective_.weights();

        for(auto e : graph_.edges())
        {
            submodelMcWeights[e] = mcWeights[e]/2.0 + lambdas_[e];
        }


        {
            auto submodelMcSolver = settings_.submodelMcFactory->create(submodeMcObjective_);
            submodelMcSolver->optimize(node_labels_mc_, nullptr);
            delete submodelMcSolver;
        }
        return submodeMcObjective_.evalNodeLabels(node_labels_mc_);
    }

    template<class OBJECTIVE>
    double HoMulticutDualDecomposition<OBJECTIVE>::get_upper(

    ){
        //  multicut 
        const auto & mcWeights = objective_.weights();
        auto & submodelMcWeights = submodeMcObjective_.weights();

        const float ub_mc = objective_.evalNodeLabels(node_labels_mc_);

        //best_ub_ = std::min(ub_mc, best_ub_);


        // crf
        for(auto e : graph_.edges())
        {
            submodelMcWeights[e] = (sol_crf_[e] == 1 ? -1:1) + 0.000001*(mcWeights[e]/2.0 + lambdas_[e]);
                //mcWeights[e]/2.0 + lambdas_[e];
        }
        {
            auto submodelMcSolver = settings_.submodelMcFactory->create(submodeMcObjective_);
            submodelMcSolver->optimize(node_labels_crf_, nullptr);
            delete submodelMcSolver;

        }
        const float ub_crfmc =  objective_.evalNodeLabels(node_labels_crf_);
        
        auto bestMcCrf = std::min(ub_crfmc, ub_mc);
        

        

        //std::cout<<"do   fuse\n";
        //std::cout<<"first fuse\n";
        fusionMove_.fuse({&node_labels_crf_, &node_labels_mc_}, &node_labels_fm_);
        //std::cout<<"first fuse done\n";
        const auto ub_fm =  objective_.evalNodeLabels(node_labels_fm_);
        NIFTY_CHECK_OP(ub_fm, <=, bestMcCrf+0.00001,"damn");

        //std::cout<<"direct bevore fuse"<<objective_.evalNodeLabels(*currentBest_);
        //std::cout<<"second fuse\n";
        fusionMove_.fuse({&node_labels_fm_,  currentBest_}, &node_labels_fm2_);
        std::copy(node_labels_fm2_.begin(), node_labels_fm2_.end(), currentBest_->begin());
        //std::cout<<"second fuse done\n";
        const auto minFmCurrentBest = std::min(double(best_ub_),ub_fm);
        //
        const float ub_fm2 =  objective_.evalNodeLabels(*currentBest_);

        NIFTY_CHECK_OP(ub_fm2, <=, minFmCurrentBest+0.00001,"damn");



        //std::cout<<"best: "<<best_ub_<<"\n";
        //std::cout<<"ub mc /crf / fm / fm2 "<<ub_mc<<" "<<ub_crfmc<<" "<<ub_fm<<" "<<ub_fm2<<"\n\n";

        best_ub_ = ub_fm2;

    }

    template<class OBJECTIVE>
    const typename HoMulticutDualDecomposition<OBJECTIVE>::ObjectiveType &
    HoMulticutDualDecomposition<OBJECTIVE>::
    objective()const{
        return objective_;
    }


} // namespace nifty::graph::opt::multicut
} // namespace nifty::graph::opt
} // namespace nifty::graph
} // namespace nifty

