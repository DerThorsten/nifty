#pragma once

#include "nifty/graph/undirected_list_graph.hxx"
#include "nifty/tools/runtime_check.hxx"
#include "nifty/graph/components.hxx"

#include "nifty/graph/opt/ho_multicut/ho_multicut_base.hxx"
#include "nifty/ilp_backend/ilp_backend.hxx"
#include "nifty/graph/detail/contiguous_indices.hxx"
#include "nifty/graph/detail/node_labels_to_edge_labels_iterator.hxx"


#include "nifty/graph/opt/multicut/multicut_base.hxx"
#include "nifty/graph/opt/common/solver_factory.hxx"
#include "nifty/graph/opt/multicut/multicut_objective.hxx"

#include <xtensor/xarray.hpp>
#include <xtensor/xstrided_view.hpp>

#include "zgm/zfunctions.hpp"
#include "zgm/dgm/zqpbo.hpp"

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
            



        // submodel multicut
        // =====================
        typedef GraphType                                                                   SubmodelMcGraph;
        typedef nifty::graph::opt::multicut::MulticutObjective<SubmodelMcGraph, WeightType> SubmodelMcObjective;
        typedef nifty::graph::opt::multicut::MulticutBase<SubmodelMcObjective>              SubmodelMcBaseType;
        typedef nifty::graph::opt::common::SolverFactoryBase<SubmodelMcBaseType>            SubmodelMcFactoryBase;
        typedef typename SubmodelMcBaseType::NodeLabelsType                                 SubmodelMcNodeLabels;


        typedef typename  GraphType:: template EdgeMap<float>                                 FloatEdgeMap;







        
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

        

            /**
             *  \brief  Maximum allowed cutting plane iterations 
             *  \details  Maximum allowed cutting plane iteration.
             *  A value of zero will be interpreted as an unlimited
             *  number of iterations.
             */
            size_t numberOfIterations{0};   

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


        const ObjectiveType & objective_;
        const GraphType & graph_;

        Components components_;

        // for all so far existing graphs EdgeIndicesToContiguousEdgeIndices
        // is a zero overhead function which just returns the edge itself
        // since all so far existing graphs have contiguous edge ids
        DenseIds denseIds_;

        SettingsType settings_;
        NodeLabelsType * currentBest_;

        FloatEdgeMap lambdas_;
        SubmodelMcObjective submodeMcObjective_;
    };

    
    template<class OBJECTIVE>
    HoMulticutDualDecomposition<OBJECTIVE>::
    HoMulticutDualDecomposition(
        const ObjectiveType & objective, 
        const SettingsType & settings
    )
    :   objective_(objective),
        graph_(objective.graph()),
        components_(graph_),
        denseIds_(graph_),
        settings_(settings),
        lambdas_(graph_, 0.0),
        submodeMcObjective_(graph_)
    {

    


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
            const auto & mcWeights = objective_.weights();
            auto & submodelMcWeights = submodeMcObjective_.weights();

            for(auto e : graph_.edges())
            {
                submodelMcWeights[e] = mcWeights[e]/2.0 + lambdas_[e];
            }

            // build submodel crf
            // optimize submodel crf
            typedef double value_type;
            typedef std::vector<uint8_t> variable_space_type;
            typedef zgm::dgm::zqpbo_inplace<variable_space_type, value_type> zqpbo_type;
            typedef typename zqpbo_type::settings_type  zqpbo_settings_type;
            typedef xt::xtensorf<value_type, xt::xshape<2>>      unary_function_type;
            typedef xt::xtensorf<value_type, xt::xshape<2, 2>>   pairwise_function_type;
            variable_space_type variable_space(graph_.numberOfEdges(), 2);
            zqpbo_settings_type qpbo_settings;
            zqpbo_type qpbo(variable_space, qpbo_settings);

            for(auto e : graph_.edges())
            {
                unary_function_type unary_function = {0.0,  mcWeights[e]/2.0 - lambdas_[e]};
                qpbo.add_factor({e}, unary_function);
            }
            for(const auto & fac : objective_.higherOrderFactors())
            {
                const auto& vis = fac.edgeIds();
                const auto& f = fac.valueTable();
                qpbo.add_factor({vis[0], vis[1]}, f); 
            }


            // compute gradient

            // do gradient step
        }


        visitorProxy.end(this);
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

