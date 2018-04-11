#pragma once

#include "nifty/graph/undirected_list_graph.hxx"
#include "nifty/tools/runtime_check.hxx"
#include "nifty/graph/components.hxx"
#include "nifty/graph/paths.hxx"
#include "nifty/graph/opt/ho_multicut/ho_multicut_base.hxx"
#include "nifty/graph/three_cycles.hxx"
#include "nifty/graph/breadth_first_search.hxx"
#include "nifty/graph/bidirectional_breadth_first_search.hxx"
#include "nifty/ilp_backend/ilp_backend.hxx"
#include "nifty/graph/detail/contiguous_indices.hxx"
#include "nifty/graph/detail/node_labels_to_edge_labels_iterator.hxx"


#include <xtensor/xarray.hpp>
#include <xtensor/xstrided_view.hpp>

namespace nifty{
namespace graph{
namespace opt{
namespace ho_multicut{


    class GraphCut



    /*!
    @brief dual decompostion solver
    @ingroup group_multicut_solver
    @tparam OBJECTIVE The multicut objective (e.g. MulticutObjective)
    @tparam CRF_SOLVER The ILP solver backend (e.g. ilp_backend::Cplex, ilp_backend::Glpk, ilp_backend::Gurobi)
    */
    template<class OBJECTIVE, class CRF_SOLVER>
    class HoMulticutDualDecomposition : public HoMulticutBase<OBJECTIVE>
    {
    public: 

        typedef OBJECTIVE ObjectiveType;
        typedef typename ObjectiveType::GraphType GraphType;
    
        
        
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
        typedef CRF_SOLVER CrfSolverType;
        typedef typename CrfSolverType::SettingsType CrfSolverSettingsType;

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

            CrfSolverSettingsType crfSolverSettings{};
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
            return std::string("HoMulticutDualDecomposition") + CRF_SOLVER::name();
        }

        
    private:


        const ObjectiveType & objective_;
        const GraphType & graph_;

        CrfSolverType  crfSolver_;
        Components components_;

        // for all so far existing graphs EdgeIndicesToContiguousEdgeIndices
        // is a zero overhead function which just returns the edge itself
        // since all so far existing graphs have contiguous edge ids
        DenseIds denseIds_;

        SettingsType settings_;
        std::vector<size_t> variables_;
        std::vector<double> coefficients_;
        NodeLabelsType * currentBest_;
        size_t addedConstraints_;
        size_t numberOfOptRuns_;
    };

    
    template<class OBJECTIVE, class CRF_SOLVER>
    HoMulticutDualDecomposition<OBJECTIVE, CRF_SOLVER>::
    HoMulticutDualDecomposition(
        const ObjectiveType & objective, 
        const SettingsType & settings
    )
    :   objective_(objective),
        graph_(objective.graph()),
        crfSolver_(settings_.crfSolverSettings),//settings.crfSolverSettings),
        components_(graph_),
        denseIds_(graph_),
        bibfs_(graph_),
        settings_(settings),
        variables_(   std::max(uint64_t(3),uint64_t(graph_.numberOfEdges()))),
        coefficients_(std::max(uint64_t(3),uint64_t(graph_.numberOfEdges())))
    {

    


    }

    template<class OBJECTIVE, class CRF_SOLVER>
    void HoMulticutDualDecomposition<OBJECTIVE, CRF_SOLVER>::
    optimize(
        NodeLabelsType & nodeLabels,  VisitorBaseType * visitor
    ){  

        //std::cout<<"nStartConstraints "<<addedConstraints_<<"\n";
        VisitorProxyType visitorProxy(visitor);

        //visitorProxy.addLogNames({"violatedConstraints"});

        currentBest_ = &nodeLabels;
        
        visitorProxy.begin(this);


        visitorProxy.end(this);
    }

    template<class OBJECTIVE, class CRF_SOLVER>
    const typename HoMulticutDualDecomposition<OBJECTIVE, CRF_SOLVER>::ObjectiveType &
    HoMulticutDualDecomposition<OBJECTIVE, CRF_SOLVER>::
    objective()const{
        return objective_;
    }


} // namespace nifty::graph::opt::multicut
} // namespace nifty::graph::opt
} // namespace nifty::graph
} // namespace nifty

