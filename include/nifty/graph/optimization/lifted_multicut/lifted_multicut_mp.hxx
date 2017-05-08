#pragma once

#include "nifty/graph/optimization/lifted_multicut/lifted_multicut_base.hxx"
#include "nifty/graph/optimization/lifted_multicut/lifted_multicut_greedy_additive.hxx"
#include "nifty/graph/optimization/lifted_multicut/lifted_multicut_kernighan_lin.hxx"
#include "nifty/ufd/ufd.hxx"

// LP_MP includes
#include "solvers/multicut/multicut.h"
#include "visitors/standard_visitor.hxx"

namespace nifty{
namespace graph{
namespace lifted_multicut{
    
    /**
     * @brief      Class for message passing based inference for the lifted multicut objective
     *             An implementation of TODO cite Paul.
     *             
     *          
     * @tparam     OBJECTIVE  { description }
     */
    template<class OBJECTIVE>
    class LiftedMulticutMp : public LiftedMulticutBase<OBJECTIVE>
    {
    public: 
    
        typedef OBJECTIVE ObjectiveType;
        typedef LiftedMulticutBase<ObjectiveType> BaseType;
        typedef typename ObjectiveType::Graph Graph;
        typedef typename ObjectiveType::LiftedGraphType LiftedGraphType;
        
        typedef typename BaseType::VisitorBase VisitorBase;
        typedef typename BaseType::VisitorProxy VisitorProxy;
        typedef typename BaseType::NodeLabels NodeLabels;
        
        // factory for the lifted primal rounder
        typedef LiftedMulticutFactoryBase<ObjectiveType> LmcFactoryBase;
    
        struct LiftedRounder{
    
            typedef Graph GraphType;
            LiftedRounder(std::shared_ptr<LmcFactoryBase> factory, const bool greedyWarmstart) 
                : factory_(factory), greedyWarmstart_(greedyWarmstart)
            {}
            
            // TODO do we have to call by value here due to using async or could we also use a call by refernce?
            // TODO need to change between between edge and node labelings -> could be done more efficient ?!
            std::vector<char> operator()(
                    GraphType originalGraph,
                    GraphType liftedGraph,
                    std::vector<double> edgeValues) {

                std::vector<char> labeling(edgeValues.size(), 0);
                if(originalGraph.numberOfEdges() > 0) {
                    
                    const size_t nLocalEdges = originalGraph.numberOfEdges();
                    ObjectiveType obj(originalGraph);

                    // insert local costs
                    size_t edgeId = 0;
                    for(;edgeId < nLocalEdges; ++edgeId) {
                        const auto & uv = originalGraph.uv(edgeId);
                        obj.setCost(uv.first, uv.second, edgeValues[edgeId]);
                    }
                    // insert lifted costs
                    for(;edgeId < nLocalEdges + liftedGraph.numberOfEdges(); ++edgeId) {
                        const auto & uv = liftedGraph.uv(edgeId - nLocalEdges);
                        obj.setCost(uv.first, uv.second, edgeValues[edgeId]);
                    }
                   
                    NodeLabels nodeLabeling(originalGraph.numberOfNodes());
                    if(greedyWarmstart_) {
                        LiftedMulticutGreedyAdditive<ObjectiveType> greedy(obj);
                        greedy.optimize(nodeLabeling, nullptr);
                    }
                    
                    auto solverPtr = factory_->createRawPtr(obj);
                    solverPtr->optimize(nodeLabeling, nullptr);
                    delete solverPtr;
                    
                    // node labeling to edge labeling
                    for(auto eId = 0; eId < edgeValues.size(); ++eId) {
                        const auto & uv = (eId < nLocalEdges) ? originalGraph.uv(eId) : liftedGraph.uv( eId - nLocalEdges );
                        labeling[eId] = uv.first != uv.second;
                    }

                }
                return labeling;

            }
            
            // Dummy implementation to work with multicut constructor
            // TODO FIXME Is this actually used? -> then we would need to impl something
            std::vector<char> operator()(
                    GraphType originalGraph,
                    std::vector<double> edgeValues) {
            }

            static std::string name() {
                return "LiftedRounder";
            }

        private:
            std::shared_ptr<LmcFactoryBase> factory_;
            bool greedyWarmstart_;
    
        };
    
        typedef LP_MP::FMC_LIFTED_MULTICUT<LiftedRounder> FMC;
        typedef LP_MP::Solver<FMC,LP_MP::LP,LP_MP::StandardTighteningVisitor,LiftedRounder> SolverBase;
        typedef LP_MP::ProblemConstructorRoundingSolver<SolverBase> SolverType;
    
        // TODO LP_MP settings
        struct Settings{
            // lifted multicut factory for the primal rounder used in lp_mp
            std::shared_ptr<LmcFactoryBase> lmcFactory;
            bool greedyWarmstart{true};
        
        };
            
        LiftedMulticutMp(const ObjectiveType & objective, const Settings & settings = Settings());
        
        virtual void optimize(NodeLabels & nodeLabels, VisitorBase * visitor);
        
        virtual ~LiftedMulticutMp() {
            delete mpSolver_;
        }
        
        virtual const ObjectiveType & objective() const {
            return objective_;
        }
    
        virtual const NodeLabels & currentBestNodeLabels() {
            return *currentBest_;
        }
    
        virtual std::string name() const {
            return std::string("LiftedMulticutMp");
        }
    
    private:
            
        void initializeMp();
        void nodeLabeling();
        std::vector<std::string> toOptionsVector() const;
        
        const ObjectiveType & objective_;
        Settings settings_;
        const Graph & graph_;
        const LiftedGraphType & liftedGraph_;
        
        NodeLabels * currentBest_;
            
        SolverType * mpSolver_;
        ufd::Ufd<uint64_t> ufd_;
    };

    template<class OBJECTIVE>
    LiftedMulticutMp<OBJECTIVE>::LiftedMulticutMp(
        const OBJECTIVE & objective,
        const Settings & settings)
    :   objective_(objective),
        settings_(settings),
        graph_(objective.graph()),
        liftedGraph_(objective.liftedGraph()),
        currentBest_(nullptr),
        mpSolver_(nullptr)
    {
        if(!bool(settings_.lmcFactory)) {
            typedef LiftedMulticutKernighanLin<ObjectiveType> DefaultSolver;
            typedef LiftedMulticutFactory<DefaultSolver> DefaultFactory;
            settings_.lmcFactory = std::make_shared<DefaultFactory>();
        }
        mpSolver_ = new SolverType( toOptionsVector(),
                LiftedRounder(settings_.lmcFactory, settings_.greedyWarmstart) );
        this->initializeMp();
    }
    
    template<class OBJECTIVE>
    void LiftedMulticutMp<OBJECTIVE>::
    initializeMp() {
        
        if(graph_.numberOfEdges()!= 0 ){
            
            auto & constructor = (*mpSolver_).template GetProblemConstructor<0>();
            const auto & weights = objective_.weights();

            const size_t nLocalEdges = graph_.numberOfEdges();
            size_t edgeId = 0;
            for(;edgeId < nLocalEdges; ++edgeId) {
                const auto & uv = graph_.uv(edgeId);
                constructor.AddUnaryFactor(uv.first, uv.second, weights[edgeId]);
            }
            for(;edgeId < nLocalEdges + liftedGraph_.numberOfEdges(); ++edgeId) {
                const auto & uv = liftedGraph_.uv(edgeId - nLocalEdges);
                constructor.AddLiftedUnaryFactor(uv.first, uv.second, weights[edgeId]);
            }
        }
    }
    
    // returns options in correct format for the LP_MP solver
    // TODO would be bettter to have a decent interface for LP_MP and then
    // get rid of this
    template<class OBJECTIVE>
    std::vector<std::string> LiftedMulticutMp<OBJECTIVE>::
    toOptionsVector() const {

        //std::vector<std::string> options = {
        //  "export_multicut", // TODO name of pyfile
        //  "-i", " ", // empty input file
        //  "--primalComputationInterval", std::to_string(settings_.primalComputationInterval),
        //  "--standardReparametrization", settings_.standardReparametrization,
        //  "--roundingReparametrization", settings_.roundingReparametrization,
        //  "--tightenReparametrization",  settings_.tightenReparametrization,
        //  "--tightenInterval",           std::to_string(settings_.tightenInterval),
        //  "--tightenIteration",          std::to_string(settings_.tightenIteration),
        //  "--tightenSlope",              std::to_string(settings_.tightenSlope),
        //  "--tightenConstraintsPercentage", std::to_string(settings_.tightenConstraintsPercentage),
        //  "--maxIter", std::to_string(settings_.numberOfIterations),
        //};
        //if(settings_.tighten)
        //    options.push_back("--tighten");
        //if(settings_.minDualImprovement > 0) {
        //    options.push_back("--minDualImprovement");
        //    options.push_back(std::to_string(settings_.minDualImprovement));
        //}
        //if(settings_.minDualImprovementInterval > 0) {
        //    options.push_back("--minDualImprovementInterval");
        //    options.push_back(std::to_string(settings_.minDualImprovementInterval));
        //}
        //if(settings_.timeout > 0) {
        //    options.push_back("--timeout");
        //    options.push_back(std::to_string(settings_.timeout));
        //}
        
        std::vector<std::string> options;
        return options;
    }


} // namespace nifty
} // namespace graph
} // namespace lifted_multicut
