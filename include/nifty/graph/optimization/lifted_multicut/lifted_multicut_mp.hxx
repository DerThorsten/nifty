#pragma once

#include "nifty/graph/optimization/lifted_multicut/lifted_multicut_objective.hxx"
#include "nifty/graph/optimization/lifted_multicut/lifted_multicut_base.hxx"
#include "nifty/graph/optimization/lifted_multicut/lifted_multicut_greedy_additive.hxx"
#include "nifty/graph/optimization/lifted_multicut/lifted_multicut_kernighan_lin.hxx"
#include "nifty/ufd/ufd.hxx"

// LP_MP includes
#include "solvers/multicut/multicut.h"
#include "visitors/standard_visitor.hxx"

namespace nifty{
namespace graph{
namespace optimization{
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
        typedef typename BaseType::VisitorBaseType VisitorBase;
        typedef typename BaseType::VisitorProxyType VisitorProxy;
        typedef typename BaseType::NodeLabelsType NodeLabels;

        // factory for the lifted primal rounder
        typedef LiftedMulticutFactoryBase<ObjectiveType> LmcFactoryBase;

        struct LiftedRounder{

            typedef UndirectedGraph<> GraphType;
            typedef LiftedMulticutObjective<GraphType, typename ObjectiveType::WeightType> PrimalRounderObjectiveType;
            typedef LiftedMulticutBase<PrimalRounderObjectiveType> PrimalRounderBaseType;
            typedef typename PrimalRounderObjectiveType::LiftedGraphType  PrimalRounderLiftedGraphType;
            typedef typename PrimalRounderBaseType::NodeLabelsType        PrimalRounderNodeLabels;
            typedef LiftedMulticutFactoryBase<PrimalRounderObjectiveType> PrimalRounderLmcFactoryBase;

            LiftedRounder(std::shared_ptr<PrimalRounderLmcFactoryBase> factory, const bool greedyWarmstart) 
                : factory_(factory), greedyWarmstart_(greedyWarmstart)
            {}

            // TODO do we have to call by value here due to using async or could we also use a call by refernce?
            // TODO need to change between between edge and node labelings -> could be done more efficient ?!
            std::vector<char> operator()(
                    GraphType &&           originalGraph,
                    PrimalRounderLiftedGraphType &&    liftedGraph,
                    std::vector<double> && edgeValues
            ) {

                std::vector<char> labeling(edgeValues.size(), 0);
                if(originalGraph.numberOfEdges() > 0) {
                    
                    const size_t nLocalEdges = originalGraph.numberOfEdges();
                    PrimalRounderObjectiveType obj(originalGraph);

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
                   
                    PrimalRounderNodeLabels nodeLabels(originalGraph.numberOfNodes());
                    if(greedyWarmstart_) {
                        LiftedMulticutGreedyAdditive<PrimalRounderObjectiveType> greedy(obj);
                        greedy.optimize(nodeLabels, nullptr);
                    }
                    
                    auto solverPtr = factory_->create(obj);
                    std::cout << "compute lifted multicut primal with " << (greedyWarmstart_ ? "GAEC + " : "") << solverPtr->name() << std::endl;
                    solverPtr->optimize(nodeLabels, nullptr);
                    delete solverPtr;
                    
                    // node labeling to edge labeling
                    for(auto eId = 0; eId < edgeValues.size(); ++eId) {
                        const auto & uv = (eId < nLocalEdges) ? originalGraph.uv(eId) : liftedGraph.uv( eId - nLocalEdges );
                        labeling[eId] = nodeLabels[uv.first] != nodeLabels[uv.second];
                    }

                }
                return labeling;

            }
            
            // Dummy implementation to work with LP_MP::MulticutConstructor
            std::vector<char> operator()(
                    GraphType &&,
                    std::vector<double> &&) {
                return std::vector<char>();
            }

            static std::string name() {
                return "LiftedRounder";
            }

        private:
            std::shared_ptr<PrimalRounderLmcFactoryBase> factory_;
            bool greedyWarmstart_;
    
        };
    
        typedef LP_MP::FMC_LIFTED_MULTICUT<LiftedRounder> FMC;
        typedef LP_MP::Solver<FMC,LP_MP::LP,LP_MP::StandardTighteningVisitor,LiftedRounder> SolverBase;
        typedef LP_MP::ProblemConstructorRoundingSolver<SolverBase> SolverType;
    
        // TODO LP_MP settings
        struct Settings{
            // lifted multicut factory for the primal rounder used in lp_mp
            std::shared_ptr<typename LiftedRounder::PrimalRounderLmcFactoryBase> lmcFactory;
            bool greedyWarmstart{true};
            // parameters for lp_mp solver TODO need better (non-completely-guessed...) default values
            double tightenSlope{0.05};
            size_t tightenMinDualImprovementInterval{0};
            double tightenMinDualImprovement{0.};
            double tightenConstraintsPercentage{0.1};
            size_t tightenConstraintsMax{0};
            size_t tightenInterval{10};
            size_t tightenIteration{100};
            std::string tightenReparametrization{"anisotropic"};
            std::string roundingReparametrization{"anisotropic"};
            std::string standardReparametrization{"anisotropic"};
            bool tighten{true};
            size_t minDualImprovementInterval{0};
            double minDualImprovement{0.};
            size_t lowerBoundComputationInterval{1};
            size_t primalComputationInterval{5};
            size_t timeout{0};
            size_t maxIter{1000};
            size_t numLpThreads{1};
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
        mpSolver_(nullptr),
        ufd_(graph_.numberOfNodes())
    {
        if(!bool(settings_.lmcFactory)) {
            typedef typename LiftedRounder::PrimalRounderObjectiveType PrimalRounderObjectiveType;
            typedef LiftedMulticutKernighanLin<PrimalRounderObjectiveType> DefaultSolverType;
            typedef LiftedMulticutFactory<DefaultSolverType> DefaultFactoryType;
            settings_.lmcFactory = std::make_shared<DefaultFactoryType>();
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

        std::vector<std::string> options = {
          "export_multicut", // TODO name of pyfile
          "-i", " ", // empty input file
          "--primalComputationInterval", std::to_string(settings_.primalComputationInterval),
          "--standardReparametrization", settings_.standardReparametrization,
          "--roundingReparametrization", settings_.roundingReparametrization,
          "--tightenReparametrization",  settings_.tightenReparametrization,
          "--tightenInterval",           std::to_string(settings_.tightenInterval),
          "--tightenIteration",          std::to_string(settings_.tightenIteration),
          "--tightenSlope",              std::to_string(settings_.tightenSlope),
          "--tightenConstraintsPercentage", std::to_string(settings_.tightenConstraintsPercentage),
          "--maxIter", std::to_string(settings_.maxIter),
          "--lowerBoundComputationInterval", std::to_string(settings_.lowerBoundComputationInterval)
          #ifdef WITH_OPENMP
          ,"--numLpThreads", std::to_string(numLpThreads)
          #endif
        };
        if(settings_.tighten)
            options.push_back("--tighten");
        if(settings_.minDualImprovement > 0) {
            options.push_back("--minDualImprovement");
            options.push_back(std::to_string(settings_.minDualImprovement));
        }
        if(settings_.minDualImprovementInterval > 0) {
            options.push_back("--minDualImprovementInterval");
            options.push_back(std::to_string(settings_.minDualImprovementInterval));
        }
        if(settings_.timeout > 0) {
            options.push_back("--timeout");
            options.push_back(std::to_string(settings_.timeout));
        }
        if(settings_.tightenConstraintsMax > 0) {
            options.push_back("--tightenConstraintsMax");
            options.push_back(std::to_string(settings_.tightenConstraintsMax));
        }
        if(settings_.tightenMinDualImprovement > 0) {
            options.push_back("--tightenMinDualImprovement");
            options.push_back(std::to_string(settings_.tightenMinDualImprovement));
        }
        if(settings_.tightenMinDualImprovementInterval > 0) {
            options.push_back("--tightenMinDualImprovementInterval");
            options.push_back(std::to_string(settings_.tightenMinDualImprovementInterval));
        }

        return options;
    }
    
    
    // TODO maybe this can be done more efficient
    // (if we only call it once, this should be fine, but if we need
    // to call this more often for some reason, this might get expensive)
    template<class OBJECTIVE>
    void LiftedMulticutMp<OBJECTIVE>::
    nodeLabeling() {

        ufd_.reset();
        auto & constructor = (*mpSolver_).template GetProblemConstructor<0>();
        for(auto e : graph_.edges()){
            const auto & uv = graph_.uv(e);
            const bool cut = constructor.get_edge_label(uv.first, uv.second);
            if(!cut){
                ufd_.merge(uv.first, uv.second);
            }
        }
        ufd_.elementLabeling(currentBest_->begin());
    }
    
    
    // TODO proper visitor
    template<class OBJECTIVE>
    void LiftedMulticutMp<OBJECTIVE>::
    optimize(
        NodeLabels & nodeLabels,  VisitorBase * visitor
    ){
        VisitorProxy visitorProxy(visitor);
        // set starting point as current best
        currentBest_ = &nodeLabels;
        
        // TODO for now the visitor is doing nothing, but we should implement one, that is
        // compatible with lp_mp visitor
        visitorProxy.begin(this);
        
        if(graph_.numberOfEdges()>0){
            mpSolver_->Solve();
            nodeLabeling();
        }
        visitorProxy.end(this);
    }


} // namespace nifty
} // namespace nifty::graph::optimization
} // namespace graph
} // namespace lifted_multicut
