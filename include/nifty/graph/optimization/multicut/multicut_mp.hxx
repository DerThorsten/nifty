#pragma once

#include<vector>

#include "nifty/tools/runtime_check.hxx"
#include "nifty/graph/optimization/multicut/multicut_base.hxx"
#include "nifty/graph/optimization/multicut/multicut_factory.hxx"
#include "nifty/graph/optimization/multicut/multicut_greedy_additive.hxx"
#include "nifty/graph/optimization/multicut/multicut_kernighan_lin.hxx"
#include "nifty/ufd/ufd.hxx"

// LP_MP includes
#include "visitors/standard_visitor.hxx"
#include "solvers/multicut/multicut.h"


namespace nifty{
namespace graph{
    
    template<class OBJECTIVE>
    class MulticutMp : public MulticutBase<OBJECTIVE>
    {
    public: 

        typedef OBJECTIVE Objective;
        typedef MulticutBase<OBJECTIVE> Base;
        typedef typename Base::VisitorBase VisitorBase;
        typedef typename Base::VisitorProxy VisitorProxy;
        typedef typename Base::EdgeLabels EdgeLabels;
        typedef typename Base::NodeLabels NodeLabels;
        typedef typename Objective::Graph Graph;
        
        // factory for the lp_mp primal rounder
        typedef MulticutFactoryBase<Objective> McFactoryBase;

    public:
        
        struct NiftyRounder {
            
            typedef Graph GraphType;

            NiftyRounder(std::shared_ptr<McFactoryBase> factory,
                    const bool greedyWarmstart) 
                : factory_(factory), greedyWarmstart_(greedyWarmstart)
            {}

            // TODO do we have to call by value here due to using async or could we also use a call by refernce?
            // TODO need to change between between edge and node labelings -> could be done more efficient ?!
            std::vector<char> operator()(GraphType g, std::vector<double> edgeValues) {

                std::vector<char> labeling(g.numberOfEdges(), 0);
                if(g.numberOfEdges() > 0) {
                    
                    Objective obj(g);

                    auto & objWeights = obj.weights();
                    for(auto eId = 0; eId < edgeValues.size(); ++eId) {
                        objWeights[eId] = edgeValues[eId];
                    }
                   
                    NodeLabels nodeLabels(g.numberOfNodes());
                    if(greedyWarmstart_) {
                        MulticutGreedyAdditive<Objective> greedy(obj);
                        greedy.optimize(nodeLabels, nullptr);
                    }
                    
                    auto solverPtr = factory_->createRawPtr(obj);
                    std::cout << "compute multicut primal with " << (greedyWarmstart_ ? "GAEC + " : "") << solverPtr->name()  << std::endl;
                    solverPtr->optimize(nodeLabels, nullptr);
                    delete solverPtr;
                    
                    // node labeling to edge labeling
                    for(auto eId = 0; eId < g.numberOfEdges(); ++eId) {
                        const auto & uv = g.uv(eId);
                        labeling[eId] = nodeLabels[uv.first] != nodeLabels[uv.second];
                    }

                }
                return labeling;

            }

            // TODO add factory name here
            static std::string name() {
                return "NiftyRounder";
            }

        private:
            std::shared_ptr<McFactoryBase> factory_;
            bool greedyWarmstart_;
        };
        
        typedef LP_MP::KlRounder Rounder;
        //typedef NiftyRounder Rounder;

        // TODO with or without odd wheel ?
        //typedef LP_MP::FMC_MULTICUT<LP_MP::MessageSendingType::SRMP,NiftyRounder> FMC;
        typedef LP_MP::FMC_ODD_WHEEL_MULTICUT<LP_MP::MessageSendingType::SRMP,Rounder> FMC;
        
        typedef LP_MP::Solver<FMC,LP_MP::LP,LP_MP::StandardTighteningVisitor,Rounder> SolverBase;
        typedef LP_MP::ProblemConstructorRoundingSolver<SolverBase> SolverType;

        // FIXME verbose deosn't have any effect right now
        struct Settings{
            // multicut factory for the primal rounder used in lp_mp
            std::shared_ptr<McFactoryBase> mcFactory;
            bool greedyWarmstart{false};
            // settings for the lp_mp solver
            size_t numberOfIterations{1000};
            int verbose{0};
            size_t primalComputationInterval{100};
            std::string standardReparametrization{"anisotropic"};
            std::string roundingReparametrization{"damped_uniform"};
            std::string tightenReparametrization{"damped_uniform"};
            bool tighten{true};
            size_t tightenInterval{100};
            size_t tightenIteration{10};
            double tightenSlope{0.02};
            double tightenConstraintsPercentage{0.1};
            double minDualImprovement{0.};
            size_t minDualImprovementInterval{0};
            size_t timeout{0};
            size_t numberOfThreads{1};
        };

        virtual ~MulticutMp(){
            delete mpSolver_;
        }
        
        MulticutMp(const Objective & objective, const Settings & settings = Settings());

        virtual void optimize(NodeLabels & nodeLabels, VisitorBase * visitor);
        
        virtual const Objective & objective() const {return objective_;}
        virtual const NodeLabels & currentBestNodeLabels() {return *currentBest_;}

        virtual std::string name() const {
            return std::string("MulticutMp");
        }
        
        // TODO do we need this, what does it do?
        // reset ?!
        //virtual void weightsChanged(){
        //}
        
    private:

        void initializeMp();
        void nodeLabeling();
        std::vector<std::string> toOptionsVector() const;

        const Objective & objective_;
        const Graph & graph_;

        Settings settings_;
        NodeLabels * currentBest_;
        size_t numberOfOptRuns_;
        SolverType * mpSolver_;
        ufd::Ufd<uint64_t> ufd_;
    };
   
    
    template<class OBJECTIVE>
    MulticutMp<OBJECTIVE>::
    MulticutMp(
        const Objective & objective, 
        const Settings & settings
    )
    :   objective_(objective),
        graph_(objective.graph()),
        settings_(settings),
        currentBest_(nullptr),
        mpSolver_(nullptr),
        ufd_(graph_.numberOfNodes())
    {
        // if we don't have a mc-factory, we use the LP_MP default rounder
        if(!bool(settings_.mcFactory)) {
            typedef MulticutKernighanLin<Objective> DefaultSolver;
            typedef MulticutFactory<DefaultSolver> DefaultFactory;
            settings_.mcFactory = std::make_shared<DefaultFactory>();
        }
        mpSolver_ = new SolverType(
                toOptionsVector()
                //,NiftyRounder(settings_.mcFactory, settings_.greedyWarmstart)
            );
        this->initializeMp();
    }

    template<class OBJECTIVE>
    void MulticutMp<OBJECTIVE>::
    initializeMp() {
        
        if(graph_.numberOfEdges()!= 0 ){
            
            auto & constructor = (*mpSolver_).template GetProblemConstructor<0>();
            const auto & weights = objective_.weights();

            for(auto e : graph_.edges()){
                const auto & uv = graph_.uv(e);
                constructor.AddUnaryFactor(uv.first, uv.second, weights[e]);
            }
        }
    }

    // returns options in correct format for the LP_MP solver
    // TODO would be bettter to have a decent interface for LP_MP and then
    // get rid of this
    template<class OBJECTIVE>
    std::vector<std::string> MulticutMp<OBJECTIVE>::
    toOptionsVector() const {

        std::vector<std::string> options = {
          "multicut_mp",
          "-i", " ", // empty input file
          "--primalComputationInterval", std::to_string(settings_.primalComputationInterval),
          "--standardReparametrization", settings_.standardReparametrization,
          "--roundingReparametrization", settings_.roundingReparametrization,
          "--tightenReparametrization",  settings_.tightenReparametrization,
          "--tightenInterval",           std::to_string(settings_.tightenInterval),
          "--tightenIteration",          std::to_string(settings_.tightenIteration),
          "--tightenSlope",              std::to_string(settings_.tightenSlope),
          "--tightenConstraintsPercentage", std::to_string(settings_.tightenConstraintsPercentage),
          "--maxIter", std::to_string(settings_.numberOfIterations),
          #ifdef WITH_OPENMP
          "--numLpThreads", std::to_string(settings_.numberOfThreads)
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
        
        return options;
    }


    // TODO maybe this can be done more efficient
    // (if we only call it once, this should be fine, but if we need
    // to call this more often for some reason, this might get expensive)
    template<class OBJECTIVE>
    void MulticutMp<OBJECTIVE>::
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


    template<class OBJECTIVE>
    void MulticutMp<OBJECTIVE>::
    optimize(
        NodeLabels & nodeLabels,  VisitorBase * visitor
    ){  

        //VisitorProxy visitorProxy(visitor);
        currentBest_ = &nodeLabels;
        
        // TODO for now the visitor is doing nothing, but we should implement one, that is
        // compatible with lp_mp visitor
        //visitorProxy.begin(this);
        
        if(graph_.numberOfEdges()>0){
            mpSolver_->Solve();
            nodeLabeling();
        }
        //visitorProxy.end(this);
    }

} // namespace nifty::graph
} // namespace nifty
