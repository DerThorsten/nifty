#pragma once

#include "nifty/tools/runtime_check.hxx"
#include "nifty/graph/components.hxx"
#include "nifty/graph/paths.hxx"
#include "nifty/graph/optimization/lifted_multicut/lifted_multicut_base.hxx"
#include "nifty/graph/three_cycles.hxx"
#include "nifty/graph/breadth_first_search.hxx"
#include "nifty/graph/bidirectional_breadth_first_search.hxx"
#include "nifty/graph/depth_first_search.hxx"
#include "nifty/ilp_backend/ilp_backend.hxx"
#include "nifty/graph/detail/contiguous_indices.hxx"
#include "nifty/graph/detail/node_labels_to_edge_labels_iterator.hxx"

namespace nifty{
namespace graph{
namespace lifted_multicut{

    template<class OBJECTIVE, class ILP_SOLVER>
    class LiftedMulticutIlp : public LiftedMulticutBase<OBJECTIVE>
    {
    public: 

        typedef OBJECTIVE Objective;
        typedef LiftedMulticutBase<OBJECTIVE> Base;
        typedef typename Base::VisitorBaseType VisitorBaseType;
        typedef typename Base::VisitorProxyType VisitorProxyType;
        //typedef typename Base::EdgeLabels EdgeLabels;
        typedef typename Base::NodeLabelsType NodeLabelsType;
        typedef ILP_SOLVER IlpSovler;
        typedef typename IlpSovler::Settings IlpSettings;
        typedef typename Objective::Graph Graph;
        typedef typename Objective::LiftedGraph LiftedGraph;

    private:
        typedef ComponentsUfd<Graph> Components;
        typedef detail_graph::EdgeIndicesToContiguousEdgeIndices<Graph> DenseIds;
        typedef DepthFirstSearch<Graph> DfsType;



        template< bool TAKE_UNCUT = true>
        struct GraphSubgraphWithCut {
            GraphSubgraphWithCut(
                const Objective & objective,
                const IlpSovler& ilpSolver, 
                const DenseIds & denseIds
            )
                :   objective_(objective),
                    ilpSolver_(ilpSolver),
                    denseIds_(denseIds)
            {}
            bool useNode(const uint64_t v) const
                { return true; }
            bool useEdge(const uint64_t graphEdge) const{ 
                const auto lifdtedGraphEdge = objective_.graphEdgeInLiftedGraph(graphEdge);
                if(TAKE_UNCUT)
                    return ilpSolver_.label(denseIds_[lifdtedGraphEdge]) <  0.5; 
                else
                    return ilpSolver_.label(denseIds_[lifdtedGraphEdge]) >= 0.5; 
            }

            const Objective & objective_;
            const IlpSovler & ilpSolver_;
            const DenseIds & denseIds_;
        };

    public:

        struct Settings{

            size_t numberOfIterations{0};
            int verbose { 0 };
            bool verboseIlp{false};
            bool addThreeCyclesConstraints{true};
            bool addOnlyViolatedThreeCyclesConstraints{true};
            IlpSettings ilpSettings;
        };

        virtual ~LiftedMulticutIlp(){
            if(ilpSolver_ != nullptr)
                delete ilpSolver_;
        }
        LiftedMulticutIlp(const Objective & objective, const Settings & settings = Settings());


        virtual void optimize(NodeLabelsType & nodeLabels, VisitorBaseType * visitor);
        virtual const Objective & objective() const;


        virtual const NodeLabelsType & currentBestNodeLabels( ){
            return *currentBest_;
        }

        virtual std::string name()const{
            return std::string("LiftedMulticutIlp") + ILP_SOLVER::name();
        }
        virtual void weightsChanged(){

            if(graph_.numberOfEdges()>0){
                if(numberOfOptRuns_<1){
                    ilpSolver_->changeObjective(objective_.weights().begin());
                }
                else{
                    delete ilpSolver_;
                    numberOfOptRuns_ = 0;
                    addedConstraints_ = 0;
                    ilpSolver_ = new IlpSovler(settings_.ilpSettings);
                    this->initializeIlp();
                    if(settings_.addThreeCyclesConstraints){
                        this->addThreeCyclesConstraintsExplicitly();
                    }
                }
            }
        }
        
    private:

        void addThreeCyclesConstraintsExplicitly(const IlpSovler & ilpSolver);
        void initializeIlp();


        void repairSolution(NodeLabelsType & nodeLabels);


        size_t addViolatedInequalities(const bool searchForCutConstraitns, VisitorProxyType & visitor);
        void addThreeCyclesConstraintsExplicitly();

        const Objective & objective_;
        const Graph & graph_;
        const LiftedGraph & liftedGraph_;

        IlpSovler * ilpSolver_;
        Components components_;
        // for all so far existing graphs EdgeIndicesToContiguousEdgeIndices
        // is a zero overhead function which just returns the edge itself
        // since all so far existing graphs have contiguous edge ids
        DenseIds denseIds_;
        BidirectionalBreadthFirstSearch<Graph> bibfs_;
        DfsType dfs_;
        Settings settings_;
        std::vector<size_t> variables_;
        std::vector<double> coefficients_;
        NodeLabelsType * currentBest_;
        size_t addedConstraints_;
        size_t numberOfOptRuns_;
    };

    
    template<class OBJECTIVE, class ILP_SOLVER>
    LiftedMulticutIlp<OBJECTIVE, ILP_SOLVER>::
    LiftedMulticutIlp(
        const Objective & objective, 
        const Settings & settings
    )
    :   objective_(objective),
        graph_(objective.graph()),
        liftedGraph_(objective.liftedGraph()),
        ilpSolver_(nullptr),//settings.ilpSettings),
        components_(graph_),
        denseIds_(liftedGraph_),
        bibfs_(graph_),
        dfs_(graph_),
        settings_(settings),
        variables_(   
            std::max(
                uint64_t(3),
                uint64_t(objective.liftedGraph().numberOfEdges())
            )
        ),
        coefficients_(
            std::max(
                uint64_t(3),
                uint64_t(objective.liftedGraph().numberOfEdges())
            )
        )
    {
        // std::cout << "LiftedMulticutIlp::LiftedMulticutIlp: new IlpSolver\n";
        ilpSolver_ = new ILP_SOLVER(settings_.ilpSettings);
        
        // std::cout << "LiftedMulticutIlp::LiftedMulticutIlp: initializeIlp\n";
        this->initializeIlp();

        // add explicit constraints
        // std::cout << "LiftedMulticutIlp::LiftedMulticutIlp: add3Cycles\n";
        if(settings_.addThreeCyclesConstraints){
            this->addThreeCyclesConstraintsExplicitly();
        }

        // std::cout << "LiftedMulticutIlp::LiftedMulticutIlp: DONE\n";
    }

    template<class OBJECTIVE, class ILP_SOLVER>
    void LiftedMulticutIlp<OBJECTIVE, ILP_SOLVER>::
    optimize(
        NodeLabelsType & nodeLabels,  VisitorBaseType * visitor
    ){  

        // std::cout << "LiftedMulticutIlp::optimize: Start\n";
        VisitorProxyType visitorProxy(visitor);

        visitorProxy.addLogNames({"violatedCycleConstraints","violatedCutConstraints"});
        currentBest_ = &nodeLabels;
        
        visitorProxy.begin(this);


        if(graph_.numberOfEdges()>0){


            // std::cout << "LiftedMulticutIlp::optimize: Set StartintPoint\n";
            // set the starting point 
            auto edgeLabelIter = detail_graph::nodeLabelsToEdgeLabelsIterBegin(liftedGraph_,  nodeLabels);
            ilpSolver_->setStart(edgeLabelIter);


            size_t i=0;
            for (  ; settings_.numberOfIterations == 0 || i < settings_.numberOfIterations; ++i){


                // solve ilp
                // std::cout << "LiftedMulticutIlp::optimize: Optimize Ilp\n";
                ilpSolver_->optimize();

                // std::cout << "LiftedMulticutIlp::optimize: addViolatedInequalities Ilp\n";
                // find violated constraints
                auto nViolated = addViolatedInequalities(false, visitorProxy);


                // std::cout << "LiftedMulticutIlp::optimize: repair Ilp\n";
                // repair the solution
                repairSolution(nodeLabels);

                
                // visit visitor
                if(!visitorProxy.visit(this))
                    break;
                
                
                // exit if we do not violate constraints
                if (nViolated == 0)
                    break;
            }
            // std::cout << "LiftedMulticutIlp::optimize: PHASE 2\n";
            for ( ; settings_.numberOfIterations == 0 || i < settings_.numberOfIterations; ++i){

                // solve ilp
                // std::cout << "LiftedMulticutIlp::optimize: Optimize Ilp\n";
                ilpSolver_->optimize();

                // find violated constraints
                // std::cout << "LiftedMulticutIlp::optimize: addViolatedInequalities Ilp\n";
                auto nViolated = addViolatedInequalities(true, visitorProxy);

                // repair the solution
                // std::cout << "LiftedMulticutIlp::optimize: repair Ilp\n";
                repairSolution(nodeLabels);

                
                // visit visitor
                if(!visitorProxy.visit(this))
                    break;
                
                
                // exit if we do not violate constraints
                if (nViolated == 0)
                    break;
            }
            ++numberOfOptRuns_;
        }
        visitorProxy.end(this);

        // std::cout << "LiftedMulticutIlp::optimize: End\n";
    }

    template<class OBJECTIVE, class ILP_SOLVER>
    const typename LiftedMulticutIlp<OBJECTIVE, ILP_SOLVER>::Objective &
    LiftedMulticutIlp<OBJECTIVE, ILP_SOLVER>::
    objective()const{
        return objective_;
    }

    template<class OBJECTIVE, class ILP_SOLVER>
    size_t LiftedMulticutIlp<OBJECTIVE, ILP_SOLVER>::
    addViolatedInequalities(
        const bool searchForCutConstraitns,
        VisitorProxyType & visitorProxy
    ){
        // std::cout << "LiftedMulticutIlp::addViolatedInequalities: Start\n";
        const auto graphSubgraphWithCutTakeUncut = GraphSubgraphWithCut<true >(objective_, *ilpSolver_, denseIds_);
        const auto graphSubgraphWithCutTakeCut   = GraphSubgraphWithCut<false>(objective_, *ilpSolver_, denseIds_);

        // build cc
        components_.build(graphSubgraphWithCutTakeUncut);

        // search for violated non-chordal cycles and add corresp. inequalities
        // and for violated cut constraints
        size_t nCycleConstraints = 0;
        size_t nCutConstraints = 0;

        // we iterate over edges and the corresponding lpEdge 
        // for a graph with dense contiguous edge ids the lpEdge 
        // is equivalent to the graph edge
        auto lpEdge = 0;
        for (auto edge : liftedGraph_.edges()){

            const auto uv = liftedGraph_.uv(edge);
            const auto v0 = uv.first;
            const auto v1 = uv.second;
            const auto areConnected = components_.areConnected(v0, v1);
            const auto ilpLabel = ilpSolver_->label(lpEdge);

            if (ilpLabel > 0.5 && areConnected){

                auto hasPath = bibfs_.runSingleSourceSingleTarget(v0, v1, graphSubgraphWithCutTakeUncut);
                NIFTY_CHECK(hasPath,"internal error");
                const auto & path = bibfs_.path();
                NIFTY_CHECK_OP(path.size(),>,0,"");
                const auto sz = path.size(); //buildPathInLargeEnoughBuffer(v0, v1, bfs.predecessors(), path.begin());

                bool chordless = true;
                if (findChord(graph_, graphSubgraphWithCutTakeCut, path.begin(), path.end(), true) != -1){
                    chordless = false;
                }

                if(chordless){
                    for (size_t j = 0; j < sz - 1; ++j){
                        const auto v = denseIds_[liftedGraph_.findEdge(path[j], path[j + 1])];
                        NIFTY_ASSERT_OP(v,<,liftedGraph_.numberOfEdges());
                        variables_[j] = v;
                        coefficients_[j] = 1.0;
                    }
                    variables_[sz - 1] = lpEdge;
                    coefficients_[sz - 1] = -1.0;

                    ++addedConstraints_;
                    ilpSolver_->addConstraint(variables_.begin(), variables_.begin() + sz, 
                                             coefficients_.begin(), 0, std::numeric_limits<double>::infinity());
                    ++nCycleConstraints;
                }
            
            }
            else if(searchForCutConstraitns && ilpLabel < 0.5 && !areConnected){

               

                //std::cout<<"Violated Cut Constraint\n";

                auto addConstraintFromCut = [&]( const uint64_t va, const uint64_t la){

                    auto nCut = 0;
                    auto dfsVisitor = [&] (
                        int64_t toNode,int64_t predecessorNode,
                        int64_t edge,int64_t distance, 
                        bool & continueSeach, bool & addToNode
                    ){
                        if(components_.componentLabel(toNode) != la){
                            coefficients_[nCut] = -1;
                            variables_[nCut] = denseIds_[objective_.graphEdgeInLiftedGraph(edge)];
                            addToNode = false;
                            ++nCut;
                        }
                    };
                    DefaultSubgraphMask<Graph> subgraphMask;
                    dfs_.run(&va, &va+1, subgraphMask, dfsVisitor);

                    coefficients_[nCut] = 1.0;
                    variables_[nCut] = lpEdge;

                    ilpSolver_->addConstraint(variables_.begin(), variables_.begin() + nCut, 
                                             coefficients_.begin(), 1.0 - nCut, 
                                             std::numeric_limits<double>::infinity());

                    //std::cout<<"    nCut: "<< nCut<<"\n";
                };

                const auto l0 = components_.componentLabel(v0);
                const auto l1 = components_.componentLabel(v1);
                addConstraintFromCut(v0, l0);
                addConstraintFromCut(v1, l1);

                ++nCutConstraints;

            }

            ++lpEdge;
        }

        // add additional logs
        visitorProxy.setLogValue(0, nCycleConstraints);
        visitorProxy.setLogValue(1, !searchForCutConstraitns ? -1.0 : double(nCutConstraints));
        // std::cout << "LiftedMulticutIlp::addViolatedInequalities: RETURN\n";
        return nCycleConstraints + nCutConstraints;
    }

    template<class OBJECTIVE, class ILP_SOLVER>
    void LiftedMulticutIlp<OBJECTIVE, ILP_SOLVER>::
    repairSolution(
        NodeLabelsType & nodeLabels
    ){
        if(graph_.numberOfEdges()!= 0 ){
            for (auto node: graph_.nodes()){
                nodeLabels[node] = components_.componentLabel(node);
            }
            auto edgeLabelIter = detail_graph::nodeLabelsToEdgeLabelsIterBegin(liftedGraph_, nodeLabels);
            ilpSolver_->setStart(edgeLabelIter);
        }
    }

    template<class OBJECTIVE, class ILP_SOLVER>
    void LiftedMulticutIlp<OBJECTIVE, ILP_SOLVER>::
    initializeIlp(){
        if(liftedGraph_.numberOfEdges()!= 0 ){

            std::vector<double> costs(liftedGraph_.numberOfEdges(),0.0);
            const auto & weights = objective_.weights();
            auto lpEdge = 0;
            for(auto e : liftedGraph_.edges()){
                if(std::abs(weights[e])<=0.00000001){
                    if(weights[e]<0.0)
                        costs[lpEdge] = -0.00000001;
                    else
                        costs[lpEdge] =  0.00000001;
                }
                else
                    costs[lpEdge] = weights[e];
                ++lpEdge;
            }
            ilpSolver_->initModel(liftedGraph_.numberOfEdges(), costs.data());
        }
    }

    template<class OBJECTIVE, class ILP_SOLVER>
    void LiftedMulticutIlp<OBJECTIVE, ILP_SOLVER>::
    addThreeCyclesConstraintsExplicitly(
    ){
        
        std::array<size_t, 3> variables;
        std::array<double, 3> coefficients;
        auto threeCycles = findThreeCyclesEdges(graph_);
        auto c = 0;
        if(!settings_.addOnlyViolatedThreeCyclesConstraints){
            for(const auto & tce : threeCycles){
                for(auto i=0; i<3; ++i){
                    variables[i] = denseIds_[objective_.graphEdgeInLiftedGraph(tce[i])];
                }
                for(auto i=0; i<3; ++i){
                    for(auto j=0; j<3; ++j){
                        if(i != j){
                            coefficients[j] = 1.0;
                        }
                    }
                    coefficients[i] = -1.0;
                    ++addedConstraints_;
                    ilpSolver_->addConstraint(variables.begin(), variables.begin() + 3, 
                        coefficients.begin(), 0, std::numeric_limits<double>::infinity());
                    ++c;
                }
            }
        }
        else{
            const auto & weights = objective_.weights();
            for(const auto & tce : threeCycles){
                // count negative edges
                auto nNeg = 0 ;
                auto negIndex = 0;
                for(auto i=0; i<3; ++i){
                    const auto edge = tce[i];
                    if(weights[edge]<0.0){
                        ++nNeg;
                        negIndex = i;
                    }
                }
                if(nNeg == 1){
                    for(auto i=0; i<3; ++i){
                        coefficients[i] = 1.0;
                        variables[i] = denseIds_[objective_.graphEdgeInLiftedGraph(tce[i])];
                    }
                    coefficients[negIndex] = -1.0;
                    ilpSolver_->addConstraint(variables.begin(), variables.begin() + 3, 
                        coefficients.begin(), 0, std::numeric_limits<double>::infinity());
                    ++c;
                }
            }
        }
        //std::cout<<"add three done\n";
        //std::cout<<"added "<<c<<" explicit constraints\n";
        
    }

} // namespace nifty::graph::lifted_multicut
} // namespace nifty::graph
} // namespace nifty

