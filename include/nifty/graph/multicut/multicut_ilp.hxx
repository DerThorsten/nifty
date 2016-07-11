#pragma once
#ifndef NIFTY_GRAPH_MULTICUT_MULTICUT_ILP_HXX
#define NIFTY_GRAPH_MULTICUT_MULTICUT_ILP_HXX


#include "nifty/graph/components.hxx"
#include "nifty/graph/paths.hxx"
#include "nifty/graph/multicut/multicut_base.hxx"
#include "nifty/graph/three_cycles.hxx"
#include "nifty/graph/breadth_first_search.hxx"
#include "nifty/graph/bidirectional_breadth_first_search.hxx"
#include "nifty/graph/multicut/ilp_backend/ilp_backend.hxx"
#include "nifty/graph/detail/contiguous_indices.hxx"
#include "nifty/graph/detail/node_labels_to_edge_labels_iterator.hxx"

namespace nifty{
namespace graph{


    template<class OBJECTIVE, class ILP_SOLVER>
    class MulticutIlp : public MulticutBase<OBJECTIVE>
    {
    public: 

        typedef OBJECTIVE Objective;
        typedef MulticutBase<OBJECTIVE> Base;
        typedef typename Base::VisitorBase VisitorBase;
        typedef typename Base::VisitorProxy VisitorProxy;
        typedef typename Base::EdgeLabels EdgeLabels;
        typedef typename Base::NodeLabels NodeLabels;
        typedef ILP_SOLVER IlpSovler;
        typedef typename IlpSovler::Settings IlpSettings;
        typedef typename Objective::Graph Graph;

    private:
        typedef ComponentsUfd<Graph> Components;
        typedef detail_graph::EdgeIndicesToContiguousEdgeIndices<Graph> DenseIds;

        struct SubgraphWithCut {
            SubgraphWithCut(const IlpSovler& ilpSolver, const DenseIds & denseIds)
                :   ilpSolver_(ilpSolver),
                    denseIds_(denseIds)
            {}
            bool useNode(const size_t v) const
                { return true; }
            bool useEdge(const size_t e) const
                { return ilpSolver_.label(denseIds_[e]) == 0; }

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

        virtual ~MulticutIlp(){
            if(ilpSolver_ != nullptr)
                delete ilpSolver_;
        }
        MulticutIlp(const Objective & objective, const Settings & settings = Settings());


        virtual void optimize(NodeLabels & nodeLabels, VisitorBase * visitor);
        virtual const Objective & objective() const;


        virtual const NodeLabels & currentBestNodeLabels( ){
            return *currentBest_;
        }

        virtual std::string name()const{
            return std::string("MulticutIlp") + ILP_SOLVER::name();
        }
        virtual void weightsChanged(){

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
        
    private:

        void addThreeCyclesConstraintsExplicitly(const IlpSovler & ilpSolver);
        void initializeIlp();


        void repairSolution(NodeLabels & nodeLabels);


        size_t addCycleInequalities();
        void addThreeCyclesConstraintsExplicitly();

        const Objective & objective_;
        const Graph & graph_;

        IlpSovler * ilpSolver_;
        Components components_;
        // for all so far existing graphs EdgeIndicesToContiguousEdgeIndices
        // is a zero overhead function which just returns the edge itself
        // since all so far existing graphs have contiguous edge ids
        DenseIds denseIds_;
        BidirectionalBreadthFirstSearch<Graph> bibfs_;
        Settings settings_;
        std::vector<size_t> variables_;
        std::vector<double> coefficients_;
        NodeLabels * currentBest_;
        size_t addedConstraints_;
        size_t numberOfOptRuns_;
    };

    
    template<class OBJECTIVE, class ILP_SOLVER>
    MulticutIlp<OBJECTIVE, ILP_SOLVER>::
    MulticutIlp(
        const Objective & objective, 
        const Settings & settings
    )
    :   objective_(objective),
        graph_(objective.graph()),
        ilpSolver_(nullptr),//settings.ilpSettings),
        components_(graph_),
        denseIds_(graph_),
        bibfs_(graph_),
        settings_(settings),
        variables_(   std::max(uint64_t(3),uint64_t(graph_.numberOfEdges()))),
        coefficients_(std::max(uint64_t(3),uint64_t(graph_.numberOfEdges())))
    {
        ilpSolver_ = new ILP_SOLVER(settings_.ilpSettings);
        if (settings_.verbose >=2)
                std::cout<<"init cplex \n";
        this->initializeIlp();
        if (settings_.verbose >=2)
                std::cout<<"init cplex done \n";

        // add explicit constraints
        if(settings_.addThreeCyclesConstraints){
            this->addThreeCyclesConstraintsExplicitly();
        }
    }

    template<class OBJECTIVE, class ILP_SOLVER>
    void MulticutIlp<OBJECTIVE, ILP_SOLVER>::
    optimize(
        NodeLabels & nodeLabels,  VisitorBase * visitor
    ){  
        //std::cout<<"nStartConstraints "<<addedConstraints_<<"\n";
        VisitorProxy visitorProxy(visitor);

        visitorProxy.addLogNames({"violatedConstraints"});

        currentBest_ = &nodeLabels;
        
        visitorProxy.begin(this);

        // set the starting point 
        auto edgeLabelIter = detail_graph::nodeLabelsToEdgeLabelsIterBegin(graph_, nodeLabels);
        ilpSolver_->setStart(edgeLabelIter);

        for (size_t i = 0; settings_.numberOfIterations == 0 || i < settings_.numberOfIterations; ++i){

            // solve ilp
            ilpSolver_->optimize();

            // find violated constraints
            auto nViolated = addCycleInequalities();

            // repair the solution
            repairSolution(nodeLabels);

            // add additional logs
            visitorProxy.setLogValue(0,nViolated);
            // visit visitor
            if(!visitorProxy.visit(this))
                break;
            
            
            // exit if we do not violate constraints
            if (nViolated == 0)
                break;
        }
        ++numberOfOptRuns_;
        visitorProxy.end(this);
    }

    template<class OBJECTIVE, class ILP_SOLVER>
    const typename MulticutIlp<OBJECTIVE, ILP_SOLVER>::Objective &
    MulticutIlp<OBJECTIVE, ILP_SOLVER>::
    objective()const{
        return objective_;
    }

    template<class OBJECTIVE, class ILP_SOLVER>
    size_t MulticutIlp<OBJECTIVE, ILP_SOLVER>::
    addCycleInequalities(
    ){

        components_.build(SubgraphWithCut(*ilpSolver_, denseIds_));

        // search for violated non-chordal cycles and add corresp. inequalities
        size_t nCycle = 0;


        // we iterate over edges and the corresponding lpEdge 
        // for a graph with dense contiguous edge ids the lpEdge 
        // is equivalent to the graph edge
        auto lpEdge = 0;
        for (auto edge : graph_.edges()){
            if (ilpSolver_->label(lpEdge) > 0.5){

                auto v0 = graph_.u(edge);
                auto v1 = graph_.v(edge);

                if (components_.areConnected(v0, v1)){   

                    auto hasPath = bibfs_.runSingleSourceSingleTarget(v0, v1, SubgraphWithCut(*ilpSolver_, denseIds_));
                    NIFTY_CHECK(hasPath,"damn");
                    const auto & path = bibfs_.path();
                    NIFTY_CHECK_OP(path.size(),>,0,"");
                    const auto sz = path.size(); //buildPathInLargeEnoughBuffer(v0, v1, bfs.predecessors(), path.begin());


                    if (findChord(graph_, path.begin(), path.end(), true) != -1){
                        ++lpEdge;
                        continue;
                    }

                    for (size_t j = 0; j < sz - 1; ++j){
                        variables_[j] = denseIds_[graph_.findEdge(path[j], path[j + 1])];
                        coefficients_[j] = 1.0;
                    }
                    variables_[sz - 1] = lpEdge;
                    coefficients_[sz - 1] = -1.0;

                    ++addedConstraints_;
                    ilpSolver_->addConstraint(variables_.begin(), variables_.begin() + sz, 
                                             coefficients_.begin(), 0, std::numeric_limits<double>::infinity());
                    ++nCycle;
                }
            }
            ++lpEdge;
        }
        return nCycle;
    }

    template<class OBJECTIVE, class ILP_SOLVER>
    void MulticutIlp<OBJECTIVE, ILP_SOLVER>::
    repairSolution(
        NodeLabels & nodeLabels
    ){
        for (auto node: graph_.nodes()){
            nodeLabels[node] = components_.componentLabel(node);
        }
        auto edgeLabelIter = detail_graph::nodeLabelsToEdgeLabelsIterBegin(graph_, nodeLabels);
        ilpSolver_->setStart(edgeLabelIter);
    }

    template<class OBJECTIVE, class ILP_SOLVER>
    void MulticutIlp<OBJECTIVE, ILP_SOLVER>::
    initializeIlp(){
        std::vector<double> costs(graph_.numberOfEdges(),0.0);
        const auto & weights = objective_.weights();
        auto lpEdge = 0;
        for(auto e : graph_.edges()){
            if(std::abs(weights[e])<=0.00000001){
                if(weights[e]<0.0){
                    costs[lpEdge] = -0.00000001;
                }
                else{
                    costs[lpEdge] =  0.00000001;
                }
            }
            else{
                costs[lpEdge] = weights[e];
            }
            ++lpEdge;
        }
        ilpSolver_->initModel(graph_.numberOfEdges(), costs.data());
    }

    template<class OBJECTIVE, class ILP_SOLVER>
    void MulticutIlp<OBJECTIVE, ILP_SOLVER>::
    addThreeCyclesConstraintsExplicitly(
    ){
        std::array<size_t, 3> variables;
        std::array<double, 3> coefficients;
        auto threeCycles = findThreeCyclesEdges(graph_);
        auto c = 0;
        if(!settings_.addOnlyViolatedThreeCyclesConstraints){
            for(const auto & tce : threeCycles){
                for(auto i=0; i<3; ++i){
                    variables[i] = tce[i];
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
                        variables[i] = tce[i];
                    }
                    coefficients[negIndex] = -1.0;
                    ilpSolver_->addConstraint(variables.begin(), variables.begin() + 3, 
                        coefficients.begin(), 0, std::numeric_limits<double>::infinity());
                    ++c;
                }
            }
        }
        //std::cout<<"added "<<c<<" explicit constraints\n";
    }


} // namespace nifty::graph
} // namespace nifty

#endif  // NIFTY_GRAPH_MULTICUT_MULTICUT_ILP_HXX
