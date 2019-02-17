#pragma once


#include "nifty/tools/runtime_check.hxx"
#include "nifty/graph/components.hxx"
#include "nifty/graph/paths.hxx"
#include "nifty/graph/opt/multicut/multicut_base.hxx"
#include "nifty/graph/three_cycles.hxx"
#include "nifty/graph/breadth_first_search.hxx"
#include "nifty/graph/bidirectional_breadth_first_search.hxx"
#include "nifty/ilp_backend/ilp_backend.hxx"
#include "nifty/graph/detail/contiguous_indices.hxx"
#include "nifty/graph/detail/node_labels_to_edge_labels_iterator.hxx"

namespace nifty{
namespace graph{
namespace opt{
namespace multicut{

    /*!
    @brief ILP Cutting plane multicut solver
    @details ILP Cutting plane multicut solver
    adding violated cycle constraints in an 
    iterative fashion.

    \if HTML_BUILD
    <b>Cite:</b> \cite Kappes-2011 \cite andres_2011_probabilistic
    \endif

    \breathe
    **Cite:** :cite:`Kappes-2011` :cite:`andres_2011_probabilistic`
    \endbreathe


    \breathe
    **Corresponding Python Classes and their template Instantiations**:
        
    *   :class:`nifty.graph.opt.multicut.MulticutIlpCplexMulticutObjectiveUndirectedGraph` 
    
        *  **OBJECTIVE** : 
        

            **C++:** :cpp:class:`nifty::graph::opt::multicut::MulticutObjective`

            **Python:** :class:`nifty.graph.opt.multicut.MulticutObjectiveUndirectedGraph` 
            

        *  **ILP_SOLVER** : 
        

            **C++:** :cpp:class:`nifty::ilp_backend::Cplex`
             

            **Python:** - 

    *   :class:`nifty.graph.opt.multicut.MulticutIlpGlpkMulticutObjectiveUndirectedGraph` 

        *  **OBJECTIVE** : 
        

            **C++:** :cpp:class:`nifty::graph::opt::multicut::MulticutObjective`
        

            **Python:** :class:`nifty.graph.opt.multicut.MulticutObjectiveUndirectedGraph` 
            
        *  **ILP_SOLVER** :  
          
         
            **C++:** :cpp:class:`nifty::ilp_backend::Glpk`
             
             
            **Python:**  - 


        






    \endbreathe
     
     
     
    @ingroup group_multicut_solver
    @tparam OBJECTIVE The multicut objective (e.g. MulticutObjective)
    @tparam ILP_SOLVER The ILP solver backend (e.g. ilp_backend::Cplex, ilp_backend::Glpk, ilp_backend::Gurobi)
    */
    template<class OBJECTIVE, class ILP_SOLVER>
    class MulticutIlp : public MulticutBase<OBJECTIVE>
    {
    public: 

        typedef OBJECTIVE ObjectiveType;
        typedef typename ObjectiveType::GraphType GraphType;
        
        /// \brief Base Type / parent class
        typedef MulticutBase<OBJECTIVE> BaseType;

        /// Visitor base class
        typedef typename BaseType::VisitorBaseType VisitorBaseType;
        
        typedef typename BaseType::NodeLabelsType NodeLabelsType;
        typedef ILP_SOLVER IlpSovler;
        typedef typename IlpSovler::SettingsType IlpSettingsType;

    private:

        typedef typename BaseType::VisitorProxyType VisitorProxyType;
        typedef ComponentsUfd<GraphType> Components;
        typedef detail_graph::EdgeIndicesToContiguousEdgeIndices<GraphType> DenseIds;

        struct SubgraphWithCut {
            SubgraphWithCut(const IlpSovler& ilpSolver, const DenseIds & denseIds)
                :   ilpSolver_(ilpSolver),
                    denseIds_(denseIds)
            {}
            bool useNode(const std::size_t v) const
                { return true; }
            bool useEdge(const std::size_t e) const
                { return ilpSolver_.label(denseIds_[e]) == 0; }

            const IlpSovler & ilpSolver_;
            const DenseIds & denseIds_;
        };

    public:

        /**
         * @brief Settings for MulticutIlp solver.
         * @details The settings for MulticutIlp
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
            std::size_t numberOfIterations{0};   

            /**
             *  \brief Explicitly add constrains for cycle of length three.
             *  \details   Should constrains for cycles of length three
             *  be added explicitly before the actual opt.
             *  This can speedup the opt process.
             */
            bool addThreeCyclesConstraints{true};
                  
            /**
             *  \brief Explicitly add violated constrains for cycle of length three.
             *  \details If addThreeCyclesConstraints is true, 
             *  should only possible violating constraints
             *  be added before the actual opt.
             *  This can speedup the opt process.
             */
            bool addOnlyViolatedThreeCyclesConstraints{true};

            /**
             *   \brief Settings of the ILP backend.
             *   \detailed ILP related options like relative and
             *   absolute gaps can be specified
             */
            IlpSettingsType ilpSettings{};
        };

        virtual ~MulticutIlp(){
            if(ilpSolver_ != nullptr)
                delete ilpSolver_;
        }
        MulticutIlp(const ObjectiveType & objective, const SettingsType & settings = SettingsType());


        virtual void optimize(NodeLabelsType & nodeLabels, VisitorBaseType * visitor);
        virtual const ObjectiveType & objective() const;


        virtual const NodeLabelsType & currentBestNodeLabels( ){
            return *currentBest_;
        }

        virtual std::string name()const{
            return std::string("MulticutIlp") + ILP_SOLVER::name();
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


        std::size_t addCycleInequalities();
        void addThreeCyclesConstraintsExplicitly();

        const ObjectiveType & objective_;
        const GraphType & graph_;

        IlpSovler * ilpSolver_;
        Components components_;
        // for all so far existing graphs EdgeIndicesToContiguousEdgeIndices
        // is a zero overhead function which just returns the edge itself
        // since all so far existing graphs have contiguous edge ids
        DenseIds denseIds_;
        BidirectionalBreadthFirstSearch<GraphType> bibfs_;
        SettingsType settings_;
        std::vector<std::size_t> variables_;
        std::vector<double> coefficients_;
        NodeLabelsType * currentBest_;
        std::size_t addedConstraints_;
        std::size_t numberOfOptRuns_;
    };

    
    template<class OBJECTIVE, class ILP_SOLVER>
    MulticutIlp<OBJECTIVE, ILP_SOLVER>::
    MulticutIlp(
        const ObjectiveType & objective, 
        const SettingsType & settings
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
        
        this->initializeIlp();

        // add explicit constraints
        if(settings_.addThreeCyclesConstraints){
            this->addThreeCyclesConstraintsExplicitly();
        }
    }

    template<class OBJECTIVE, class ILP_SOLVER>
    void MulticutIlp<OBJECTIVE, ILP_SOLVER>::
    optimize(
        NodeLabelsType & nodeLabels,  VisitorBaseType * visitor
    ){  

        //std::cout<<"nStartConstraints "<<addedConstraints_<<"\n";
        VisitorProxyType visitorProxy(visitor);

        visitorProxy.addLogNames({"violatedConstraints"});

        currentBest_ = &nodeLabels;
        
        visitorProxy.begin(this);
        if(graph_.numberOfEdges()>0){
            // set the starting point 
            auto edgeLabelIter = detail_graph::nodeLabelsToEdgeLabelsIterBegin(graph_, nodeLabels);
            ilpSolver_->setStart(edgeLabelIter);

            for (std::size_t i = 0; settings_.numberOfIterations == 0 || i < settings_.numberOfIterations; ++i){

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
        }
        visitorProxy.end(this);
    }

    template<class OBJECTIVE, class ILP_SOLVER>
    const typename MulticutIlp<OBJECTIVE, ILP_SOLVER>::ObjectiveType &
    MulticutIlp<OBJECTIVE, ILP_SOLVER>::
    objective()const{
        return objective_;
    }

    template<class OBJECTIVE, class ILP_SOLVER>
    std::size_t MulticutIlp<OBJECTIVE, ILP_SOLVER>::
    addCycleInequalities(
    ){

        components_.build(SubgraphWithCut(*ilpSolver_, denseIds_));

        // search for violated non-chordal cycles and add corresp. inequalities
        std::size_t nCycle = 0;


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

                    for (std::size_t j = 0; j < sz - 1; ++j){
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
        NodeLabelsType & nodeLabels
    ){
        if(graph_.numberOfEdges()!= 0 ){
            for (auto node: graph_.nodes()){
                nodeLabels[node] = components_.componentLabel(node);
            }
            auto edgeLabelIter = detail_graph::nodeLabelsToEdgeLabelsIterBegin(graph_, nodeLabels);
            ilpSolver_->setStart(edgeLabelIter);
        }
    }

    template<class OBJECTIVE, class ILP_SOLVER>
    void MulticutIlp<OBJECTIVE, ILP_SOLVER>::
    initializeIlp(){
        if(graph_.numberOfEdges()!= 0 ){
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
    }

    template<class OBJECTIVE, class ILP_SOLVER>
    void MulticutIlp<OBJECTIVE, ILP_SOLVER>::
    addThreeCyclesConstraintsExplicitly(
    ){
        //std::cout<<"add three cyckes\n";
        std::array<std::size_t, 3> variables;
        std::array<double, 3> coefficients;
        auto threeCycles = findThreeCyclesEdges(graph_);
        auto c = 0;
        if(!settings_.addOnlyViolatedThreeCyclesConstraints){
            for(const auto & tce : threeCycles){
                for(auto i=0; i<3; ++i){
                    variables[i] = denseIds_[tce[i]];
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
                        variables[i] = denseIds_[tce[i]];
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


} // namespace nifty::graph::opt::multicut
} // namespace nifty::graph::opt
} // namespace nifty::graph
} // namespace nifty

