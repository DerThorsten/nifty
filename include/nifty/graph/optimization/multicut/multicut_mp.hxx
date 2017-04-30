#pragma once

#include "nifty/tools/runtime_check.hxx"
#include "nifty/graph/optimization/multicut/multicut_base.hxx"

// TODO LP_MP includes

namespace nifty{
namespace graph{

    // TODO expose the primal solver for the mp multicut, maybe by template, depending 
    // on how we implement this in LP_MP
    //template<class OBJECTIVE, class PRIMAL_SOLVER>
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

    // TODO I think we can get rid of all this
    //private:
    //    typedef ComponentsUfd<Graph> Components;
    //    typedef detail_graph::EdgeIndicesToContiguousEdgeIndices<Graph> DenseIds;

    //    struct SubgraphWithCut {
    //        SubgraphWithCut(const IlpSovler& ilpSolver, const DenseIds & denseIds)
    //            :   ilpSolver_(ilpSolver),
    //                denseIds_(denseIds)
    //        {}
    //        bool useNode(const size_t v) const
    //            { return true; }
    //        bool useEdge(const size_t e) const
    //            { return ilpSolver_.label(denseIds_[e]) == 0; }

    //        const IlpSovler & ilpSolver_;
    //        const DenseIds & denseIds_;
    //    };

    public:

        // TODO LP_MP settings
        struct Settings{

            size_t numberOfIterations{0};
            int verbose {0};
        };

        virtual ~MulticutMp(){
        }
        MulticutIlp(const Objective & objective, const Settings & settings = Settings());


        virtual void optimize(NodeLabels & nodeLabels, VisitorBase * visitor);
        
        virtual const Objective & objective() const{return objective_};
        virtual const NodeLabels & currentBestNodeLabels( ){return *currentBest_;}

        virtual std::string name()const{
            return std::string("MulticutMp"); // TODO primal_solver name
        }
        
        // TODO do we need this ?
        //virtual void weightsChanged(){
        //}
        
    private:

        void initializeLpMp();

        const Objective & objective_;
        const Graph & graph_;

        Settings settings_;
        // TODO do we need this ?
        std::vector<size_t> variables_;
        std::vector<double> coefficients_;
        NodeLabels * currentBest_;
        // TODO need num
        //numberOfOptRuns_; again
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
        variables_(   std::max(uint64_t(3),uint64_t(graph_.numberOfEdges()))),
        coefficients_(std::max(uint64_t(3),uint64_t(graph_.numberOfEdges())))
    {
        // TODO initialize LP_MP multicut
    }

    template<class OBJECTIVE, class ILP_SOLVER>
    void MulticutIlp<OBJECTIVE, ILP_SOLVER>::
    optimize(
        NodeLabels & nodeLabels,  VisitorBase * visitor
    ){  

        //std::cout<<"nStartConstraints "<<addedConstraints_<<"\n";
        VisitorProxy visitorProxy(visitor);
        currentBest_ = &nodeLabels;
        
        // TODO for now the visitor is doing nothing, but we should implement one, that is
        // compatible with lp_mp visitor
        //visitorProxy.begin(this);
        
        if(graph_.numberOfEdges()>0){

            // TODO for now only run lp_mp once,
            // then integrate the solver properly
            
            // set the starting point 
            //auto edgeLabelIter = detail_graph::nodeLabelsToEdgeLabelsIterBegin(graph_, nodeLabels);
            //ilpSolver_->setStart(edgeLabelIter);

            //for (size_t i = 0; settings_.numberOfIterations == 0 || i < settings_.numberOfIterations; ++i){

            //    // solve ilp
            //    ilpSolver_->optimize();

            //    // add additional logs
            //    visitorProxy.setLogValue(0,nViolated);
            //    // visit visitor
            //    if(!visitorProxy.visit(this))
            //        break;
            //    
            //}
            //++numberOfOptRuns_;
        }
        visitorProxy.end(this);
    }

} // namespace nifty::graph
} // namespace nifty
