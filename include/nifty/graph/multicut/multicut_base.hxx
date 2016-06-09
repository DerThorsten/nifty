#pragma once
#ifndef NIFTY_GRAPH_MULTICUT_MULTICUT_BASE_HXX
#define NIFTY_GRAPH_MULTICUT_MULTICUT_BASE_HXX




namespace nifty {
namespace graph {

    template<class OBJECTIVE> 
    class MulticutVisitorBase{
    public:
    private:
    };


    template<class OBJECTIVE> 
    class MulticutVerboseVisitor : public MulticutVisitorBase<OBJECTIVE>{
    public:
    private:
    };


    template<class OBJECTIVE>
    class MulticutBase{
    
    public:
        typedef OBJECTIVE Objective;
        typedef MulticutVisitorBase<OBJECTIVE> VisitorBase;
        typedef typename Objective::Graph Graph;
        typedef typename Graph:: template EdgeMap<uint8_t>  EdgeLabels;
        typedef typename Graph:: template NodeMap<uint64_t> NodeLabels;

        virtual void optimize(NodeLabels & nodeLabels, VisitorBase * visitor) = 0;
        virtual const Objective & objective() const = 0;



        /*
        virtual void setStartNodeLabels(const NodeLabels & ndoeLabels) = 0; 
        virtual void getNodeLabels(NodeLabels & ndoeLabels) = 0;
        virtual uint64_t getNodeLabel(uint64_t node) = 0;
        // with default implementation
        virtual void setStartEdgeLabels(const EdgeLabels & edgeLabels);
        virtual void getEdgeLabels(EdgeLabels & edgeLabels);
        virtual uint8_t getEdgeLabel(uint64_t edge);
    */

    };

} // namespace graph
} // namespace nifty

#endif // #ifndef NIFTY_GRAPH_MULTICUT_MULTICUT_BASE_HXX
