#pragma once
#ifndef NIFTY_GRAPH_MULTICUT_MULTICUT_BASE_HXX
#define NIFTY_GRAPH_MULTICUT_MULTICUT_BASE_HXX




namespace nifty {
namespace graph {

    template<class GRAPH>
    class MulticutBase{
    
    public:
        typedef GRAPH Graph;
        typedef typename Graph:: template<uint8_t>  EdgeLabels;
        typedef typename Graph:: template<uint64_t> NodeLabels;

        void optimize() = 0;



        virtual void setStartNodeLabels(const NodeLabels & ndoeLabels) = 0; 
        virtual void getNodeLabels(NodeLabels & ndoeLabels) = 0;
        virtual uint64_t getNodeLabel(uint64_t node) = 0;

        // with default implementation
        virtual void setStartEdgeLabels(const EdgeLabels & edgeLabels);
        virtual void getEdgeLabels(EdgeLabels & edgeLabels);
        virtual uint8_t getEdgeLabel(uint64_t edge);


    };

} // namespace graph
} // namespace nifty

#endif // #ifndef NIFTY_GRAPH_MULTICUT_MULTICUT_BASE_HXX
