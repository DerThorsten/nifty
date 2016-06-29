#pragma once
#ifndef NIFTY_GRAPH_MULTICUT_MULTICUT_BASE_HXX
#define NIFTY_GRAPH_MULTICUT_MULTICUT_BASE_HXX

#include <string>
#include <initializer_list>
#include <sstream>

#include "nifty/graph/multicut/multicut_visitor_base.hxx"

namespace nifty {
namespace graph {







    template<class OBJECTIVE>
    class MulticutBase{
    
    public:
        typedef OBJECTIVE Objective;
        typedef MulticutVisitorBase<Objective> VisitorBase;
        typedef MulticutVisitorProxy<Objective> VisitorProxy;
        typedef typename Objective::Graph Graph;
        typedef typename Graph:: template EdgeMap<uint8_t>  EdgeLabels;
        typedef typename Graph:: template NodeMap<uint64_t> NodeLabels;

        virtual ~MulticutBase(){};
        virtual void optimize(NodeLabels & nodeLabels, VisitorBase * visitor) = 0;
        virtual const Objective & objective() const = 0;
        virtual const NodeLabels & currentBestNodeLabels() = 0;


        //// inform the solver that the objective has changed
        //virtual void weightsChanged(){
        //}   

        

        // with default implementation
        virtual double currentBestEnergy() {
            const auto & nl = this->currentBestNodeLabels();
            const auto & obj = this->objective();
            return obj.evalNodeLabels(nl);
        }



    };

} // namespace graph
} // namespace nifty

#endif // #ifndef NIFTY_GRAPH_MULTICUT_MULTICUT_BASE_HXX
