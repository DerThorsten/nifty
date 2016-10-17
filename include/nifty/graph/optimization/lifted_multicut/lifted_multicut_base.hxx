#pragma once
#ifndef NIFTY_GRAPH_OPTIMIZATION_LIFTED_MULTICUT_LIFTED_MULTICUT_BASE_HXX
#define NIFTY_GRAPH_OPTIMIZATION_LIFTED_MULTICUT_LIFTED_MULTICUT_BASE_HXX

#include <string>
#include <initializer_list>
#include <sstream>
#include <stdexcept>
    
#include "nifty/exceptions/exceptions.hxx"
#include "nifty/graph/optimization/lifted_multicut/lifted_multicut_visitor_base.hxx"

namespace nifty {
namespace graph {
namespace lifted_multicut{


    template<class OBJECTIVE>
    class LiftedMulticutBase{
    
    public:
        typedef OBJECTIVE Objective;
        typedef LiftedMulticutVisitorBase<Objective> VisitorBase;
        typedef LiftedMulticutVisitorProxy<Objective> VisitorProxy;
        typedef typename Objective::GraphType GraphType;
        typedef typename GraphType:: template EdgeMap<uint8_t>  EdgeLabels;
        typedef typename GraphType:: template NodeMap<uint64_t> NodeLabels;

        virtual ~LiftedMulticutBase(){};
        virtual void optimize(NodeLabels & nodeLabels, VisitorBase * visitor) = 0;
        virtual const Objective & objective() const = 0;
        virtual const NodeLabels & currentBestNodeLabels() = 0;


        virtual std::string name() const = 0 ;

        /**
         * @brief Inform solver about a change of weights.
         * @details Inform solver that all weights could have changed. 
         * If a particular solver does not overload this function, an
         * WeightsChangedNotSupported exception is thrown.
         * After a call of this function it is save to run optimize
         * again, therefore it resets the solver
         * 
         * 
         */
        virtual void weightsChanged(){
            std::stringstream ss;
            ss<<this->name()<<" does not support changing weights";
            throw exceptions::WeightsChangedNotSupported(ss.str());
        }   

        

        // with default implementation
        virtual double currentBestEnergy() {
            const auto & nl = this->currentBestNodeLabels();
            const auto & obj = this->objective();
            return obj.evalNodeLabels(nl);
        }



    };

} // namespace lifted_multicut
} // namespace graph
} // namespace nifty

#endif // #ifndef NIFTY_GRAPH_OPTIMIZATION_LIFTED_MULTICUT_LIFTED_MULTICUT_BASE_HXX
