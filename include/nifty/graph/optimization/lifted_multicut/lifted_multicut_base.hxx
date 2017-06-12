#pragma once

#include <string>
#include <initializer_list>
#include <sstream>
#include <stdexcept>
    
#include "nifty/exceptions/exceptions.hxx"

#include "nifty/graph/optimization/common/solver_base.hxx"

namespace nifty {
namespace graph {
namespace optimization{
namespace lifted_multicut{



    template<class OBJECTIVE>
    class LiftedMulticutBase :
        public nifty::graph::optimization::common::SolverBase<
            OBJECTIVE,
            LiftedMulticutBase<OBJECTIVE>
        >
    {

    };

    #if 0

    template<class OBJECTIVE>
    class LiftedMulticutBase{
    
    public:
        typedef OBJECTIVE Objective;
        typedef LiftedMulticutVisitorBase<Objective> VisitorBaseType;
        typedef LiftedMulticutVisitorProxy<Objective> VisitorProxy;
        typedef typename Objective::Graph Graph;
        typedef typename Graph:: template EdgeMap<uint8_t>  EdgeLabels;
        typedef typename Graph:: template NodeMap<uint64_t> NodeLabels;

        virtual ~LiftedMulticutBase(){};
        virtual void optimize(NodeLabels & nodeLabels, VisitorBaseType * visitor) = 0;
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
    #endif

} // namespace lifted_multicut
} // namespace nifty::graph::optimization
} // namespace graph
} // namespace nifty

