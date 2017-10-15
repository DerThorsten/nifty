#pragma once

#include <string>
#include <initializer_list>
#include <sstream>

#include "nifty/graph/opt/common/solver_base.hxx"
#include "nifty/graph/opt/multicut/multicut_visitor_base.hxx"

namespace nifty {
namespace graph {
namespace opt{
namespace multicut{


    template<class OBJECTIVE>
    class MulticutBase :
        public nifty::graph::opt::common::SolverBase<
            OBJECTIVE,
            MulticutBase<OBJECTIVE>
        >
    {

    };


    #if 0
    template<class OBJECTIVE>
    class MulticutBase{
    
    public:
        typedef OBJECTIVE Objective;
        typedef OBJECTIVE ObjectiveType;
        typedef MulticutVisitorBase<Objective> VisitorBaseType;
        typedef MulticutVisitorProxy<Objective> VisitorProxy;
        typedef typename Objective::Graph Graph;
        typedef Graph GraphType;
        typedef typename Graph:: template EdgeMap<uint8_t>  EdgeLabels;
        typedef typename Graph:: template NodeMap<uint64_t> NodeLabels;

        virtual ~MulticutBase(){};
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
} // namespace nifty::graph::opt::multicut
} // namespace nifty::graph::opt
} // namespace graph
} // namespace nifty

