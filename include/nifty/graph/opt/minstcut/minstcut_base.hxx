#pragma once

#include <string>
#include <initializer_list>
#include <sstream>

#include "nifty/graph/opt/common/solver_base.hxx"
#include "minstcut_visitor_base.hxx"

namespace nifty {
namespace graph {
namespace opt{
namespace minstcut{


    template<class OBJECTIVE>
    class MinstcutBase :
        public nifty::graph::opt::common::SolverBase<
            OBJECTIVE,
            MinstcutBase<OBJECTIVE>
        >
    {

    };

    #if 0
    
    template<class OBJECTIVE>
    class MinstcutBase{
    
    public:
        typedef OBJECTIVE ObjectiveType;
        typedef MinstcutVisitorBase<ObjectiveType> VisitorBaseType;
        typedef MinstcutVisitorProxy<ObjectiveType> VisitorProxy;
        typedef typename ObjectiveType::Graph Graph;
        typedef typename Graph:: template NodeMap<uint64_t> NodeLabelsType;

        virtual ~MinstcutBase(){};
        virtual void optimize(NodeLabelsType & nodeLabels, VisitorBaseType * visitor) = 0;
        virtual const ObjectiveType & objective() const = 0;
        virtual const NodeLabelsType & currentBestNodeLabels() = 0;


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

} // namespace nifty::graph::opt::minstcut
} // namespace nifty::graph::opt
} // namespace graph
} // namespace nifty

