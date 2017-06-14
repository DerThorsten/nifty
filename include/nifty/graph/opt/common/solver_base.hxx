#pragma once


#include <string>
#include <initializer_list>
#include <sstream>
#include <stdexcept>
    
#include "nifty/exceptions/exceptions.hxx"

#include "nifty/graph/opt/common/solver_base.hxx"
#include "nifty/graph/opt/common/visitor_base.hxx"



namespace nifty {
namespace graph {
namespace opt{
namespace common{

    template<class OBJECTIVE, class CHILD>
    class SolverBase{
    
    public:
        typedef OBJECTIVE                           ObjectiveType;
        typedef SolverBase<ObjectiveType,CHILD>     SelfType;
        typedef VisitorBase<CHILD>                  VisitorBaseType;
        typedef VisitorProxy<CHILD>                 VisitorProxyType;
        typedef typename ObjectiveType::GraphType   GraphType;

        typedef typename ObjectiveType::NodeLabelsType NodeLabelsType;

        virtual ~SolverBase(){};
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

} // namespace nifty::graph::opt::common
} // namespace nifty::graph::opt
} // namespace nifty::graph
} // namespace nifty

