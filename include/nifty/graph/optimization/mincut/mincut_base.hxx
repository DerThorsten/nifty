#pragma once

#include <string>
#include <initializer_list>
#include <sstream>

#include "nifty/exceptions/exceptions.hxx"
#include "mincut_visitor_base.hxx"

namespace nifty {
namespace graph {
namespace optimization{
namespace mincut{


    
    template<class OBJECTIVE>
    class MincutBase{
    
    public:
        typedef OBJECTIVE ObjectiveType;
        typedef MincutVisitorBase<ObjectiveType> VisitorBase;
        typedef MincutVisitorProxy<ObjectiveType> VisitorProxy;
        typedef typename ObjectiveType::Graph Graph;
        typedef typename Graph:: template EdgeMap<uint8_t> EdgeLabels;
        typedef typename Graph:: template NodeMap<uint64_t> NodeLabels;

        virtual ~MincutBase(){};
        virtual void optimize(NodeLabels & nodeLabels, VisitorBase * visitor) = 0;
        virtual const ObjectiveType & objective() const = 0;
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
} // namespace nifty::graph::optimization::mincut
} // namespace nifty::graph::optimization
} // namespace graph
} // namespace nifty

