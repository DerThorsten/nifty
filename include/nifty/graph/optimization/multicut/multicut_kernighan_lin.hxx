// reimplementation of kerninhanlin in 
// https://github.com/bjoern-andres/graph
#pragma once

//#include "andres/graph/multicut/kernighan-lin.hxx"
//#include "andres/graph/graph.hxx"

#include "nifty/graph/optimization/multicut/multicut_base.hxx"
#include "nifty/graph/optimization/common/kernighan_lin.hxx"
#include "nifty/graph/optimization/common/detail/twocut_kernighan_lin.hxx"

namespace nifty{
namespace graph{
//namespace optimization{
//namespace multicut{
    
    
    template<class OBJECTIVE>
    class MulticutKernighanLin 
    : public nifty::graph::optimization::common::KernighanLin<
        OBJECTIVE,
        MulticutBase<OBJECTIVE>
      >
    {
    public: 
        typedef OBJECTIVE Objective;
        typedef nifty::graph::optimization::common::KernighanLin<Objective,MulticutBase<Objective>> Base;
        typedef nifty::graph::optimization::common::TwoCut<Objective> TwoCutType;
        
        typedef typename Base::VisitorBase VisitorBase;
        typedef typename Base::NodeLabels NodeLabels;
        typedef typename Base::Settings Settings;
        
        //typedef typename Base::VisitorProxy VisitorProxy;
        
        MulticutKernighanLin(const Objective & objective, const Settings & settings = Settings());

        virtual void optimize(NodeLabels & nodeLabels, VisitorBase * visitor);
        virtual const Objective & objective() const {return Base::objective();}
        virtual const NodeLabels & currentBestNodeLabels() {return Base::currentBestNodeLabels();}
        virtual std::string name() const {return "MulticutKernighanLin";}

    };


    template<class OBJECTIVE>
    MulticutKernighanLin<OBJECTIVE>::MulticutKernighanLin(
        const Objective & objective, 
        const Settings & settings
    ) : Base(objective, TwoCutType( objective, objective.graph() ), settings)
    {}

    template<class OBJECTIVE>
    void MulticutKernighanLin<OBJECTIVE>::optimize(
        NodeLabels & nodeLabels,  VisitorBase * visitor
    ){  
        Base::optimize(nodeLabels, visitor);
    }

//}
//}
}
}
