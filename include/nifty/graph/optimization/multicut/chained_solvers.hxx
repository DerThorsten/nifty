#pragma once


#include "nifty/tools/runtime_check.hxx"

#include "nifty/graph/optimization/common/solver_factory_base.hxx"
#include "nifty/graph/optimization/multicut/multicut_base.hxx"
#include "nifty/graph/optimization/multicut/multicut_objective.hxx"






namespace nifty{
namespace graph{
namespace optimization{
namespace multicut{



    template<class OBJECTIVE>
    class ChainedSolvers : public MulticutBase<OBJECTIVE>
    {
    public:

        typedef OBJECTIVE Objective;
        typedef OBJECTIVE ObjectiveType;
        typedef typename ObjectiveType::WeightType WeightType;
        typedef MulticutBase<ObjectiveType> BaseType;
        typedef typename BaseType::VisitorBase VisitorBase;
        typedef typename BaseType::VisitorProxy VisitorProxy;
        typedef typename BaseType::EdgeLabels EdgeLabels;
        typedef typename BaseType::NodeLabels NodeLabels;
        typedef typename ObjectiveType::Graph Graph;
        typedef typename ObjectiveType::GraphType GraphType;
        typedef typename ObjectiveType::WeightsMap WeightsMap;
        typedef typename GraphType:: template EdgeMap<uint8_t> IsDirtyEdge;


        typedef nifty::graph::optimization::common::SolverFactoryBase<BaseType>  McFactoryBase;




        class NoBeginEndVisitor : public VisitorBase{
        public:

            NoBeginEndVisitor(VisitorBase * visitor)
            :   visitor_(visitor){
            }

            virtual void begin(BaseType * solver) {
                // nothing
            }
            virtual bool visit(BaseType * solver) {
                if(visitor_ != nullptr){
                    return visitor_->visit(solver);
                }
                else{
                    return true;
                }
            }
            virtual void end(BaseType * solver)   {
                // nothing
            }

            virtual void clearLogNames(){
                if(visitor_ != nullptr)
                    visitor_->clearLogNames();
            }
            virtual void addLogNames(std::initializer_list<std::string> logNames){
                if(visitor_ != nullptr)
                    visitor_->addLogNames(logNames);
            }

            virtual void setLogValue(const size_t logIndex, double logValue){
                if(visitor_ != nullptr)
                    visitor_->setLogValue(logIndex, logValue);
            }

            virtual void printLog(const nifty::logging::LogLevel logLevel, const std::string & logString){
                if(visitor_ != nullptr)
                    visitor_->printLog(logLevel, logString);
            }


        private:
            VisitorBase * visitor_;
        };


    public:

        struct SettingsType{
            std::vector<
                std::shared_ptr<McFactoryBase>
            > multicutFactories;
        };

        virtual ~ChainedSolvers(){

        }
        ChainedSolvers(const Objective & objective, const SettingsType & settings = SettingsType());


        virtual void optimize(NodeLabels & nodeLabels, VisitorBase * visitor);
        virtual const Objective & objective() const;


        virtual const NodeLabels & currentBestNodeLabels( ){
            return *currentBest_;
        }

        virtual std::string name()const{
            return std::string("ChainedSolvers");
        }
        virtual void weightsChanged(){
        }
        //virtual double currentBestEnergy() {
        //   return currentBestEnergy_;
        // }
    private:


        const Objective & objective_;
        SettingsType settings_;
        NodeLabels * currentBest_;
        double currentBestEnergy_;

    };


    template<class OBJECTIVE>
    ChainedSolvers<OBJECTIVE>::
    ChainedSolvers(
        const Objective & objective,
        const SettingsType & settings
    )
    :   objective_(objective),
        settings_(settings),
        currentBest_(nullptr)
        //,
        //currentBestEnergy_(std::numeric_limits<double>::infinity())
    {

    }

    template<class OBJECTIVE>
    void ChainedSolvers<OBJECTIVE>::
    optimize(
        NodeLabels & nodeLabels,  VisitorBase * visitor
    ){



        VisitorProxy visitorProxy(visitor);
        NoBeginEndVisitor noBeginEndVisitor(visitor);



        currentBest_ = &nodeLabels;
        //currentBestEnergy_ = objective_.evalNodeLabels(nodeLabels);

        visitorProxy.begin(this);

        for(auto & mcFactory : settings_.multicutFactories){







            auto solver = mcFactory->create(objective_);
            visitorProxy.printLog(nifty::logging::LogLevel::INFO,
                std::string("Starting Solver: ")+solver->name());


            if(visitor != nullptr){
                visitor->clearLogNames();
                solver->optimize(nodeLabels, &noBeginEndVisitor);
            }
            else{
                solver->optimize(nodeLabels, nullptr);
            }
            delete solver;
        }

        visitorProxy.end(this);
    }

    template<class OBJECTIVE>
    const typename ChainedSolvers<OBJECTIVE>::Objective &
    ChainedSolvers<OBJECTIVE>::
    objective()const{
        return objective_;
    }


} // namespace nifty::graph::optimization::multicut
} // namespace nifty::graph::optimization
} // namespace nifty::graph
} // namespace nifty
