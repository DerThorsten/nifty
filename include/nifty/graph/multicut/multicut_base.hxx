#pragma once
#ifndef NIFTY_GRAPH_MULTICUT_MULTICUT_BASE_HXX
#define NIFTY_GRAPH_MULTICUT_MULTICUT_BASE_HXX

#include <string>
#include <initializer_list>
#include <sstream>

namespace nifty {
namespace graph {


    template<class OBJECTIVE>
    class MulticutBase;


    template<class OBJECTIVE> 
    class MulticutVisitorBase{
    public:
        typedef OBJECTIVE Objective;
        typedef MulticutBase<Objective> McBase;

        // maybe the solver ptr will become a shared ptr
        virtual void begin(McBase * solver) = 0;
        virtual bool visit(McBase * solver) = 0;
        virtual void end(McBase * solver) = 0;

        virtual void addLogNames(std::initializer_list<std::string> logNames){

        }
        virtual void setLogValue(const size_t logIndex, double logValue){

        }
    };


    template<class OBJECTIVE> 
    class MulticutVerboseVisitor : public MulticutVisitorBase<OBJECTIVE>{
    public:
        typedef OBJECTIVE Objective;
        typedef MulticutBase<Objective> McBase;

        MulticutVerboseVisitor(const int printNth = 1)
        :   printNth_(printNth),
            runOpt_(true){
        }

        virtual void begin(McBase * solver) {

            std::cout<<"begin inference\n";
        }
        virtual bool visit(McBase * solver) {
            std::stringstream ss;
            ss<<solver->currentBestEnergy()<<" ";
            for(size_t i=0; i<logNames_.size(); ++i){
                ss<<logNames_[i]<<" "<<logValues_[i]<<" ";
            }
            ss<<"\n";
            std::cout<<ss.str();
            return runOpt_;
        }
        virtual void end(McBase * solver)   {
            std::cout<<"end inference\n";
        }
        virtual void addLogNames(std::initializer_list<std::string> logNames){
            logNames_.assign(logNames.begin(), logNames.end());
            logValues_.resize(logNames.size());
        }
        virtual void setLogValue(const size_t logIndex, double logValue){
            logValues_[logIndex] = logValue;
        }
        void stopOptimize(){
            runOpt_ = false;
        }
    private:
        bool runOpt_;
        int printNth_;
        std::vector<std::string> logNames_;
        std::vector<double> logValues_;
    };

    template<class OBJECTIVE> 
    class MulticutEmptyVisitor : public MulticutVisitorBase<OBJECTIVE>{
    public:
        typedef OBJECTIVE Objective;
        typedef MulticutBase<Objective> McBase;
        virtual void begin(McBase * solver) {}
        virtual bool visit(McBase * solver) {return true;}
        virtual void end(McBase * solver)   {}
    private:
    };


    template<class OBJECTIVE>
    class MulticutBase{
    
    public:
        typedef OBJECTIVE Objective;
        typedef MulticutVisitorBase<OBJECTIVE> VisitorBase;
        typedef typename Objective::Graph Graph;
        typedef typename Graph:: template EdgeMap<uint8_t>  EdgeLabels;
        typedef typename Graph:: template NodeMap<uint64_t> NodeLabels;

        virtual ~MulticutBase(){};
        virtual void optimize(NodeLabels & nodeLabels, VisitorBase * visitor) = 0;
        virtual const Objective & objective() const = 0;
        virtual const NodeLabels & currentBestNodeLabels() = 0;


        virtual double currentBestEnergy() {
            const auto & nl = this->currentBestNodeLabels();
            const auto & obj = this->objective();
            return obj.evalNodeLabels(nl);
        }

        /*
        virtual void setStartNodeLabels(const NodeLabels & ndoeLabels) = 0; 
        virtual void getNodeLabels(NodeLabels & ndoeLabels) = 0;
        virtual uint64_t getNodeLabel(uint64_t node) = 0;
        // with default implementation
        virtual void setStartEdgeLabels(const EdgeLabels & edgeLabels);
        virtual void getEdgeLabels(EdgeLabels & edgeLabels);
        virtual uint8_t getEdgeLabel(uint64_t edge);
    */

    };

} // namespace graph
} // namespace nifty

#endif // #ifndef NIFTY_GRAPH_MULTICUT_MULTICUT_BASE_HXX
