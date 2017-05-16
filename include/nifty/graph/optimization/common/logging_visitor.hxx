#pragma once

#include <string>
#include <initializer_list>
#include <sstream>
#include <iostream>
#include <chrono>

#include "nifty/tools/timer.hxx"
#include "nifty/tools/logging.hxx"
#include "nifty/graph/optimization/common/visitor_base.hxx"

namespace nifty {
namespace graph {
namespace optimization{
namespace common{



    //template<class SOLVER> 
    //class VisitorBase;
    

    template<class SOLVER> 
    class LogginVisitor : public VisitorBase<SOLVER>{
    public:
        typedef SOLVER SolverType;
        typedef nifty::tools::Timer TimerType;

        LogginVisitor(
            const int visitNth = 1, 
            const bool verbose = true,
            const double timeLimitSolver = std::numeric_limits<double>::infinity(),
            const double timeLimitTotal  = std::numeric_limits<double>::infinity()
        )
        :   visitNth_(visitNth),
            verbose_(true),
            runOpt_(true),
            iter_(1),
            timeLimitSolver_(timeLimitSolver),
            timeLimitTotal_(timeLimitTotal),
            runtimeSolver_(0.0),
            runtimeTotal_(0.0)
        {}

        virtual void begin(SolverType * ) {
            if(verbose_){
                std::cout<<"begin inference\n";
            }
            timerSolver_.start();
            timerTotal_.start();
        }
        
        virtual bool visit(SolverType * solver) {
            timerSolver_.stop();
            timerTotal_.stop();
            runtimeTotal_  += timerTotal_.elapsedSeconds();
            timerTotal_.reset().start();
            runtimeSolver_ += timerSolver_.elapsedSeconds();           
            if(iter_%visitNth_ == 0){
                if(verbose_){
                    std::stringstream ss;
                    ss << "E: " << solver->currentBestEnergy() << " ";
                    ss << "t[s]: " << runtimeSolver_ << " ";
                    ss << "/ " << runtimeTotal_ << " ";
                    for(size_t i=0; i<logNames_.size(); ++i){
                        ss<<logNames_[i]<<" "<<logValues_[i]<<" ";
                    }
                    ss<<"\n";
                    std::cout<<ss.str();
                }
            }
            checkRuntime();
            ++iter_;
            timerSolver_.reset().start();
            return runOpt_;
        }
        
        virtual void end(SolverType * )   {
            if(verbose_){
                std::cout<<"end inference\n";
            }
            timerSolver_.stop();
        }
        
        virtual void clearLogNames(){
            logNames_.clear();
            logValues_.clear();
        }
        virtual void addLogNames(std::initializer_list<std::string> logNames){
            logNames_.assign(logNames.begin(), logNames.end());
            logValues_.resize(logNames.size());
        }
        
        virtual void setLogValue(const size_t logIndex, double logValue){
            logValues_[logIndex] = logValue;
        }

        virtual void printLog(const nifty::logging::LogLevel logLevel, const std::string & logString){
            if(verbose_){
                std::cout<<"LOG["<<int(logLevel)<<"]: "<<logString<<"\n";
            }
        }

        void stopOptimize(){
            runOpt_ = false;
        }
    
    private:
       
        int visitNth_;
        bool verbose_;
        bool runOpt_;
        int iter_;
        
        double timeLimitTotal_;
        double timeLimitSolver_;
        double runtimeSolver_;
        double runtimeTotal_;
        TimerType timerSolver_;
        TimerType timerTotal_;
        std::vector<std::string> logNames_;
        std::vector<double> logValues_;

        inline void checkRuntime() {
            if(runtimeSolver_ > timeLimitSolver_) {
                if(verbose_){
                    std::cout << runtimeSolver_ << " " << timeLimitSolver_ << std::endl;
                    std::cout << "Inference has exceeded solver time limit and is stopped \n";
                }
                runOpt_ = false;
            }
            if(runtimeTotal_ > timeLimitTotal_) {
                if(verbose_){
                    std::cout << runtimeTotal_ << " " << timeLimitTotal_ << std::endl;
                    std::cout << "Inference has exceeded total time limit and is stopped \n";
                }
                runOpt_ = false;
            }
        }
    };






} // namespace nifty::graph::optimization::common
} // namespacen ifty::graph::optimization
} // namespace nifty::graph
} // namespace nifty
