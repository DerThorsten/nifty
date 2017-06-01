#pragma once

#include <string>
#include <initializer_list>
#include <sstream>
#include <iostream>
#include <chrono>

#include "nifty/tools/timer.hxx"
#include "nifty/tools/logging.hxx"

namespace nifty {
namespace graph {
namespace optimization{
namespace common{



    template<class SOLVER> 
    class VisitorBase{
    public:

        typedef SOLVER SolverType;

        // maybe the solver ptr will become a shared ptr
        virtual void begin(SolverType * solver) = 0;
        virtual bool visit(SolverType * solver) = 0;
        virtual void end(SolverType * solver) = 0;


        virtual void clearLogNames(){

        }
        virtual void addLogNames(std::initializer_list<std::string> logNames){

        }
        virtual void setLogValue(const size_t logIndex, double logValue){

        }

        virtual void printLog(const nifty::logging::LogLevel logLevel, const std::string & logString){

        }


    };


    /*
    template<class SOLVER> 
    class VerboseVisitor : public VisitorBase<SOLVER>{
    public:
        typedef SOLVER SolverType;
        typedef std::chrono::seconds TimeType;
        typedef std::chrono::time_point<std::chrono::steady_clock> TimePointType;

        VerboseVisitor(
            const int printNth = 1, 
            const double timeLimit = std::numeric_limits<double>::infinity()
        )
        :   printNth_(printNth),
            runOpt_(true),
            iter_(1),
            timeLimit_(timeLimit),
            runtime_(0)
        {}

        virtual void begin(SolverType * ) {
            std::cout<<"begin inference\n";
            startTime_ = std::chrono::steady_clock::now();
        }
        
        virtual bool visit(SolverType * solver) {
            runtime_ = std::chrono::duration_cast<TimeType>(std::chrono::steady_clock::now() - startTime_).count();
            if(iter_%printNth_ == 0){
                std::stringstream ss;
                ss << "Energy: " << solver->currentBestEnergy() << " ";
                ss << "Runtime: " << runtime_ << " ";
                for(size_t i=0; i<logNames_.size(); ++i){
                    ss<<logNames_[i]<<" "<<logValues_[i]<<" ";
                }
                ss<<"\n";
                std::cout<<ss.str();
            }
            checkRuntime();
            ++iter_;
            return runOpt_;
        }
        
        virtual void end(SolverType * )   {
            std::cout<<"end inference\n";
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
            std::cout<<"LOG["<<int(logLevel)<<"]: "<<logString<<"\n";
        }

        void stopOptimize(){
            runOpt_ = false;
        }
    
    private:
        bool runOpt_;
        int printNth_;
        int iter_;
        
        double timeLimit_;
        TimePointType startTime_;
        size_t runtime_;

        std::vector<std::string> logNames_;
        std::vector<double> logValues_;

        inline void checkRuntime() {
            if(runtime_ > timeLimit_) {
                std::cout << runtime_ << " " << timeLimit_ << std::endl;
                std::cout << "Inference has exceeded time limit and is stopped \n";
                stopOptimize();
            }
        }
    };
    */

    template<class SOLVER> 
    class VerboseVisitor : public VisitorBase<SOLVER>{
    public:
        typedef SOLVER SolverType;
        typedef nifty::tools::Timer TimerType;

        VerboseVisitor(
            const int printNth = 1, 
            const double timeLimitSolver = std::numeric_limits<double>::infinity(),
            const double timeLimitTotal = std::numeric_limits<double>::infinity()
        )
        :   printNth_(printNth),
            runOpt_(true),
            iter_(1),
            timeLimitSolver_(timeLimitSolver),
            timeLimitTotal_(timeLimitTotal),
            runtimeSolver_(0.0),
            runtimeTotal_(0.0)
        {}

        virtual void begin(SolverType * ) {
            std::cout<<"begin inference\n";
            timerSolver_.start();
            timerTotal_.start();
        }
        
        virtual bool visit(SolverType * solver) {
            timerSolver_.stop();
            timerTotal_.stop();
            runtimeTotal_  += timerTotal_.elapsedSeconds();
            timerTotal_.reset().start();
            runtimeSolver_ += timerSolver_.elapsedSeconds();           
            if(iter_%printNth_ == 0){
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
            checkRuntime();
            ++iter_;
            timerSolver_.reset().start();
            return runOpt_;
        }
        
        virtual void end(SolverType * )   {
            std::cout<<"end inference\n";
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
            std::cout<<"LOG["<<int(logLevel)<<"]: "<<logString<<"\n";
        }

        void stopOptimize(){
            runOpt_ = false;
        }
        
        double runtimeSolver() const{
            return runtimeSolver_;
        }
        double runtimeTotal() const{
            return runtimeTotal_;
        }

        double timeLimitTotal() const{
            return timeLimitTotal_;
        }
        double timeLimitSolver() const{
            return timeLimitSolver_;
        }

    private:
        bool runOpt_;
        int printNth_;
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
                std::cout << runtimeSolver_ << " " << timeLimitSolver_ << std::endl;
                std::cout << "Inference has exceeded solver time limit and is stopped \n";
                runOpt_ = false;
            }
            if(runtimeTotal_ > timeLimitTotal_) {
                std::cout << runtimeTotal_ << " " << timeLimitTotal_ << std::endl;
                std::cout << "Inference has exceeded total time limit and is stopped \n";
                runOpt_ = false;
            }
        }
    };


    template<class SOLVER> 
    class EmptyVisitor : public VisitorBase<SOLVER>{
    public:
        typedef SOLVER SolverType;

        virtual void begin(SolverType * solver) {}
        virtual bool visit(SolverType * solver) {return true;}
        virtual void end(SolverType * solver)   {}
    private:
    };



    template<class SOLVER>
    class VisitorProxy{
    public:
        typedef SOLVER SolverType;
        typedef VisitorBase<SOLVER> VisitorBaseTpe;
        VisitorProxy(VisitorBaseTpe * visitor)
        :   visitor_(visitor){

        }

        void addLogNames(std::initializer_list<std::string> logNames){
            if(visitor_  != nullptr){
                visitor_->addLogNames(logNames);
            }
        }
        void begin(SolverType * solver) {
            if(visitor_ != nullptr){
                visitor_->begin(solver);
            }
        }
        bool visit(SolverType * solver) {
            if(visitor_ != nullptr){
                return visitor_->visit(solver);
            }
            return true;
        }
        void end(SolverType * solver)   {
            if(visitor_ != nullptr){
                visitor_->end(solver);
            }
        }
        void clearLogNames()   {
            if(visitor_ != nullptr){
                visitor_->clearLogNames();
            }
        }

        void setLogValue(const size_t logIndex, const double logValue)   {
            if(visitor_ != nullptr){
                visitor_->setLogValue(logIndex, logValue);
            }
        }

        void printLog(const nifty::logging::LogLevel logLevel, const std::string & logString){
            if(visitor_ != nullptr){
                visitor_->printLog(logLevel, logString);
            }
        }
        operator bool() const{
            return visitor_ != nullptr;
        }

    private:
        VisitorBaseTpe * visitor_;
    };




} // namespace nifty::graph::optimization::common
} // namespace nifty::graph::optimization
} // namespace nifty::graph
} // namespace nifty
