#pragma once

#include <cstddef>
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
    //class VisitorBaseType;


    template<class SOLVER>
    class LoggingVisitor : public VisitorBase<SOLVER>{
    public:
        typedef SOLVER SolverType;
        typedef nifty::tools::Timer TimerType;

        LoggingVisitor(
            const int visitNth = 1,
            const bool verbose = true,
            const double timeLimitSolver = std::numeric_limits<double>::infinity(),
            const double timeLimitTotal  = std::numeric_limits<double>::infinity(),
            const nifty::logging::LogLevel logLevel = nifty::logging::LogLevel::WARN
        )
        :   visitNth_(visitNth),
            verbose_(verbose),
            runOpt_(true),
            iter_(1),
            timeLimitSolver_(timeLimitSolver),
            timeLimitTotal_(timeLimitTotal),
            runtimeSolver_(0.0),
            runtimeTotal_(0.0),
            logLevel_(logLevel),
            logNames_(),
            logValues_(),
            iterations_(),
            energies_(),
            runtimes_()
        {

        }

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
            runtimeSolver_ +=  timerSolver_.elapsedSeconds();
            if(iter_%visitNth_ == 0){


                const auto e = solver->currentBestEnergy();

                iterations_.push_back(iter_);
                energies_.push_back(e);
                runtimes_.push_back(runtimeSolver_);

                if(verbose_){
                    std::stringstream ss;
                    ss << "E: " << e << " ";
                    ss << "t[s]: " << runtimeSolver_ << " ";
                    ss << "/ " << runtimeTotal_ << " ";
                    for(std::size_t i=0; i<logNames_.size(); ++i){
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
        
        virtual void end(SolverType * solver)   {


            timerSolver_.stop();
            timerTotal_.stop();
            runtimeTotal_  += timerTotal_.elapsedSeconds();
            timerTotal_.reset().start();
            runtimeSolver_ +=  timerSolver_.elapsedSeconds();


            const auto e = solver->currentBestEnergy();
            iterations_.push_back(iter_);
            energies_.push_back(e);
            runtimes_.push_back(runtimeSolver_);

            std::stringstream ss;
            ss << "E: " << e << " ";
            ss << "t[s]: " << runtimeSolver_ << " ";
            ss << "/ " << runtimeTotal_ << " ";
            ss<<"\n";
            std::cout<<ss.str();

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

        virtual void setLogValue(const std::size_t logIndex, double logValue){
            logValues_[logIndex] = logValue;
        }

        virtual void printLog(const nifty::logging::LogLevel logLevel, const std::string & logString){
            if(int(logLevel) <= int(logLevel_)){
                std::cout<<"LOG["<<nifty::logging::logLevelName(logLevel)<<"]: "<<logString<<"\n";
            }
        }

        void stopOptimize(){
            runOpt_ = false;
        }

        /**
         * @brief logged iteration number  vector
         * @details iteration number for each logged iteration
         * @return iterations vector
         */
        const std::vector<uint32_t> & iterations()const{
            return iterations_;
        }
        /**
         * @brief logged current best energies vector
         * @details get the current best energies for each logged iteration
         * @return energy vector
         */
        const std::vector<double>   & energies()const{
            return energies_;
        }
        /**
         * @brief logged runtime vector
         * @details get cumulative runtime for each logged iteration
         * @return runtime vector
         */
        const std::vector<double>   & runtimes()const{
            return runtimes_;
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
        nifty::logging::LogLevel logLevel_;
        TimerType timerSolver_;
        TimerType timerTotal_;

        std::vector<std::string>    logNames_;
        std::vector<double>         logValues_;
        // logging vectors
        std::vector<uint32_t> iterations_;
        std::vector<double>   energies_;
        std::vector<double>   runtimes_;

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
} // namespace nifty::graph::optimization
} // namespace nifty::graph
} // namespace nifty
