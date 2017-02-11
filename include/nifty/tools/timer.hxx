#pragma once
#ifndef NIFTY_TOOLS_TIMER_HXX
#define NIFTY_TOOLS_TIMER_HXX

#include <sstream>
#include <chrono>
#include <string>

// heavily inspired by http://www.andres.sc/graph.html


namespace nifty{
namespace tools{

class Timer{
public:
    Timer();

    double elapsedSeconds() const;
    Timer &  reset();
    void start();
    Timer &  stop();
    
    std::string toString() const;
    
private:
    double seconds_;

    decltype(std::chrono::high_resolution_clock::now()) timeObject_;
};


class VerboseTimer : public Timer{
public:

    using Timer::start;

    VerboseTimer(const bool verbose = true, const std::string name = std::string())
    :   Timer(),
        verbose_(verbose),
        name_(name){
    }
    void startAndPrint(const std::string name = std::string()){
        if(!name.empty())
            name_ = name;
        if(verbose_){
            std::cout<<name_<<" started\n";
        }
        this->start();
    }
    VerboseTimer & stopAndPrint(){
        this->stop();
        if(verbose_){
            std::cout<<name_<<" took "<<this->toString()<<"\n";
        }
        return *this;
    }

private:
    bool verbose_;
    std::string name_;
};


inline Timer::Timer(){
    reset();
}

inline double Timer::elapsedSeconds() const{
    return seconds_;
}

inline Timer & Timer::reset(){
    seconds_ = .0;
    return *this;
}

inline void Timer::start(){
    timeObject_ = std::chrono::high_resolution_clock::now();
}

inline Timer &  Timer::stop(){
    typedef std::chrono::duration<double> DDouble;
    seconds_ += std::chrono::duration_cast<DDouble>(std::chrono::high_resolution_clock::now() - timeObject_).count();
    return *this;
}

inline std::string Timer::toString() const{
    const auto h = static_cast<uint64_t>(seconds_) / 3600;
    const auto m = (static_cast<uint64_t>(seconds_) - h*3600) / 60;
    const auto s = seconds_ - 3600.0*h - 60.0*m;
    std::ostringstream ss(std::ostringstream::out);
    ss <<h << "h " << m << "m " << s << "s";
    return ss.str();
}


} // end namespace nifty::tools
} // end namespace nifty

#endif /*NIFTY_TOOLS_TIMER_HXX*/
