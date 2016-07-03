#pragma once
#ifndef NIFTY_TOOLS_TIMER_HXX
#define NIFTY_TOOLS_TIMER_HXX

#include <sstream>
#include <chrono>


namespace nifty{
namespace tools{

class Timer
{
public:
	Timer();

	double get_elapsed_seconds() const;

	unsigned long long hours() const;

	unsigned long long minutes() const;
	
	void reset();

	double seconds() const;

	void start();
	
	void stop();
	
	std::string to_string() const;
	
private:
	double m_seconds;

	decltype(std::chrono::high_resolution_clock::now()) m_timeObject;
};




inline Timer::Timer()
{
	reset();
}

inline double Timer::get_elapsed_seconds() const
{
	return m_seconds;
}

inline unsigned long long Timer::hours() const
{
	return static_cast<unsigned long long>(m_seconds) / 3600ULL;
}

inline unsigned long long Timer::minutes() const
{
	return (static_cast<unsigned long long>(m_seconds) - hours()*3600ULL) / 60ULL;
}

inline void Timer::reset()
{
	m_seconds = .0;
}

inline double Timer::seconds() const
{
	return m_seconds - 3600.0*hours() - 60.0*minutes();
}

inline void Timer::start()
{
	m_timeObject = std::chrono::high_resolution_clock::now();
}

inline void Timer::stop()
{
	m_seconds += std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - m_timeObject).count();
}

inline std::string Timer::to_string() const
{
	std::ostringstream s(std::ostringstream::out);
	
	s << hours() << "h " << minutes() << "m " << seconds() << "s";
	
	return s.str();
}


} // end namespace nifty::tools
} // end namespace nifty

#endif /*NIFTY_TOOLS_TIMER_HXX*/
