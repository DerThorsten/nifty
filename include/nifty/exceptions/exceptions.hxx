#ifndef NIFTY_EXCEPTIONS_EXCEPTIONS_HXX
#define NIFTY_EXCEPTIONS_EXCEPTIONS_HXX

#include <stdexcept>


namespace nifty{
namespace exceptions{


    class WeightsChangedNotSupported
    : public std::runtime_error{
    public:
        WeightsChangedNotSupported(const std::string msg = std::string())
        : std::runtime_error(msg){

        }
    };


    class ResetNotSupported
    : public std::runtime_error{
    public:
        ResetNotSupported(const std::string msg = std::string())
        : std::runtime_error(msg){

        }
    };



}
}

#endif // NIFTY_EXCEPTIONS_EXCEPTIONS_HXX