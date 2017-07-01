#pragma once

#include <string>

namespace nifty{
namespace logging{


enum class LogLevel{
    NONE = 0,
    FATAL = 1,
    ERROR = 2,
    WARN  = 3,
    INFO  = 4,
    DEBUG = 5,
    TRACE = 6
};

inline std::string logLevelName(const LogLevel logLevel){
    switch(int(logLevel)){
        case 0:
            return std::string("NONE");
        case 1:
            return std::string("FATAL");
        case 2:
            return std::string("ERROR");
        case 3:
            return std::string("WARN");
        case 4:
            return std::string("INFO");
        case 5:
            return std::string("DEBUG");
        case 6:
            return std::string("TRACE");
    }
}


}
}
