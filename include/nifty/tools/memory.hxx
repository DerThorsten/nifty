#pragma once

#include <memory>

namespace nifty {
namespace tools {

#if __cplusplus < 201402L
    template<typename T, typename... Args>
    std::unique_ptr<T> make_unique(Args&&... args)
    {
            return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
    }
#else
    using namespace std;
#endif

}
}
