#pragma once

namespace nifty{
namespace tools{


template<class ITERATOR>
class ConstIteratorRange{
public:
    typedef ITERATOR const_iterator;
    ConstIteratorRange(const_iterator b, const_iterator e)
    :   begin_(b),
        end_(e){

    }
    const_iterator begin()const{
        return begin_;
    }
    const_iterator begin(){
        return begin_;
    }

    const_iterator end()const{
        return end_;
    }
    const_iterator end(){
        return end_;
    }

private:
    ITERATOR begin_,end_;
};


} // end namespace nifty::tools
} // end namespace nifty

