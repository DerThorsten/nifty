#ifndef NIFTY_UFD_ITERABLE_UFD_HXX
#define NIFTY_UFD_ITERABLE_UFD_HXX

#include <vector>
#include <iterator>

#include <boost/iterator/iterator_facade.hpp>

namespace nifty{
namespace ufd{


namespace detail_ufd{

// \cond SUPPRESS_DOXYGEN

// representative element iterator
// for IterableUfd
// only useful for merge graphs internal usage
template<class T>
struct  ConstRepIter
:  public boost::iterator_facade<
ConstRepIter<T>,T,std::forward_iterator_tag
>
{
    typedef IterableUfd<T> IterableUfdType;
    ConstRepIter(const IterableUfdType & p,const T cr)
    :  partition_(&p),
    currentRep_(cr){

    }


    ConstRepIter()
    :  partition_(NULL),
    currentRep_()
    {
    }

private:
    friend class vigra::IteratorFacadeCoreAccess;


    bool isBegin()const{
        return partition_!=NULL  && currentRep_==partition_->firstRep();
    }
    bool isEnd()const{
        return  partition_==NULL || currentRep_>partition_->lastRep();
    }

    bool equal(const ConstRepIter & other)const{
        return   (this->isEnd() && other.isEnd() )  || ((this->isEnd()==other.isEnd() ) && this->currentRep_==other.currentRep_);
    }

    void increment(){
        if(partition_->jumpVec_[currentRep_].second==0){
            currentRep_+=1;
        }
        else{
            currentRep_+=partition_->jumpVec_[currentRep_].second;
        }
    }

    void decrement(){
        if(partition_->jumpVec_[currentRep_].first==0){
            //VIGRA_ASSERT_OP(currentRep_,==,partition_->firstRep());
            //currentRep_+=1;
        }
        else{
            currentRep_-=partition_->jumpVec_[currentRep_].first;
        }
    }

    const T & dereference()const{
        return currentRep_;
    }



    const IterableUfdType * partition_;
    T currentRep_;

};


}

// \endcond

// ufd  data structure structure for merge graph
// only useful for merge graphs internal usage
/// Disjoint set data structure with path compression.
/// \ingroup datastructures
template<class T>
class IterableUfd {
public:
    friend struct ConstRepIter<T>;
    typedef T value_type;
    typedef ConstRepIter<T> const_iterator;

    IterableUfd();
    IterableUfd(const value_type&);

    // query
    value_type find(const value_type&) const; // without path compression
    value_type find(value_type); // with path compression
    value_type numberOfElements() const;
    value_type numberOfSets() const;
    template<class Iterator> void elementLabeling(Iterator) const;
    template<class Iterator> void representatives(Iterator) const;
    void representativeLabeling(std::map<value_type, value_type>&) const;

    // manipulation
    void reset(const value_type&);
    void merge(value_type, value_type);

    value_type firstRep()const{
        return firstRep_;
    }
    value_type lastRep()const{
        return lastRep_;
    }

    const_iterator begin()const{
        if(numberOfSets_!=0)
            return ConstRepIter<T>(*this,firstRep_);
        else
            return ConstRepIter<T>(*this,lastRep_+1);
    }

    const_iterator end()const{
        return ConstRepIter<T>(*this,lastRep_+1);
    }


    const_iterator iteratorAt(const value_type & rep)const{
        if(numberOfSets_!=0)
            return const_iterator(*this,rep);
        else
            return const_iterator(*this,lastRep_+1);
    }

    bool isErased(const value_type & value)const{
        return jumpVec_[value].first == -1 && jumpVec_[value].second == -1;
    }

    void eraseElement(const value_type & value,const bool reduceSize=true){
        const T notRep=value;
        const T jumpMinus = jumpVec_[notRep].first;
        const T jumpPlus  = jumpVec_[notRep].second;

        if(jumpMinus==0){
            const T nextRep = notRep+jumpPlus;
            firstRep_=nextRep;
            jumpVec_[nextRep].first=0;
        }
        else if(jumpPlus==0){
            //VIGRA_ASSERT_OP(lastRep_,==,notRep);
            const T prevRep = notRep-jumpMinus;
            lastRep_=prevRep;
            jumpVec_[prevRep].second=0;
        }
        else{
            const T nextRep = notRep+jumpPlus;
            const T prevRep = notRep-jumpMinus;
            jumpVec_[nextRep].first+=jumpVec_[notRep].first;
            jumpVec_[prevRep].second+=jumpVec_[notRep].second;
        }   
        if(reduceSize){
            --numberOfSets_;
        }
        jumpVec_[notRep].first  =-1;
        jumpVec_[notRep].second =-1;
    }

private:
    std::vector<value_type> parents_;
    std::vector<value_type> ranks_;
    std::vector< std::pair< vigra::Int64, vigra::Int64> > jumpVec_;
    value_type firstRep_;
    value_type lastRep_;
    value_type numberOfElements_;
    value_type numberOfSets_;
};


/// Construct a partition.
template<class T>
IterableUfd<T>::IterableUfd()
: parents_(),
  ranks_(),
  jumpVec_(),
  firstRep_(0),
  lastRep_(0),
  numberOfElements_(0),
  numberOfSets_(0)
{}

/// Construct a partition.
///
/// \param size Number of distinct sets.
///
template<class T>
inline
IterableUfd<T>::IterableUfd
(
   const value_type& size
)
: parents_(static_cast<SizeTType>(size)),
  ranks_(static_cast<SizeTType>(size)),
  jumpVec_(static_cast<SizeTType>(size)),
  firstRep_(0),
  lastRep_(static_cast<SizeTType>(size)-1),
  numberOfElements_(size),
  numberOfSets_(size)
{
    for(T j=0; j<size; ++j) {
        parents_[static_cast<SizeTType>(j)] = j;
    }

    jumpVec_.front().first=0;
    jumpVec_.front().second=1;
    for(T j=1; j<size-1;++j){
        jumpVec_[j].first =1;
        jumpVec_[j].second=1;
    }
    jumpVec_.back().first=1;
    jumpVec_.back().second=0;
}

/// Reset a partition such that each set contains precisely one element
///
/// \param size Number of distinct sets.
///
template<class T>
inline void
IterableUfd<T>::reset
(
   const value_type& size
)
{
    numberOfElements_ = size;
    numberOfSets_ = size;
    ranks_.resize(static_cast<SizeTType>(size));
    parents_.resize(static_cast<SizeTType>(size));
    jumpVec_.resize(static_cast<SizeTType>(size));
    firstRep_=0;
    lastRep_=static_cast<SizeTType>(size)-1;
    for(T j=0; j<size; ++j) {
        ranks_[static_cast<SizeTType>(j)] = 0;
        parents_[static_cast<SizeTType>(j)] = j;
    }

    jumpVec_.front().first=0;
    jumpVec_.front().second=1;
    for(T j=1; j<size-1;++j){
        jumpVec_[j].first =1;
        jumpVec_[j].second=1;
    }
    jumpVec_.back().first=1;
    jumpVec_.back().second=0;
}

/// Find the representative element of the set that contains the given element.
///
/// This constant function does not compress the search path.
///
/// \param element Element.
///
template<class T>
inline typename IterableUfd<T>::value_type
IterableUfd<T>::find
(
   const value_type& element
) const
{
    // find the root
    value_type root = element;
    while(parents_[static_cast<SizeTType>(root)] != root) {
        root = parents_[static_cast<SizeTType>(root)];
    }
    return root;
}

/// Find the representative element of the set that contains the given element.
///
/// This mutable function compresses the search path.
///
/// \param element Element.
///
template<class T>
inline typename IterableUfd<T>::value_type
IterableUfd<T>::find
(
   value_type element // copy to work with
)
{
    // find the root
    value_type root = element;
    while(parents_[static_cast<SizeTType>(root)] != root) {
        root = parents_[static_cast<SizeTType>(root)];
    }
    // path compression
    while(element != root) {
        value_type tmp = parents_[static_cast<SizeTType>(element)];
        parents_[static_cast<SizeTType>(element)] = root;
        element = tmp;
    }
    return root;
}

/// Merge two sets.
///
/// \param element1 Element in the first set.
/// \param element2 Element in the second set.
///
template<class T>
inline void
IterableUfd<T>::merge
(
   value_type element1,
   value_type element2
)
{
    // merge by rank
    element1 = find(element1);
    element2 = find(element2);
    if(element1!=element2){
        T notRep;
        if(ranks_[static_cast<SizeTType>(element1)] < ranks_[static_cast<SizeTType>(element2)]) {
            parents_[static_cast<SizeTType>(element1)] = element2;
            --numberOfSets_;
            //rep=element2;
            notRep=element1;
        }
        else if(ranks_[static_cast<SizeTType>(element1)] > ranks_[static_cast<SizeTType>(element2)]) {
            parents_[static_cast<SizeTType>(element2)] = element1;
            --numberOfSets_;
            //rep=element1;
            notRep=element2;
        }
        else if(element1 != element2) {
            parents_[static_cast<SizeTType>(element2)] = element1;
            ++ranks_[static_cast<SizeTType>(element1)];
            --numberOfSets_;
            //rep=element1;
            notRep=element2;
        }
        this->eraseElement(notRep,false);
    }
}  

template<class T>
inline typename IterableUfd<T>::value_type
IterableUfd<T>::numberOfElements() const
{
    return numberOfElements_;
}

template<class T>
inline typename IterableUfd<T>::value_type
IterableUfd<T>::numberOfSets() const
{
    return numberOfSets_;
}


} // end namespace nifty::ufd
} // end namespace nifty

#endif /*NIFTY_UFD_ITERABLE_UFD_HXX*/
