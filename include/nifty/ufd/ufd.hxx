#pragma once
#ifndef ANDRES_PARTITION_HXX
#define ANDRES_PARTITION_HXX

#include <cstddef>
#include <vector>
#include <map>


namespace nifty {
namespace ufd{

/// Disjoint set data structure with path compression.
template<class T = uint64_t>
class Ufd {
public:
    typedef T Index;

    Ufd(const Index = 0);
    void assign(const Index = 0);
    void reset(){
        if(numberOfSets_ < parents_.size()){
            numberOfSets_ = parents_.size();
            for(Index j = 0; j < numberOfSets_; ++j) {
                parents_[static_cast<uint64_t>(j)] = j;
            }
        }  
    }
    Index find(const Index) const; // without path compression
    Index find(Index); // with path compression
    Index numberOfElements() const;
    Index numberOfSets() const;
    template<class Iterator>
        void elementLabeling(Iterator) const;
    template<class Iterator>
        void representatives(Iterator) const;

    template<class MAP_LIKE>
    void representativeLabeling(MAP_LIKE &) const;

    void merge(Index, Index);
    void insert(const Index);

private:
    std::vector<Index> parents_;
    std::vector<Index> ranks_;
    Index numberOfSets_;
};

/// Construct a ufd (with a number of sets each containing one element).
///
/// \param size Number of distinct sets. 
///
template<class T>
inline 
Ufd<T>::Ufd(
    const Index size
)
:   parents_(static_cast<uint64_t>(size)),
    ranks_(static_cast<uint64_t>(size)),
    numberOfSets_(size)
{
    for(Index j = 0; j < size; ++j) {
        parents_[static_cast<uint64_t>(j)] = j;
    }
}

/// Reset the ufd (to a number of sets each containing one element).
///
/// \param size Number of distinct sets.
///
template<class T>
inline void
Ufd<T>::assign(
    const Index size
) {
    parents_.resize(static_cast<uint64_t>(size));
    ranks_.resize(static_cast<uint64_t>(size));
    numberOfSets_ = size;
    for(Index j = 0; j < size; ++j) {
        parents_[static_cast<uint64_t>(j)] = j;
    }
}

template<class T>
inline typename Ufd<T>::Index 
Ufd<T>::numberOfElements() const {
    return static_cast<Index>(parents_.size());
}

template<class T>
inline typename Ufd<T>::Index 
Ufd<T>::numberOfSets() const {
    return numberOfSets_; 
}

/// Find the representative element of the set that contains the given element (without path compression).
/// 
/// \param element Element. 
///
template<class T>
inline typename Ufd<T>::Index
Ufd<T>::find(
    const Index element
) const {
    // find the root
    Index root = element;
    while(parents_[static_cast<uint64_t>(root)] != root) {
        root = parents_[static_cast<uint64_t>(root)];
    }
    return root;
}

/// Find the representative element of the set that contains the given element (with path compression).
/// 
/// This mutable function compresses the search path.
///
/// \param element Element. 
///
template<class T>
inline typename Ufd<T>::Index
Ufd<T>::find(
    Index element // copy to work with
) {
    // find the root
    Index root = element;
    while(parents_[static_cast<uint64_t>(root)] != root) {
        root = parents_[static_cast<uint64_t>(root)];
    }
    // path compression
    while(element != root) {
        const Index tmp = parents_[static_cast<uint64_t>(element)];
        parents_[static_cast<uint64_t>(element)] = root;
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
Ufd<T>::merge(
    Index element1,
    Index element2
) {
    // merge by rank
    element1 = find(element1);
    element2 = find(element2);
    if(ranks_[static_cast<uint64_t>(element1)] < ranks_[static_cast<uint64_t>(element2)]) {
        parents_[static_cast<uint64_t>(element1)] = element2;
        --numberOfSets_;
    }
    else if(ranks_[static_cast<uint64_t>(element1)] > ranks_[static_cast<uint64_t>(element2)]) {
        parents_[static_cast<uint64_t>(element2)] = element1;
        --numberOfSets_;
    }
    else if(element1 != element2) {
        parents_[static_cast<uint64_t>(element2)] = element1;
        ++ranks_[static_cast<uint64_t>(element1)];
        --numberOfSets_;
    }
}

/// Insert a number of new sets, each containing one element.
/// 
/// \param number Number of sets to insert. 
///
template<class T>
inline void 
Ufd<T>::insert(
    const Index number
) {
    const Index numberOfElements = static_cast<Index>(parents_.size());
    ranks_.insert(ranks_.end(), static_cast<uint64_t>(number), 0);
    parents_.insert(parents_.end(), static_cast<uint64_t>(number), 0);
    for(Index j = numberOfElements; j < numberOfElements + number; ++j) {
        parents_[static_cast<uint64_t>(j)] = j;
    }
    numberOfSets_ += number;
}

/// Output all elements which are set representatives.
/// 
/// \param it (Output) Iterator.
///
template<class T>
template<class Iterator>
inline void 
Ufd<T>::representatives(
    Iterator it
) const {
    for(Index j = 0; j < numberOfElements(); ++j) {
        if(parents_[static_cast<uint64_t>(j)] == j) {
            *it = j;
            ++it;
        }
    }
}

/// Output a contiguous labeling of the representative elements.
/// 
/// \param out (Output) A map that assigns each representative element to an integer label.
///
template<class T>
template<class MAP_LIKE>
inline void 
Ufd<T>::representativeLabeling(
    MAP_LIKE & out
) const {
    out.clear();	
    std::vector<Index> r(static_cast<uint64_t>(numberOfSets()));
    representatives(r.begin());
    for(Index j = 0; j < numberOfSets(); ++j) {
        out[r[static_cast<uint64_t>(j)]] = j;
    }
}

/// Output a contiguous labeling of all elements.
/// 
/// \param out (Output) Iterator into a container in which the j-th entry becomes the label of the j-th element.
///
template<class T>
template<class Iterator>
inline void 
Ufd<T>::elementLabeling(
    Iterator out
) const {
    std::map<Index, Index> rl;
    representativeLabeling(rl);
    for(Index j = 0; j < numberOfElements(); ++j) {
        *out = rl[find(j)];
        ++out;
    }
}

} // namespace ufd
} // namespace nifty

#endif // #ifndef ANDRES_PARTITION_HXX
