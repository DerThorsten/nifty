#pragma once

#include <boost/pending/disjoint_sets.hpp>
#include <xtensor/xtensor.hpp>


namespace nifty {
namespace ufd {

    template<class T = uint64_t>
    class BoostUfd {

    public:
        typedef T value_type;

        // initialize from number of elements
        // for consecutive elements
        BoostUfd(const value_type size) : n_elements_(size),
                                          ranks_(n_elements_),
                                          parents_(n_elements_),
                                          sets_(&ranks_[0], &parents_[0]) {
            for(value_type elem = 0; elem < n_elements_; ++elem) {
                sets_.make_set(elem);
            }
        }


        // initialize from element list for non-consecutive elements
        template<class ELEMENTS>
        BoostUfd(const xt::xexpression<ELEMENTS> & elements) : n_elements_(elements.derived_cast().size()),
                                                               ranks_(n_elements_),
                                                               parents_(n_elements_),
                                                               sets_(&ranks_[0], &parents_[0]) {
            const auto & elems = elements.derived_cast();
            for(const value_type elem : elems) {
                sets_.make_set(elem);
            }
        }

        // TODO make const and not const version for path compression ?
        inline value_type find(const value_type elem) {
            return sets_.find_set(elem);
        }


        inline void merge(const value_type elem1, const value_type elem2) {
            // std::cout << "Merge elems " << elem1 << " " << elem2 << std::endl;
            sets_.link(elem1, elem2);
        }


        inline size_t numberOfElements() const {
            return n_elements_;
        }

    private:
        size_t n_elements_;
        std::vector<value_type> ranks_;
        std::vector<value_type> parents_;
        boost::disjoint_sets<value_type*, value_type*> sets_;
    };

}
}
