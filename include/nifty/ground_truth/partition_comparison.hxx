// TAKEN FROM
// https://raw.githubusercontent.com/bjoern-andres/partition-comparison/master/include/andres/partition-comparison.hxx

#pragma once
#define ANDRES_PARTITION_COMPARISON_HXX

#include <map>
#include <utility> // pair
#include <iterator> // iterator_traits
#include <cmath> // log
#include <stdexcept> // runtime_error


namespace nifty {
namespace ground_truth{

template<class T = double>
class RandError {
public:
    typedef T value_type;

    template<class ITERATOR_TRUTH, class ITERATOR_PRED>
    RandError(ITERATOR_TRUTH begin0, ITERATOR_TRUTH end0, ITERATOR_PRED begin1, bool ignoreDefaultLabel = false)
    {
        typedef typename std::iterator_traits<ITERATOR_TRUTH>::value_type Label0;
        typedef typename std::iterator_traits<ITERATOR_PRED>::value_type Label1;
        typedef std::pair<Label0, Label1> Pair;
        typedef std::map<Pair, std::size_t> OverlapMatrix;
        typedef std::map<Label0, std::size_t> TruthSumMap;
        typedef std::map<Label1, std::size_t> PredSumMap;

        OverlapMatrix n;
        TruthSumMap truthSum;
        PredSumMap predSum;

        elements_ = std::distance(begin0, end0);

        if (ignoreDefaultLabel)
        {
            elements_ = 0;

            for(; begin0 != end0; ++begin0, ++begin1)
                if (*begin0 != Label0() && *begin1 != Label1())
                {
                    ++n[Pair(*begin0, *begin1)];
                    ++truthSum[*begin0];
                    ++predSum[*begin1];
                    ++elements_;
                }

            if (elements_ == 0)
                throw std::runtime_error("No element is labeled in both partitions.");
        }
        else
            for(; begin0 != end0; ++begin0, ++begin1)
            {
                ++n[Pair(*begin0, *begin1)];
                ++truthSum[*begin0];
                ++predSum[*begin1];
            }

        for (auto const& it : predSum)
            falseJoins_ += it.second * it.second;

        for (auto const& it : truthSum)
            falseCuts_ += it.second * it.second;

        for (auto const& it : n)
        {
            const std::size_t i = it.first.first;
            const std::size_t j = it.first.second;
            const std::size_t n_ij = it.second;

            trueJoins_ += n_ij * (n_ij - 1) / 2;
            falseCuts_ -= n_ij * n_ij;
            falseJoins_ -= n_ij * n_ij;
        }

        falseJoins_ /= 2;
        falseCuts_ /= 2;

        trueCuts_ = pairs() - joinsInPrediction() - falseCuts_;
    }

    std::size_t elements() const
        { return elements_; }
    std::size_t pairs() const
        { return elements_ * (elements_ - 1) / 2; }

    std::size_t trueJoins() const
        { return trueJoins_; }
    std::size_t trueCuts() const
        { return trueCuts_; }
    std::size_t falseJoins() const
        { return falseJoins_; }
    std::size_t falseCuts() const
        { return falseCuts_; }

    std::size_t joinsInPrediction() const
        { return trueJoins_ + falseJoins_; }
    std::size_t cutsInPrediction() const
        { return trueCuts_ + falseCuts_; }
    std::size_t joinsInTruth() const
        { return trueJoins_ + falseCuts_; }
    std::size_t cutsInTruth() const
        { return trueCuts_ + falseJoins_; }

    value_type recallOfCuts() const
        {
            if(cutsInTruth() == 0)
                return 1;
            else
                return static_cast<value_type>(trueCuts()) / cutsInTruth();
        }
    value_type precisionOfCuts() const
        {
            if(cutsInPrediction() == 0)
                return 1;
            else
                return static_cast<value_type>(trueCuts()) / cutsInPrediction();
        }

    value_type recallOfJoins() const
        {
            if(joinsInTruth() == 0)
                return 1;
            else
                return static_cast<value_type>(trueJoins()) / joinsInTruth();
        }
    value_type precisionOfJoins() const
        {
            if(joinsInPrediction() == 0)
                return 1;
            else
                return static_cast<value_type>(trueJoins()) / joinsInPrediction();
        }

    value_type error() const
        { return static_cast<value_type>(falseJoins() + falseCuts()) / pairs(); }
    value_type index() const
        { return static_cast<value_type>(trueJoins() + trueCuts()) / pairs(); }

private:
    std::size_t elements_;
    std::size_t trueJoins_ { std::size_t() };
    std::size_t trueCuts_ { std::size_t() };
    std::size_t falseJoins_ { std::size_t() };
    std::size_t falseCuts_ { std::size_t() };
};

template<class T = double>
class VariationOfInformation {
public:
    typedef T value_type;

    template<class ITERATOR_TRUTH, class ITERATOR_PRED>
    VariationOfInformation(ITERATOR_TRUTH begin0, ITERATOR_TRUTH end0, ITERATOR_PRED begin1, bool ignoreDefaultLabel = false)
    {
        typedef typename std::iterator_traits<ITERATOR_TRUTH>::value_type Label0;
        typedef typename std::iterator_traits<ITERATOR_PRED>::value_type Label1;
        typedef std::pair<Label0, Label1> Pair;
        typedef std::map<Pair, double> PMatrix;
        typedef std::map<Label0, double> PVector0;
        typedef std::map<Label1, double> PVector1;

        // count
        std::size_t N = std::distance(begin0, end0);

        PMatrix pjk;
        PVector0 pj;
        PVector1 pk;

        if (ignoreDefaultLabel)
        {
            N = 0;

            for (; begin0 != end0; ++begin0, ++begin1)
                if (*begin0 != Label0() && *begin1 != Label1())
                {
                    ++pj[*begin0];
                    ++pk[*begin1];
                    ++pjk[Pair(*begin0, *begin1)];
                    ++N;
                }
        }
        else
            for (; begin0 != end0; ++begin0, ++begin1)
            {
                ++pj[*begin0];
                ++pk[*begin1];
                ++pjk[Pair(*begin0, *begin1)];
            }

        // normalize
        for (auto& p : pj)
            p.second /= N;
        
        for (auto& p : pk)
            p.second /= N;
        
        for (auto& p : pjk)
            p.second /= N;

        // compute information
        auto H0 = value_type();
        for (auto const& p : pj)
            H0 -= p.second * std::log2(p.second);
        
        auto H1 = value_type();
        for (auto const& p : pk)
            H1 -= p.second * std::log2(p.second);
        
        auto I = value_type();
        for (auto const& p : pjk)
        {
            auto j = p.first.first;
            auto k = p.first.second;
            auto pjk_here = p.second;
            auto pj_here = pj[j];
            auto pk_here = pk[k];

            I += pjk_here * std::log2( pjk_here / (pj_here * pk_here) );
        }

        value_ = H0 + H1 - 2.0 * I;
        precision_ = H1 - I;
        recall_ = H0 - I;
    }

    value_type value() const
    {
        return value_;
    }

    value_type valueFalseCut() const
    {
        return precision_;
    }

    value_type valueFalseJoin() const
    {
        return recall_;
    }

private:
    value_type value_;
    value_type precision_;
    value_type recall_;
};

} // namespace ground_truth
} // namespace nifty

