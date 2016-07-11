#pragma once
#ifndef NIFTY_GRAPH_GALA_CLASSIFIER_RF_HXX
#define NIFTY_GRAPH_GALA_CLASSIFIER_RF_HXX

#include <iostream>

#include "vigra/multi_array.hxx"
#include "vigra/random_forest.hxx"
#include "vigra/random_forest_3.hxx"
#include "nifty/tools/timer.hxx"

namespace nifty{
namespace graph{

    
    template<class T>
    class RfClassifier{
    public:
        RfClassifier(const size_t numberOfTrees=255)
        :   classifier_(nullptr),
            numberOfTrees_(numberOfTrees){

        }
        void initialize(const size_t numberOfFeatures){
            numberOfFeatures_ = numberOfFeatures;
            classifier_ = nullptr;
        }
        void addTrainingExample(const T * features, const uint8_t label){
            newRfLabels_.push_back(label);
            for(size_t fi=0; fi<numberOfFeatures_; ++fi){
                newRfFeatruresFlat_.push_back(features[fi]);
            }
        }
        void train(){
            tools::VerboseTimer t(false);
            t.startAndPrint("       makeTZrainingSet");
            makeTrainingSet();
            t.stopAndPrint();
            t.reset();

             t.startAndPrint("      do training");
            if(classifier_ != nullptr){
                delete classifier_;
            }
            auto rfOpts = vigra::RandomForestOptions();  
            rfOpts.tree_count(numberOfTrees_);
            rfOpts.predict_weighted();
            classifier_ = new  vigra::RandomForest<uint8_t>(rfOpts);

            // construct visitor to calculate out-of-bag error
            vigra::rf::visitors::OOB_Error oob_v;
            // perform training
            classifier_->learn(rfFeatures_, rfLabels_, vigra::rf::visitors::create_visitor(oob_v));
            t.stopAndPrint();
            t.reset();
        }

        double predictProbability(const T * features)const{
            // copy stuff atm 
            vigra::MultiArray<2, T>       f(vigra::Shape2(1,numberOfFeatures_));     
            vigra::MultiArray<2, double>  p(vigra::Shape2(1,2));
            for(size_t i=0; i<numberOfFeatures_; ++i){
                f[i] = features[i];
            }
            classifier_->predictProbabilities(f,p);
            return p(0,1);
        }
    private:
        void makeTrainingSet(){

            const auto nOld = rfFeatures_.shape(0);
            const auto nAdded = newRfLabels_.size();
            const auto nTotal = nOld + nAdded;

            //std::cout<<"new added examples "<<nAdded<<" total "<<nTotal<<"\n";
            vigra::MultiArray<2, T> fNew(vigra::Shape2( nTotal ,numberOfFeatures_));
            vigra::MultiArray<2, uint8_t> lNew(vigra::Shape2( nTotal ,1));

            for(auto i=0; i<nOld; ++i){
                lNew[i] = rfLabels_[i];
                for(auto fi=0; fi<numberOfFeatures_; ++fi)
                    fNew(i, fi) = rfFeatures_(i,fi);
            }
            auto c = 0;
            for(auto i=nOld; i<nTotal; ++i){
                lNew[i] = newRfLabels_[i-nOld];
                for(auto fi=0; fi<numberOfFeatures_; ++fi){
                    fNew(i, fi) = newRfFeatruresFlat_[c];
                    ++c;
                }
            }
            rfFeatures_ = fNew;
            rfLabels_ = lNew;
            newRfLabels_.resize(0);
            newRfFeatruresFlat_.resize(0);
        }


        vigra::RandomForest<uint8_t> *  classifier_;
        const size_t                    numberOfTrees_;
        vigra::MultiArray<2, T>         rfFeatures_;     
        vigra::MultiArray<2, uint8_t>   rfLabels_;

        std::vector< T >        newRfFeatruresFlat_;
        std::vector<uint8_t>    newRfLabels_;
        uint64_t numberOfFeatures_;
    };
    

    /*
    template<class T>
    class RfClassifier{
    public:

        typedef vigra::MultiArray<2, double>            Features;
        typedef vigra::MultiArray<1, uint8_t>           Labels;
        typedef typename vigra::rf3::DefaultRF<Features, Labels>::type RfImpl;



        RfClassifier(const size_t numberOfTrees=255)
        :   classifier_(),
            numberOfTrees_(numberOfTrees){

        }
        void initialize(const size_t numberOfFeatures){
            numberOfFeatures_ = numberOfFeatures;
        }
        void addTrainingExample(const T * features, const uint8_t label){
            newRfLabels_.push_back(label);
            for(size_t fi=0; fi<numberOfFeatures_; ++fi){
                newRfFeatruresFlat_.push_back(features[fi]);
            }
        }
        void train(){
            tools::VerboseTimer t(false);
            t.startAndPrint("       makeTZrainingSet");
            makeTrainingSet();
            t.stopAndPrint();
            t.reset();

            t.startAndPrint("      do training");


            vigra::rf3::RandomForestOptions const options = vigra::rf3::RandomForestOptions()
                                                   .tree_count(numberOfTrees_)
                                                   .bootstrap_sampling(true)
                                                   .n_threads(-1);
            vigra::rf3::OOBError oob;
            classifier_ = random_forest(rfFeatures_, rfLabels_, options, create_visitor(oob));


            t.stopAndPrint();
            t.reset();
        }

        double predictProbability(const T * features)const{
            // copy stuff atm 
            vigra::MultiArray<2, T>       f(vigra::Shape2(1,numberOfFeatures_));     
            vigra::MultiArray<2, double>  p(vigra::Shape2(1,2));
            for(size_t i=0; i<numberOfFeatures_; ++i){
                f[i] = features[i];
            }
            classifier_.predict_proba(f,p);
            return p(0,1);
        }
    private:
        void makeTrainingSet(){

            const auto nOld = rfFeatures_.shape(0);
            const auto nAdded = newRfLabels_.size();
            const auto nTotal = nOld + nAdded;

            //std::cout<<"new added examples "<<nAdded<<" total "<<nTotal<<"\n";
            Features fNew(vigra::Shape2( nTotal ,numberOfFeatures_));
            Labels lNew = Labels(vigra::Shape1( nTotal));

            for(auto i=0; i<nOld; ++i){
                lNew[i] = rfLabels_[i];
                for(auto fi=0; fi<numberOfFeatures_; ++fi)
                    fNew(i, fi) = rfFeatures_(i,fi);
            }
            auto c = 0;
            for(auto i=nOld; i<nTotal; ++i){
                lNew[i] = newRfLabels_[i-nOld];
                for(auto fi=0; fi<numberOfFeatures_; ++fi){
                    fNew(i, fi) = newRfFeatruresFlat_[c];
                    ++c;
                }
            }
            rfFeatures_ = fNew;
            rfLabels_ = lNew;
            newRfLabels_.resize(0);
            newRfFeatruresFlat_.resize(0);
        }


        RfImpl classifier_;
        const size_t                    numberOfTrees_;
        vigra::MultiArray<2, T>         rfFeatures_;     
        vigra::MultiArray<1, uint8_t>   rfLabels_;

        std::vector< T >        newRfFeatruresFlat_;
        std::vector<uint8_t>    newRfLabels_;
        uint64_t numberOfFeatures_;
    };
    */

} // namespace nifty::graph
} // namespace nifty

#endif  // NIFTY_GRAPH_GALA_CLASSIFIER_RF_HXX
