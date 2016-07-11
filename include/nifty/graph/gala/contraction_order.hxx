



template< class CALLBACK>
class McGreedyHybridBase{
private:
    typedef CALLBACK CallbackType;
    typedef typename CallbackType::ValueType                          ValueType
    typedef typename CallbackType::GraphType                          GraphType
    typedef typename CallbackType::EdgeContractionGraphType           CGraphType
    typedef typename GraphType:: template EdgeMap<ValueType>          EdgeMapDouble;
    typedef vigra::ChangeablePriorityQueue< T ,std::less<ValueType> > QueueType;

public: 
    struct Setttings{
        std::array<T, 4> weights_{ValueType(1),ValueType(1),ValueType(1),ValueType(1)};
        T stopWeight{0.5};
    };

    McGreedyHybridBase(
        const GRAPH & graph, 
        const CGraphType & cgraph,
        const bool training,
        const Setttings & settings = Setttings()
    )
    :   graph_(graph),
        cgraph_(cgraph),
        training_(training),
        settings_(settings),
        pq_(graph.maxEdgeId()+1),
        localRfProbs_(graph),
        mcPerturbAndMapProbs_(graph),
        mcMapProbs_(graph),
        wardProbs_(graph),
        constraints_(graph,0),
        hasMcPerturbAndMapProbs_(false),
        hasMcMapProbs_(false)
    {

    }

    uint64_t edgeToContractNext(){
        
    }

    bool stopContraction(){
        if(!training_){
            return pq_.topPriority() >= settings_.stopWeight;
        }
        else{
            return pq_.topPriority() >= 1.5;
        }
    }

    void reset(){
        while(!pq_.empty()){
            pq_.pop();
        }
        hasMcPerturbAndMapProbs_ = false;
        hasMcMapProbs_ = false;
    }

private:
    T makeTotalWeight(const uint64_t edge)const{

        if(constraints_[edge] > 0.000001){
            return std::numeric_limits<ValueType>::infinity();
        }
        else{
            ValueType wSum = 0.0;
            ValueType pAcc = 0.0;
            const auto & w = weights_;

            wSum += w[0];
            pAcc += w[0]*localRfProbs_[edge];
            
            if(hasMcPerturbAndMapProbs_){
                wSum += w[1];
                pAcc += w[1]*mcPerturbAndMapProbs_[edge];
            }

            if(hasMcMapProbs_){
                wSum += w[2];
                pAcc += w[2]*mcMapProbs_[edge];
            }

            wSum += w[3];
            pAcc += w[3]*wardProbs_[edge];


            return pAcc/wSum;
        }
    }

    const GraphType & graph_;
    const CGraphType & cgraph_;

    Setttings settings_;
    QueueType pq_;
    EdgeMapDouble localRfProbs_;
    EdgeMapDouble mcPerturbAndMapProbs_;
    EdgeMapDouble mcMapProbs_;
    EdgeMapDouble wardProbs_;
    EdgeMapDouble constraints_;
    bool hasMcPerturbAndMapProbs_;
    bool hasMcMapProbs_;

};




struct McGreedy{
    template<class CB>
    struct Training{
        typedef McGreedyHybridBase<CB> ResultType;
    };

    template<class CB>
    struct Test{
        typedef McGreedyHybridBase<CB> ResultType;
    };
};
