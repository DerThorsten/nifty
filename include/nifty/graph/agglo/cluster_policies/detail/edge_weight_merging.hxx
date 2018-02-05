



template<class GRAPH, class T>
class MaxMerging{

private:
    typedef typename GRAPH:: template EdgeMap<double> FloatEdgeMap;

public:
    SettingsType{
    };

    MaxMerging(const SettingsType & settings = SettingsType()){
    }

    void set(const uint64_t edge, const T val){
        values_[edge] = val;
    }
    void get(const uint64_t edge, const T val)const{
        return values_;
    }
    void merge(
        const uint64_t aliveEdge,
        const uint64_t deadEdge
    ){
        auto & a = values_[aliveEdge];
        a = std::max(a, values_[deadEdge]);
    }

private:
    FloatEdgeMap values_;

};



