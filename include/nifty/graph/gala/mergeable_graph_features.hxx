



template<class GRAPH, class T>
MergableGraphFeaturesBase{
public:

    virtual size_t numberOfEdgeFeatures() const  = 0;
    virtual void computeFetures(T * features) = 0;

    void mergeEdges(const size_t )
}
