#pragma once

#include <boost/pending/disjoint_sets.hpp>
#include "nifty/graph/edge_contraction_graph.hxx"
#include "nifty/graph/opt/multicut/multicut_base.hxx"
#include "nifty/tools/changable_priority_queue.hxx"

namespace nifty{
namespace graph{
namespace opt{
namespace multicut{

namespace detail_multicut_greedy_fixation{

    template<class WEIGHT_TYPE>
    class DynamicGraph
    {
    public:
        DynamicGraph(size_t n) :
            n_nodes_(n), vertices_(n), cut_edges_(n)
        {}

        inline bool edgeExists(size_t a, size_t b) const
        {
            return !vertices_[a].empty() && vertices_[a].find(b) != vertices_[a].end();
        }

        inline std::unordered_map<size_t, WEIGHT_TYPE> const& getAdjacentVertices(size_t v) const
        {
            return vertices_[v];
        }

        inline WEIGHT_TYPE getEdgeWeight(size_t a, size_t b) const
        {
            return vertices_[a].at(b);
        }

        inline bool haveConstraint(size_t a, size_t b) const
        {
            return !cut_edges_[a].empty() && cut_edges_[a].find(b) != cut_edges_[a].end();
        }

        inline void addConstraint(size_t a, size_t b)
        {
            cut_edges_[a].insert(b);
            cut_edges_[b].insert(a);
        }

        inline void removeVertex(size_t v)
        {
            for (auto& p : vertices_[v])
            {
                vertices_[p.first].erase(v);
                cut_edges_[p.first].erase(v);
            }

            vertices_[v].clear();
            cut_edges_[v].clear();
            --n_nodes_;
        }

        inline void setEdgeWeight(size_t a, size_t b, WEIGHT_TYPE w)
        {
            vertices_[a][b] = w;
            vertices_[b][a] = w;
        }

        inline std::size_t numberOfNodes() {
            return n_nodes_;
        }

    private:
        std::size_t n_nodes_;
        std::vector<std::unordered_set<size_t>> cut_edges_;
        std::vector<std::unordered_map<size_t, WEIGHT_TYPE>> vertices_;
    };


    template<class WEIGHT_TYPE>
    struct Edge
    {
        Edge(size_t _a, size_t _b, WEIGHT_TYPE _w)
        {
            if (_a > _b)
                std::swap(_a, _b);

            a = _a;
            b = _b;

            w = _w;
        }

        size_t a;
        size_t b;
        size_t edition;
        WEIGHT_TYPE w;

        inline bool operator <(Edge const& other) const
        {
            return std::abs(w) < std::abs(other.w);
        }
    };

} // end namespace detail_multicut_greedy_fixation

    // greedy fixation implementation based on
    // https://github.com/bjoern-andres/graph/blob/master/include/andres/graph/multicut/greedy-fixation.hxx
    // I tried to implement it based on edgeContraction graph, but didn't manage to get a working version
    // too many abstractions and std::priority_queue + boost ufd are doing a totally fine job here
    template<class OBJECTIVE>
    class MulticutGreedyFixation : public MulticutBase<OBJECTIVE>
    {
        public:

        struct SettingsType{
            double weightStopCond{0.0};
            double nodeNumStopCond{-1};
            int visitNth{1};
        };

        typedef OBJECTIVE ObjectiveType;
        typedef typename ObjectiveType::GraphType GraphType;
        typedef MulticutBase<OBJECTIVE> BaseType;
        typedef typename BaseType::VisitorBaseType VisitorBaseType;
        typedef typename BaseType::NodeLabelsType NodeLabelsType;
        typedef typename ObjectiveType::WeightType WeightType;
        typedef detail_multicut_greedy_fixation::DynamicGraph<WeightType> DynamicGraphType;
        typedef detail_multicut_greedy_fixation::Edge<WeightType> EdgeType;

        virtual ~MulticutGreedyFixation(){}
        MulticutGreedyFixation(const ObjectiveType & objective, const SettingsType & settings=SettingsType());

        virtual void optimize(NodeLabelsType & nodeLabels, VisitorBaseType * visitor);
        virtual const ObjectiveType & objective() const;

        void reset();
        void changeSettings(const SettingsType & settings);

        virtual void weightsChanged(){
            this->reset();
        }

        virtual const NodeLabelsType & currentBestNodeLabels( ){
            for(auto node : graph_.nodes()){
                currentBest_->operator[](node) = partition_.find_set(node);
            }
            return *currentBest_;
        }

        virtual std::string name() const {
            return std::string("MulticutGreedyFixation");
        }

        private:

        inline bool done(const WeightType weight) {
            const auto nnsc = settings_.nodeNumStopCond;
            // exit if weight stop cond kicks in
            if(std::abs(weight) <= settings_.weightStopCond){
                return true;
            }
            if(nnsc > 0.0){
                uint64_t ns;
                if(nnsc >= 1.0){
                    ns = static_cast<uint64_t>(nnsc);
                }
                else{
                    ns = static_cast<uint64_t>(double(graph_.numberOfNodes())*nnsc +0.5);
                }
                if(dynamicGraph_.numberOfNodes() <= ns){
                    return true;
                }
            }
            if(dynamicGraph_.numberOfNodes() <= 1){
                return true;
            }
            if(pq_.empty()){
                return true;
            }
            return false;
        }

        inline void mergeNodes(std::size_t a, std::size_t b) {
            partition_.link(a, b);

            auto stable_vertex = a;
            auto merge_vertex = b;
            if(partition_.find_set(stable_vertex) != stable_vertex) {
                std::swap(stable_vertex, merge_vertex);
            }

            for (auto& p : dynamicGraph_.getAdjacentVertices(merge_vertex))
            {
                if (p.first == stable_vertex)
                    continue;

                WeightType t = 0;

                if (dynamicGraph_.edgeExists(stable_vertex, p.first))
                    t = dynamicGraph_.getEdgeWeight(stable_vertex, p.first);

                if (dynamicGraph_.haveConstraint(merge_vertex, p.first))
                    dynamicGraph_.addConstraint(stable_vertex, p.first);

                dynamicGraph_.setEdgeWeight(stable_vertex, p.first, p.second + t);

                auto e = EdgeType(stable_vertex, p.first, p.second + t);
                e.edition = ++edge_editions_[e.a][e.b];

                pq_.push(e);
            }

            dynamicGraph_.removeVertex(merge_vertex);
        }

        SettingsType settings_;
        const ObjectiveType & objective_;
        const GraphType & graph_;
        NodeLabelsType * currentBest_;
        DynamicGraphType dynamicGraph_;
        std::priority_queue<EdgeType> pq_;

        // union find
        std::vector<size_t> ranks_;
        std::vector<size_t> parents_;
        boost::disjoint_sets<std::size_t*, std::size_t*> partition_;

        std::vector<std::unordered_map<size_t, size_t>> edge_editions_;
    };

    template<class OBJECTIVE>
    MulticutGreedyFixation<OBJECTIVE>::
    MulticutGreedyFixation(
        const ObjectiveType & objective,
        const SettingsType & settings
    )
    :   settings_(settings),
        objective_(objective),
        graph_(objective.graph()),
        currentBest_(nullptr),
        dynamicGraph_(objective.graph().numberOfNodes()),
        ranks_(objective.graph().numberOfNodes()),
        parents_(objective.graph().numberOfNodes()),
        partition_(&ranks_[0], &parents_[0]),
        edge_editions_(objective.graph().numberOfNodes())
    {
        this->reset();
    }

    template<class OBJECTIVE>
    void MulticutGreedyFixation<OBJECTIVE>::
    optimize(
        NodeLabelsType & nodeLabels,  VisitorBaseType * visitor
    ){

        if(graph_.numberOfEdges() == 0) {
            return;
        }

        if(visitor!=nullptr){
            visitor->addLogNames({"#nodes","topWeight"});
            visitor->begin(this);
        }

        currentBest_ = &nodeLabels;
        std::size_t c = 1;

        while(true)
        {
            auto edge = pq_.top();
            pq_.pop();

            if(done(edge.w)) {
                break;
            }

            // skip edges that are not part of the graph any more
            if (!dynamicGraph_.edgeExists(edge.a, edge.b) || edge.edition < edge_editions_[edge.a][edge.b]) {
                continue;
            }

            const bool haveConstraint = dynamicGraph_.haveConstraint(edge.a, edge.b);
            // std::cout << "Edge " << edge.a << "<->" << edge.b << " weight: " << edge.w << " constraint: " << haveConstraint << std::endl;

            if(!haveConstraint) {  // only do stuff if we don't have a cannott link constraint

                if (edge.w > 0)  // attractiv edge -> merge nodes
                {
                    mergeNodes(edge.a, edge.b);
                }
                else if (edge.w < 0) {  // repulsive edge -> add cannot link constraint
                    dynamicGraph_.addConstraint(edge.a, edge.b);
                }
            }

            if(c % settings_.visitNth == 0){

                if(visitor!=nullptr){
                   visitor->setLogValue(0, dynamicGraph_.numberOfNodes());
                   visitor->setLogValue(1, edge.w);
                   if(!visitor->visit(this)){
                        std::cout<<"end by visitor\n";
                       break;
                   }
                }
            }
            ++c;
        }

        for(auto node : graph_.nodes()){
            nodeLabels[node] = partition_.find_set(node);
        }

        if(visitor!=nullptr) {
            visitor->end(this);
        }
    }

    template<class OBJECTIVE>
    const typename MulticutGreedyFixation<OBJECTIVE>::ObjectiveType &
    MulticutGreedyFixation<OBJECTIVE>::
    objective()const{
        return objective_;
    }


    template<class OBJECTIVE>
    void MulticutGreedyFixation<OBJECTIVE>::
    reset(
    ){
        // initialise ufd
        for(auto node : graph_.nodes()) {
            partition_.make_set(node);
        }

        const auto & edge_values = objective_.weights();

        // fill pq
        for (size_t i = 0; i < graph_.numberOfEdges(); ++i)
        {
            auto uv = graph_.uv(i);
            auto u = uv.first;
            auto v = uv.second;

            dynamicGraph_.setEdgeWeight(u, v, edge_values[i]);

            auto e = EdgeType(u, v, edge_values[i]);
            e.edition = ++edge_editions_[u][v];

            pq_.push(e);
        }
    }

    template<class OBJECTIVE>
    inline void
    MulticutGreedyFixation<OBJECTIVE>::
    changeSettings(
        const SettingsType & settings
    ){
        settings_ = settings;
    }


} // namespace nifty::graph::opt::multicut
} // namespace nifty::graph::opt
} // namespace nifty::graph
} // namespace nifty
