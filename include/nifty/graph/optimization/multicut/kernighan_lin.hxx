#pragma once

#include <set>
#include <stack>
#include <vector>
#include <iomanip>

#include "nifty/tools/runtime_check.hxx"
#include "nifty/container/boost_flat_set.hxx"

#include "nifty/graph/optimization/multicut/multicut_base.hxx"
#include "nifty/graph/optimization/multicut/multicut_objective.hxx"






namespace nifty{
namespace graph{
namespace optimization{
namespace multicut{
   


    template<class OBJECTIVE>
    class KernighanLin : public MulticutBase<OBJECTIVE>
    {
    public: 

        typedef OBJECTIVE Objective;
        typedef OBJECTIVE ObjectiveType;
        typedef typename ObjectiveType::WeightType WeightType;
        typedef MulticutBase<ObjectiveType> BaseType;
        typedef typename BaseType::VisitorBaseType VisitorBaseType;
        typedef typename BaseType::VisitorProxyType VisitorProxyType;
        typedef typename BaseType::NodeLabelsType NodeLabelsType;
        typedef typename ObjectiveType::Graph Graph;
        typedef typename ObjectiveType::GraphType GraphType;
        typedef typename ObjectiveType::WeightsMap WeightsMap;
        typedef typename GraphType:: template EdgeMap<uint8_t> IsDirtyEdge;






    
    public:

        struct SettingsType{
            uint64_t numberOfInnerIterations { std::numeric_limits<uint64_t>::max() };
            uint64_t numberOfOuterIterations { 100 };
            double epsilon { 1e-6 };
            bool verbose { false };
        };

        virtual ~KernighanLin(){
            
        }
        KernighanLin(const Objective & objective, const SettingsType & settings = SettingsType());


        virtual void optimize(NodeLabelsType & nodeLabels, VisitorBaseType * visitor);
        virtual const Objective & objective() const;


        virtual const NodeLabelsType & currentBestNodeLabels( ){
            return *currentBest_;
        }

        virtual std::string name()const{
            return std::string("KernighanLin");
        }
        virtual void weightsChanged(){ 
        }
        //virtual double currentBestEnergy() {
        //   return currentBestEnergy_;
        //}
    private:


        struct TwoCutBuffers{
            TwoCutBuffers(const GraphType & graph) :
                differences(graph),
                is_moved(graph),
                referenced_by(graph),
                vertex_labels(graph)
            {

            }

            typename GraphType:: template NodeMap<uint64_t> border;
            typename GraphType:: template NodeMap<double> differences;
            typename GraphType:: template NodeMap<char>   is_moved;
            uint64_t max_not_used_label;
            typename GraphType:: template NodeMap<uint64_t> referenced_by;
            NodeLabelsType vertex_labels;    
        };


        template<class NODE_LABELS>
        uint64_t maxLabel(const NODE_LABELS & nodeLabels){
            uint64_t mx = 0;
            for(const auto node : graph_.nodes()){
                mx = std::max(mx, nodeLabels[node]);
            }
            return mx;
        }

        double update_bipartition(std::vector<uint64_t>& A, std::vector<uint64_t>& B);

        const Objective & objective_;
        const GraphType & graph_;
        SettingsType settings_;
        NodeLabelsType * currentBest_;
        double currentBestEnergy_;

        TwoCutBuffers buffer_;
    
    };

    
    template<class OBJECTIVE>
    KernighanLin<OBJECTIVE>::
    KernighanLin(
        const Objective & objective, 
        const SettingsType & settings
    )
    :   objective_(objective),
        graph_(objective.graph()),
        settings_(settings),
        currentBest_(nullptr),
        currentBestEnergy_(std::numeric_limits<double>::infinity()),
        buffer_(objective.graph())
    {

    }

    template<class OBJECTIVE>
    void KernighanLin<OBJECTIVE>::
    optimize(
        NodeLabelsType & nodeLabels,  VisitorBaseType * visitor
    ){  

        
        VisitorProxyType visitorProxy(visitor);
     
        currentBest_ = &nodeLabels;
        currentBestEnergy_ = objective_.evalNodeLabels(nodeLabels);

        visitorProxy.begin(this);


        // its actually an upper bound
        // but lets keep the ``andres`` 
        // variable names atm
        auto numberOfComponents = this->maxLabel(nodeLabels) + 1; 

        // build partitions
        visitorProxy.printLog(nifty::logging::LogLevel::DEBUG, "build partitions");

        std::vector<std::vector<uint64_t>> partitions(numberOfComponents);
        for(const auto node : graph_.nodes()){
            const auto nodeLabel = nodeLabels[node];
            partitions[nodeLabel].push_back(node);
            buffer_.vertex_labels[node] = nodeLabel;
        }
        buffer_.max_not_used_label = partitions.size();


        //NodeLabelsType last_good_vertex_labels(graph_);

        auto & last_good_vertex_labels = nodeLabels;


        for(const auto node : graph_.nodes()){
            last_good_vertex_labels[node] = buffer_.vertex_labels[node];
        }


        // auxillary array for BFS/DFS
        typename GraphType:: template NodeMap<char> visited(graph_);

        // 1 if i-th partitioned changed since last iteration, 0 otherwise
        std::vector<char> changed(numberOfComponents, 1);

  
        // interatively update bipartition in order to minimize the total cost of the multicut
        // interatively update bipartition in order to minimize the total cost of the multicut
        for (auto k = 0; k < settings_.numberOfOuterIterations; ++k)
        {
            auto energy_decrease = .0;

            // TODO !!! replace with random access set
            //typedef nifty::container::BoostFlatSet<uint64_t> SetType;
            typedef std::set<uint64_t> SetType;
            std::vector<SetType> edges(numberOfComponents);
            for (const auto e : graph_.edges()){

                auto const v0 = buffer_.vertex_labels[graph_.u(e)];
                auto const v1 = buffer_.vertex_labels[graph_.v(e)];
                if (v0 != v1){
                    edges[std::min(v0, v1)].insert(std::max(v0, v1));
                }
            }


            for (auto i = 0; i < numberOfComponents; ++i){
                if (!partitions[i].empty()){
                    for (auto j : edges[i]){
                        if (!partitions[j].empty() && (changed[j] || changed[i])){


                            auto ret = update_bipartition(partitions[i], partitions[j]);

                            if (ret > settings_.epsilon){
                                changed[i] = changed[j] = 1;
                            }

                            energy_decrease += ret;

                            if (partitions[i].size() == 0){
                                break;
                            }
                        }
                    }
                }
            }
            


            auto ee = energy_decrease;

            // remove partitions that became empty after the previous step
            auto new_end = std::partition(partitions.begin(), partitions.end(), [](
                const std::vector<uint64_t>& s
            ) { 
                return !s.empty(); 
            });
            partitions.resize(new_end - partitions.begin());



            // try to intoduce new partitions
            for (int i = 0, p_size = int(partitions.size()); i < p_size; ++i)
            {
                if (!changed[i])
                    continue;

                while (1)
                {
                    std::vector<uint64_t> new_set;
                    energy_decrease += update_bipartition(partitions[i], new_set);

                    if (new_set.empty())
                        break;

                    partitions.emplace_back(std::move(new_set));
                }
            }

    
            if (!visitorProxy.visit(this)){
                break;
            }

            if (energy_decrease == .0)
                break;


            std::stack<uint64_t> S;
        
            std::fill(visited.begin(), visited.end(), 0);

            partitions.clear();
            numberOfComponents = 0;


            // do connected component labeling on the original graph and form new partitions

            for(const auto i : graph_.nodes()){
                if (!visited[i]){

                    S.push(i);
                    visited[i] = 1;

                    auto label = buffer_.vertex_labels[i];

                    buffer_.referenced_by[i] = numberOfComponents;

                    partitions.emplace_back(std::vector<uint64_t>());
                    partitions.back().push_back(i);

                    while (!S.empty())
                    {
                        auto v = S.top();
                        S.pop();

                        for(const auto adj : graph_.adjacency(v)){
                            if (buffer_.vertex_labels[adj.node()] == label && !visited[adj.node()])
                            {
                                S.push(adj.node());
                                visited[adj.node()] = 1;
                                buffer_.referenced_by[adj.node()] = numberOfComponents;
                                partitions.back().push_back(adj.node());
                            }
                        }
                    }
                    ++numberOfComponents;
                }
            }

            buffer_.vertex_labels = buffer_.referenced_by;
            buffer_.max_not_used_label = numberOfComponents;


            bool didnt_change = true;
            for(const auto i : graph_.edges()){
            

                auto const v0 = graph_.u(i);
                auto const v1 = graph_.v(i);

                auto edge_label = buffer_.vertex_labels[v0] == buffer_.vertex_labels[v1] ? 0 : 1;

                if (static_cast<bool>(edge_label) != (last_good_vertex_labels[v0] != last_good_vertex_labels[v1]))
                    didnt_change = false;
            }

            if (didnt_change){
                break;
            }

            changed.resize(numberOfComponents);

            std::fill(changed.begin(), changed.end(), 0);
            std::fill(visited.begin(), visited.end(), 0);






            for(const auto i : graph_.nodes()){
            
                if (!visited[i]){
                    S.push(i);
                    visited[i] = 1;

                    auto label_new = buffer_.vertex_labels[i];
                    auto label_old = last_good_vertex_labels[i];

                    while (!S.empty())
                    {
                        auto v = S.top();
                        S.pop();

                        for(const auto adj : graph_.adjacency(v)){

                        //for (auto w = graph.verticesFromVertexBegin(v); w != graph.verticesFromVertexEnd(v); ++w){



                            if (last_good_vertex_labels[adj.node()] == label_old && buffer_.vertex_labels[adj.node()] != label_new)
                                changed[label_new] = 1;

                            if (visited[adj.node()])
                                continue;

                            if (buffer_.vertex_labels[adj.node()] == label_new)
                            {
                                S.push(adj.node());
                                visited[adj.node()] = 1;

                                if (last_good_vertex_labels[adj.node()] != label_old)
                                    changed[label_new] = 1;
                            }
                        }
                    }
                }
            }

            for(const auto node: graph_.nodes()){
                last_good_vertex_labels[node] = buffer_.vertex_labels[node];
            }
            //currentBestEnergy_ -= (energy_decrease - ee);
            //std::cout<<"currentBestEnergy_"<<currentBestEnergy_<<"\n";
            if (false){ //settings.verbose){
                std::cout << std::setw(4) << k+1 << std::setw(16) << energy_decrease << std::setw(15) << ee << std::setw(15) << (energy_decrease - ee) << std::setw(14) << partitions.size() << std::endl;
            }


        }
      
        visitorProxy.end(this);
    }


    template<class OBJECTIVE>
    double KernighanLin<OBJECTIVE>::
    update_bipartition(
        std::vector<uint64_t>& A, 
        std::vector<uint64_t>& B
    ){

        struct Move
        {
            int v { -1 };
            double difference { std::numeric_limits<double>::lowest() };
            uint64_t new_label;
        };

        const auto & weights  = objective_.weights();

        auto gain_from_merging = .0;


        auto compute_differences = [&](
            const std::vector<uint64_t>& AA, 
            uint64_t label_A, 
            uint64_t label_B
        ){
            for (long int i = 0; i < AA.size(); ++i){

                double diffExt = .0;
                double diffInt = .0;
                uint64_t ref_cnt = 0;

                for(const auto adj : graph_.adjacency(AA[i])){
                    
                    const auto node = adj.node();
                    const auto edge = adj.edge();


                    const auto lbl = buffer_.vertex_labels[node];

                    if (lbl == label_A){
                        diffInt += weights[edge];
                    }
                    else if (lbl == label_B)
                    {
                        diffExt += weights[edge];
                        ++ref_cnt;
                    }
                }

                buffer_.differences[AA[i]] = diffExt - diffInt;
                buffer_.referenced_by[AA[i]] = ref_cnt;
                buffer_.is_moved[AA[i]] = 0;

                gain_from_merging += diffExt;
            }
        };

        if (A.empty())
            return .0;

        auto label_A = buffer_.vertex_labels[A[0]];
        auto label_B = (!B.empty()) ? buffer_.vertex_labels[B[0]] : buffer_.max_not_used_label;
        
        compute_differences(A, label_A, label_B);
        compute_differences(B, label_B, label_A);


        buffer_.border.clear();
        
        for (auto a : A)
            if (buffer_.referenced_by[a] > 0)
                buffer_.border.push_back(a);

        for (auto b : B)
            if (buffer_.referenced_by[b] > 0)
                buffer_.border.push_back(b);


        std::vector<Move> moves;
        double cumulative_diff = .0;
        std::pair<double, uint64_t> max_move { std::numeric_limits<double>::lowest(), 0 };

        for (auto k = 0; k < settings_.numberOfInnerIterations; ++k){
            Move m;

            if (B.empty() && k == 0){
                for (auto a : A){
                    if (buffer_.differences[a] > m.difference){
                        m.v = a;
                        m.difference = buffer_.differences[a];
                    }
                }
            }
            else{
                auto size = buffer_.border.size();
                
                for (auto i = 0; i < size; )
                    if (buffer_.referenced_by[buffer_.border[i]] == 0)
                        std::swap(buffer_.border[i], buffer_.border[--size]);
                    else
                    {
                        if (buffer_.differences[buffer_.border[i]] > m.difference)
                        {
                            m.v = buffer_.border[i];
                            m.difference = buffer_.differences[m.v];
                        }
                        
                        ++i;
                    }

                buffer_.border.erase(buffer_.border.begin() + size, buffer_.border.end());
            }


            if (m.v == -1)
                break;

            const auto old_label = buffer_.vertex_labels[m.v];

            if (old_label == label_A)
                m.new_label = label_B;
            else
                m.new_label = label_A;



            // update differences and references
            for(const auto adj : graph_.adjacency(m.v)){

                if (buffer_.is_moved[adj.node()]){
                    continue;
                }

                const auto lbl = buffer_.vertex_labels[adj.node()];
                // edge to an element of the new set
                if (lbl == m.new_label){
                    buffer_.differences[adj.node()] -= 2.0*weights[adj.edge()];
                    --buffer_.referenced_by[adj.node()];
                }
                // edge to an element of the old set
                else if (lbl == old_label){
                    buffer_.differences[adj.node()] += 2.0*weights[adj.edge()];
                    ++buffer_.referenced_by[adj.node()];

                    if (buffer_.referenced_by[adj.node()] == 1){
                        buffer_.border.push_back(adj.node());
                    }
                }
            }

            buffer_.vertex_labels[m.v] = m.new_label;
            buffer_.referenced_by[m.v] = 0;
            buffer_.differences[m.v] = std::numeric_limits<double>::lowest();
            buffer_.is_moved[m.v] = 1;
            moves.push_back(m);

            cumulative_diff += m.difference;

            if (cumulative_diff > max_move.first)
                max_move = std::make_pair(cumulative_diff, moves.size());

        }


        if (gain_from_merging > max_move.first && gain_from_merging > settings_.epsilon){
            A.insert(A.end(), B.begin(), B.end());

            for (auto a : A)
                buffer_.vertex_labels[a] = label_A;

            for (auto b : B)
                buffer_.vertex_labels[b] = label_A;

            B.clear();

            return gain_from_merging;
        }

        else if (max_move.first > settings_.epsilon)
        {
            // revert some changes
            for (uint64_t i = max_move.second; i < moves.size(); ++i)
            {
                buffer_.is_moved[moves[i].v] = 0;

                if (moves[i].new_label == label_B)
                    buffer_.vertex_labels[moves[i].v] = label_A;
                else
                    buffer_.vertex_labels[moves[i].v] = label_B;
            }

            if (B.empty())
                ++buffer_.max_not_used_label;

            A.erase(std::partition(A.begin(), A.end(), [&](uint64_t a) { return !buffer_.is_moved[a]; }), A.end());
            B.erase(std::partition(B.begin(), B.end(), [&](uint64_t b) { return !buffer_.is_moved[b]; }), B.end());

            for (uint64_t i = 0; i < max_move.second; ++i)
                // move vertex to the other set
                if (moves[i].new_label == label_B)
                    B.push_back(moves[i].v);
                else
                    A.push_back(moves[i].v);

            return max_move.first;
        }
        else{
            for (auto i = 0; i < moves.size(); ++i){
                if (moves[i].new_label == label_B){
                    buffer_.vertex_labels[moves[i].v] = label_A;
                }
                else{
                    buffer_.vertex_labels[moves[i].v] = label_B;
                }
            }
        }

        return .0;

    }

    template<class OBJECTIVE>
    const typename KernighanLin<OBJECTIVE>::Objective &
    KernighanLin<OBJECTIVE>::
    objective()const{
        return objective_;
    }


} // namespace nifty::graph::optimization::multicut
} // namespace nifty::graph::optimization
} // namespace nifty::graph
} // namespace nifty

