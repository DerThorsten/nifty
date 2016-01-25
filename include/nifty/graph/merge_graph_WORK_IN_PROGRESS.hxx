#ifndef NIFTY_GRAPH_MERGE_GRAPH_HXX
#define NIFTY_GRAPH_MERGE_GRAPH_HXX



/* std library */
#include <vector>
#include <functional>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <map>



namespace nifty {
namespace graph{


// callbacks of merge graph 
// to update node and edge maps w.r.t. edge contractions
template<class MG>
class MergeGraphCallbacks{
    public:

        typedef std::function<void (const int64_t ,const int64_t)>   MergeNodeCallBackType;
        typedef std::function<void (const int64_t ,const int64_t)>   MergeEdgeCallBackType;
        typedef std::function<void (const int64_t)>                  EraseEdgeCallBackType;

        MergeGraphCallbacks(){}

        void registerMergeNodeCallBack(MergeNodeCallBackType  f){
            mergeNodeCallbacks_.push_back(f);
        }
        void registerMergeEdgeCallBack(MergeEdgeCallBackType  f){
            mergeEdgeCallbacks_.push_back(f);
        }
        void registerEraseEdgeCallBack(EraseEdgeCallBackType  f){
            eraseEdgeCallbacks_.push_back(f);
        }

    protected:
        void callMergeNodeCallbacks(const int64_t a,const NODE & b){
            for(size_t i=0;i<mergeNodeCallbacks_.size();++i)
                mergeNodeCallbacks_[i](a,b);
        }
        void callMergeEdgeCallbacks(const int64_t a,const int64_t b){
            for(size_t i=0;i<mergeEdgeCallbacks_.size();++i)
                mergeEdgeCallbacks_[i](a,b);
        }
        void callEraseEdgeCallbacks(const int64_t a){
            for(size_t i=0;i<eraseEdgeCallbacks_.size();++i)
                eraseEdgeCallbacks_[i](a);
        }
        void clearCallbacks(){
            mergeNodeCallbacks_.clear();
            mergeEdgeCallbacks_.clear();
            eraseEdgeCallbacks_.clear();
        }
    private:
        std::vector<MergeNodeCallBackType> mergeNodeCallbacks_;
        std::vector<MergeEdgeCallBackType> mergeEdgeCallbacks_;
        std::vector<EraseEdgeCallBackType> eraseEdgeCallbacks_;
};



/** \brief undirected graph adaptor 
      for edge contraction and feature merging
    */
template<class GRAPH>
class MergeGraphAdaptor 
:   public MergeGraphCallbacks<
        MergeGraphCallbacks<GRAPH>
    > 

{

    public:
    typedef MergeGraphAdaptor<GRAPH> MergeGraphType;
    typedef GRAPH Graph;




    

    //typedef  std::set<index_type>   NodeStorageEdgeSet;
    typedef detail::GenericNodeImpl<index_type,false >  NodeStorage;


    private:
        
        typedef nifty::ufd::IterablePartition<IdType> UfdType;
        typedef typename UfdType::const_iterator ConstUdfIter;
        typedef ConstUdfIter                                                EdgeIdIt;
        typedef ConstUdfIter                                                NodeIdIt;

    public:

        
    private:
        MergeGraphAdaptor() = delete;                               // non empty-construction
        MergeGraphAdaptor( const MergeGraphAdaptor& other ) = delete;      // non construction-copyable
        MergeGraphAdaptor& operator=( const MergeGraphAdaptor& ) = delete; // non copyable
    public:
        MergeGraphAdaptor(const Graph &  graph);
        //void   setInitalEdge(const size_t initEdge,const size_t initNode0,const size_t initNode1);

        // query (sizes) 
        size_t edgeNum()const;
        size_t nodeNum()const;

        IdType maxEdgeId()const;
        IdType maxNodeId()const;


        // query (iterators )
        EdgeIdIt  edgeIdsBegin()const;
        EdgeIdIt  edgeIdsEnd()const;
        NodeIdIt  nodeIdsBegin()const;
        NodeIdIt  nodeIdsEnd()const;









        // query ( has edge )
        bool hasEdgeId(const IdType edgeIndex)const;
        bool hasNodeId(const IdType nodeIndex)const;



        Edge findEdge(const Node & a,const Node & b)const;





        size_t degree(const Node & node)const;



        Node  u(const Edge & edge)const;
        Node  v(const Edge & edge)const;



        // query (w.r.t. inital nodesIds/edgesIds)
        IdType reprEdgeId(const IdType edgeIndex)const;
        IdType reprNodeId(const IdType nodeIndex)const;
        bool stateOfInitalEdge(const IdType initalEdge)const;
        // modification
        void contractEdge(const Edge & edge);


    
        // special merge graph members 
        GraphEdge reprGraphEdge(const GraphEdge & edge)const{
            return  graph_.edgeFromId(reprEdgeId(graph_.id(edge)));
        }
        GraphNode reprGraphNode(const GraphNode & node)const{
            return graph_.nodeFromId(reprNodeId(graph_.id(node)));
        }


        Edge reprEdge(const GraphEdge & edge)const{
            return  edgeFromId(reprEdgeId(graph_.id(edge)));
        }
        Node reprNode(const GraphNode & node)const{
            return nodeFromId(reprNodeId(graph_.id(node)));
        }

        const Graph & graph()const{
            return graph_;
        }
        const Graph & graph(){
            return graph_;
        }

        // in which node is a "merged inactive" edge
        Node inactiveEdgesNode(const Edge edge)const{
            return reprNodeId(graphUId(id(edge)));
        }
        size_t maxDegree()const{
            size_t md=0;
            for(NodeIt it(*this);it!=lemon::INVALID;++it){
                std::max(md, size_t( degree(*it) ) );
            }
            return md;
        }

        void reset();

    private:
        // needs acces to const nodeImpl
        template<class G,class NIMPL,class FILT>
        friend class detail::GenericIncEdgeIt;

        template<class G>
        friend struct detail::NeighborNodeFilter;
        template<class G>
        friend struct detail::IncEdgeFilter;
        template<class G>
        friend struct detail::BackEdgeFilter;
        template<class G>
        friend struct detail::IsOutFilter;
        template<class G>
        friend struct detail::IsInFilter;
        friend class MergeGraphNodeIt<MergeGraphType>;
        friend class MergeGraphEdgeIt<MergeGraphType>;


        index_type  graphUId(const index_type edgeId)const;
        index_type  graphVId(const index_type edgeId)const;
        //index_type  uId(const Edge & edge)const{return uId(id(edge));}
        //index_type  vId(const Edge & edge)const{return vId(id(edge));}
        const NodeStorage & nodeImpl(const Node & node)const{
            return nodeVector_[id(node)];
        }
        NodeStorage & nodeImpl(const Node & node){
            return nodeVector_[id(node)];
        }


        const GRAPH & graph_;
        UfdType nodeUfd_;
        UfdType edgeUfd_;

        std::vector< NodeStorage >  nodeVector_;

        size_t nDoubleEdges_;
        std::vector<std::pair<index_type,index_type> > doubleEdges_;
};


template<class GRAPH>
MergeGraphAdaptor<GRAPH>::MergeGraphAdaptor(const GRAPH & graph )
:   MergeGraphCallbacks<Node,Edge >(),
    graph_(graph),
    nodeUfd_(graph.maxNodeId()+1),
    edgeUfd_(graph.maxEdgeId()+1),
    nodeVector_(graph.maxNodeId()+1),
    nDoubleEdges_(0),
    doubleEdges_(graph_.edgeNum()/2 +1)
{
    for(index_type possibleNodeId = 0 ; possibleNodeId <= graph_.maxNodeId(); ++possibleNodeId){
        if(graph_.nodeFromId(possibleNodeId)==lemon::INVALID){
            nodeUfd_.eraseElement(possibleNodeId);
        }
        else{
            nodeVector_[possibleNodeId].id_ = possibleNodeId;
        }
    }
    for(index_type possibleEdgeId = 0 ; possibleEdgeId <= graph_.maxEdgeId(); ++possibleEdgeId){
        const GraphEdge possibleEdge(graph_.edgeFromId(possibleEdgeId));
        if(possibleEdge==lemon::INVALID){
            edgeUfd_.eraseElement(possibleEdgeId);
        }
        else{
            const index_type guid = graphUId(possibleEdgeId);
            const index_type gvid = graphVId(possibleEdgeId);
            nodeVector_[ guid ].insert(gvid,possibleEdgeId);
            nodeVector_[ gvid ].insert(guid,possibleEdgeId);   
        }
    }
    
}


template<class GRAPH>
void MergeGraphAdaptor<GRAPH>::reset  (){

    nodeUfd_.reset(graph_.maxNodeId()+1),
    edgeUfd_.reset(graph_.maxEdgeId()+1),

    this->clearCallbacks();

    // clean nodes_
    for(index_type possibleNodeId = 0 ; possibleNodeId <= graph_.maxNodeId(); ++possibleNodeId){

        nodeVector_[possibleNodeId].clear();
        if(graph_.nodeFromId(possibleNodeId)==lemon::INVALID){
            nodeUfd_.eraseElement(possibleNodeId);
        }
        else{
            nodeVector_[possibleNodeId].id_ = possibleNodeId;
        }
    }

    for(index_type possibleEdgeId = 0 ; possibleEdgeId <= graph_.maxEdgeId(); ++possibleEdgeId){
        const GraphEdge possibleEdge(graph_.edgeFromId(possibleEdgeId));
        if(possibleEdge==lemon::INVALID){
            edgeUfd_.eraseElement(possibleEdgeId);
        }
        else{
            const index_type guid = graphUId(possibleEdgeId);
            const index_type gvid = graphVId(possibleEdgeId);
            nodeVector_[ guid ].insert(gvid,possibleEdgeId);
            nodeVector_[ gvid ].insert(guid,possibleEdgeId);   
        }
    }
}


template<class GRAPH>
inline  typename MergeGraphAdaptor<GRAPH>::Edge
MergeGraphAdaptor<GRAPH>::findEdge  (
    const typename MergeGraphAdaptor<GRAPH>::Node & a,
    const typename MergeGraphAdaptor<GRAPH>::Node & b
)const{

    if(a!=b){
        std::pair<index_type,bool> res =  nodeVector_[id(a)].findEdge(id(b));
        if(res.second){
            return Edge(res.first);
        }
    }
    return Edge(lemon::INVALID);
}

template<class GRAPH>
inline typename MergeGraphAdaptor<GRAPH>::Node 
MergeGraphAdaptor<GRAPH>::u(const Edge & edge)const{
    return nodeFromId(uId(id(edge)));
}

template<class GRAPH>
inline typename MergeGraphAdaptor<GRAPH>::Node 
MergeGraphAdaptor<GRAPH>::v(const Edge & edge)const{
    return nodeFromId(vId(id(edge)));
}




template<class GRAPH>
inline typename MergeGraphAdaptor<GRAPH>::index_type 
MergeGraphAdaptor<GRAPH>::graphUId(const index_type edgeId)const{
    return graph_.id(graph_.u(graph_.edgeFromId(edgeId)));
}

template<class GRAPH>
inline typename MergeGraphAdaptor<GRAPH>::index_type 
MergeGraphAdaptor<GRAPH>::graphVId(const index_type edgeId)const{
    return graph_.id(graph_.v(graph_.edgeFromId(edgeId)));
}


template<class GRAPH>
inline typename MergeGraphAdaptor<GRAPH>::IdType 
MergeGraphAdaptor<GRAPH>::maxEdgeId()const {
    return static_cast<index_type>(edgeUfd_.lastRep());
}
template<class GRAPH>
inline typename MergeGraphAdaptor<GRAPH>::IdType 
MergeGraphAdaptor<GRAPH>::maxNodeId()const {
    return static_cast<index_type>(nodeUfd_.lastRep());
}

template<class GRAPH>
inline size_t 
MergeGraphAdaptor<GRAPH>::degree(
    typename MergeGraphAdaptor<GRAPH>::Node const & node
)const{
    return static_cast<size_t>( nodeVector_[id(node)].edgeNum() );
}



template<class GRAPH>
inline typename MergeGraphAdaptor<GRAPH>::EdgeIdIt 
MergeGraphAdaptor<GRAPH>::edgeIdsBegin()const{
    return edgeUfd_.begin();
}

template<class GRAPH>
inline typename MergeGraphAdaptor<GRAPH>::EdgeIdIt 
MergeGraphAdaptor<GRAPH>::edgeIdsEnd()const{
    return edgeUfd_.end();
}


template<class GRAPH>
inline typename MergeGraphAdaptor<GRAPH>::NodeIdIt 
MergeGraphAdaptor<GRAPH>::nodeIdsBegin()const{
    return nodeUfd_.begin();
}

template<class GRAPH>
inline typename MergeGraphAdaptor<GRAPH>::NodeIdIt 
MergeGraphAdaptor<GRAPH>::nodeIdsEnd()const{
    return nodeUfd_.end();
}





template<class GRAPH>
inline bool 
MergeGraphAdaptor<GRAPH>::hasEdgeId(
    const typename MergeGraphAdaptor<GRAPH>::IdType edgeIndex
)const{
    if(edgeIndex<=maxEdgeId() && !edgeUfd_.isErased(edgeIndex)){
        const IdType reprEdgeIndex = reprEdgeId(edgeIndex);
        if(reprEdgeIndex!=edgeIndex){
            return false;
        }
        else{
            const index_type rnid0=  uId(reprEdgeIndex);
            const index_type rnid1=  vId(reprEdgeIndex);
            return rnid0!=rnid1;
        }
    }
    else{
        return false;
    }
}

template<class GRAPH>
inline bool 
MergeGraphAdaptor<GRAPH>::hasNodeId(
    const typename MergeGraphAdaptor<GRAPH>::IdType nodeIndex
)const{

    return nodeIndex<=maxNodeId() &&  !nodeUfd_.isErased(nodeIndex) && nodeUfd_.find(nodeIndex)==nodeIndex;
}

template<class GRAPH>
inline typename MergeGraphAdaptor<GRAPH>::IdType 
MergeGraphAdaptor<GRAPH>::reprEdgeId(
    const typename MergeGraphAdaptor<GRAPH>::IdType edgeIndex
)const{
    return edgeUfd_.find(edgeIndex);
}

template<class GRAPH>
inline typename MergeGraphAdaptor<GRAPH>::IdType 
MergeGraphAdaptor<GRAPH>::reprNodeId(
    const typename MergeGraphAdaptor<GRAPH>::IdType nodeIndex
)const{
    return nodeUfd_.find(nodeIndex);
}

template<class GRAPH>
inline bool MergeGraphAdaptor<GRAPH>::stateOfInitalEdge(
    const typename MergeGraphAdaptor<GRAPH>::IdType initalEdge
)const{
    const index_type rep = reprEdgeId(initalEdge);

    const index_type rnid0=  reprNodeId( graphUId(initalEdge) );
    const index_type rnid1=  reprNodeId( graphVId(initalEdge) );
    return rnid0!=rnid1;
}

template<class GRAPH>
inline size_t MergeGraphAdaptor<GRAPH>::nodeNum()const{
    return nodeUfd_.numberOfSets();
}

template<class GRAPH>
inline size_t MergeGraphAdaptor<GRAPH>::edgeNum()const{
    return edgeUfd_.numberOfSets();
}

template<class GRAPH>
void MergeGraphAdaptor<GRAPH>::contractEdge(
    const typename MergeGraphAdaptor<GRAPH>::Edge & toDeleteEdge
){
    //std::cout<<"node num "<<nodeNum()<<"\n";
    const index_type toDeleteEdgeIndex = id(toDeleteEdge);
    const index_type nodesIds[2]={id(u(toDeleteEdge)),id(v(toDeleteEdge))};

    // merge the two nodes
    nodeUfd_.merge(nodesIds[0],nodesIds[1]);
    const IdType newNodeRep    = reprNodeId(nodesIds[0]);
    const IdType notNewNodeRep =  (newNodeRep == nodesIds[0] ? nodesIds[1] : nodesIds[0] );

    typename NodeStorage::AdjIt iter=nodeVector_[notNewNodeRep].adjacencyBegin();
    typename NodeStorage::AdjIt end =nodeVector_[notNewNodeRep].adjacencyEnd();
   
    nDoubleEdges_=0;
    for(;iter!=end;++iter){
        const size_t adjToDeadNodeId = iter->nodeId(); 
        if(adjToDeadNodeId!=newNodeRep){

            // REFACTOR ME,  we can make that faster if
            // we do that in set intersect style
            std::pair<index_type,bool> found=nodeVector_[adjToDeadNodeId].findEdge(newNodeRep);


            if(found.second){
                edgeUfd_.merge(iter->edgeId(),found.first);
                
                const index_type edgeA = iter->edgeId();
                const index_type edgeB = found.first;
                const index_type edgeR  = edgeUfd_.find(edgeA);
                const index_type edgeNR = edgeR==edgeA ? edgeB : edgeA; 

                nodeVector_[adjToDeadNodeId].eraseFromAdjacency(notNewNodeRep);

                // refactor me ... this DOES NOT change the key
                nodeVector_[adjToDeadNodeId].eraseFromAdjacency(newNodeRep);
                nodeVector_[adjToDeadNodeId].insert(newNodeRep,edgeR);

                // refactor me .. this DOES NOT change the key
                nodeVector_[newNodeRep].eraseFromAdjacency(adjToDeadNodeId);
                nodeVector_[newNodeRep].insert(adjToDeadNodeId,edgeR);

                doubleEdges_[nDoubleEdges_]=std::pair<index_type,index_type>(edgeR,edgeNR );
                ++nDoubleEdges_;
            }
            else{
                nodeVector_[adjToDeadNodeId].eraseFromAdjacency(notNewNodeRep);
                //nodeVector_[adjToDeadNodeId].eraseFromAdjacency(newNodeRep);
                nodeVector_[adjToDeadNodeId].insert(newNodeRep,iter->edgeId());

                // symetric
                //nodeVector_[newNodeRep].eraseFromAdjacency(adjToDeadNodeId);
                nodeVector_[newNodeRep].insert(adjToDeadNodeId,iter->edgeId());

            }
        }
    }

    //nodeVector_[newNodeRep].merge(nodeVector_[notNewNodeRep]);
    nodeVector_[newNodeRep].eraseFromAdjacency(notNewNodeRep);
    //nodeVector_[newNodeRep].eraseFromAdjacency(newNodeRep); // no self adjacecy
    nodeVector_[notNewNodeRep].clear();
    
    edgeUfd_.eraseElement(toDeleteEdgeIndex);

    //std::cout<<"merge nodes callbacks\n";
    
    this->callMergeNodeCallbacks(Node(newNodeRep),Node(notNewNodeRep));

    //std::cout<<"merge double edge callbacks\n";
    for(size_t de=0;de<nDoubleEdges_;++de){
        this->callMergeEdgeCallbacks(Edge(doubleEdges_[de].first),Edge(doubleEdges_[de].second));
    }
    //std::cout<<"erase edge callbacks\n";
    this->callEraseEdgeCallbacks(Edge(toDeleteEdgeIndex));

    //std::cout<<"and done\n";
}



} // end namespace nifty::graph
} // end namespace nifty



#endif //NIFTY_GRAPH_MERGE_GRAPH_HXX
