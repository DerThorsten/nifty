// This header was copied from https://github.com/bjoern-andres/graph.
// It comes with the following license:
/*
Copyright (c) by Bjoern Andres (bjoern@andres.sc).

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

    Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    The name of the author must not be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once
#ifndef ANDRES_GRAPH_ADJACENCY_HXX
#define ANDRES_GRAPH_ADJACENCY_HXX

namespace andres {
namespace graph {

/// The adjacency of a vertex consists of a vertex and a connecting edge.
template<class T = std::size_t>
class Adjacency {
public:
    typedef T Value;

    Adjacency(const Value = T(), const Value = T());
    Value vertex() const;
    Value& vertex();
    Value edge() const;
    Value& edge();
    bool operator<(const Adjacency<Value>&) const;
    bool operator<=(const Adjacency<Value>&) const;
    bool operator>(const Adjacency<Value>&) const;
    bool operator>=(const Adjacency<Value>&) const;
    bool operator==(const Adjacency<Value>&) const;
    bool operator!=(const Adjacency<Value>&) const;

private:
    Value vertex_;
    Value edge_;
};

/// Construct an adjacency.
///
/// \param vertex Vertex.
/// \param edge Edge.
///
template<class T>
inline
Adjacency<T>::Adjacency(
    const Value vertex,
    const Value edge
)
:   vertex_(vertex),
    edge_(edge)
{}

/// Access the vertex.
///
template<class T>
inline typename Adjacency<T>::Value
Adjacency<T>::vertex() const {
    return vertex_;
}

/// Access the vertex.
///
template<class T>
inline typename Adjacency<T>::Value&
Adjacency<T>::vertex() {
    return vertex_;
}

/// Access the edge.
///
template<class T>
inline typename Adjacency<T>::Value
Adjacency<T>::edge() const {
    return edge_;
}

/// Access the edge.
///
template<class T>
inline typename Adjacency<T>::Value&
Adjacency<T>::edge() {
    return edge_;
}

/// Adjacencies are ordered first wrt the vertex, then wrt the edge.
///
template<class T>
inline bool
Adjacency<T>::operator<(
    const Adjacency<T>& in
) const {
    if(vertex_ < in.vertex_) {
        return true;
    }
    else if(vertex_ == in.vertex_) {
        return edge_ < in.edge_;
    }
    else {
        return false;
    }
}

/// Adjacencies are ordered first wrt the vertex, then wrt the edge.
///
template<class T>
inline bool
Adjacency<T>::operator<=(
    const Adjacency<T>& in
) const {
    if(vertex_ < in.vertex_) {
        return true;
    }
    else if(vertex_ == in.vertex_) {
        return edge_ <= in.edge_;
    }
    else {
        return false;
    }
}

/// Adjacencies are ordered first wrt the vertex, then wrt the edge.
///
template<class T>
inline bool
Adjacency<T>::operator>(
    const Adjacency<T>& in
) const {
    if(vertex_ > in.vertex_) {
        return true;
    }
    else if(vertex_ == in.vertex_) {
        return edge_ > in.edge_;
    }
    else {
        return false;
    }
}

/// Adjacencies are ordered first wrt the vertex, then wrt the edge.
///
template<class T>
inline bool
Adjacency<T>::operator>=(
    const Adjacency<T>& in
) const {
    if(vertex_ > in.vertex_) {
        return true;
    }
    else if(vertex_ == in.vertex_) {
        return edge_ >= in.edge_;
    }
    else {
        return false;
    }
}

/// Adjacencies are equal if both the vertex and the edge are equal.
///
template<class T>
inline bool
Adjacency<T>::operator==(
    const Adjacency<T>& in
) const {
    return vertex_ == in.vertex_ && edge_ == in.edge_;
}

/// Adjacencies are unequal if either the vertex or the edge are unqual.
///
template<class T>
inline bool
Adjacency<T>::operator!=(
    const Adjacency<T>& in
) const {
    return !(*this == in);
}

} // namespace graph
} // namespace andres

#endif // #ifndef ANDRES_GRAPH_ADJACENCY_HXX
