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
#ifndef ANDRES_GRAPH_VISITOR_HXX
#define ANDRES_GRAPH_VISITOR_HXX

#include <cstddef>
#include <iostream>

namespace andres {
namespace graph {

/// Visitors can be used to follow the indices of vertices and edges.
///
/// These indices change due to the insetion and removal of vertices and edges.
///
template<class S = std::size_t>
struct IdleGraphVisitor {
    typedef S size_type;

    IdleGraphVisitor() {}
    void insertVertex(const size_type a) const {}
    void insertVertices(const size_type a, const size_type n) const {}
    void eraseVertex(const size_type a) const {}
    void relabelVertex(const size_type a, const size_type b) const {}
    void insertEdge(const size_type a) const {}
    void eraseEdge(const size_type a) const {}
    void relabelEdge(const size_type a, const size_type b) const {}
};

/// Visitors can be used to follow the indices of vertices and edges.
///
/// These indices change due to the insetion and removal of vertices and edges.
///
template<class S = std::size_t>
struct VerboseGraphVisitor {
    typedef S size_type;

    VerboseGraphVisitor() {}
    void insertVertex(const size_type a) const
        { std::cout << "inserting vertex " << a << std::endl; }
    void insertVertices(const size_type a, const size_type n) const
        { std::cout << "inserting " << n << " vertices, starting from index " << a << std::endl; }
    void eraseVertex(const size_type a) const
        { std::cout << "removing vertex " << a << std::endl; }
    void relabelVertex(const size_type a, const size_type b) const
        { std::cout << "relabeling vertex " << a << ". new label is " << b << std::endl; }
    void insertEdge(const size_type a) const
        { std::cout << "inserting edge " << a << std::endl; }
    void eraseEdge(const size_type a) const
        { std::cout << "removing edge " << a << std::endl; }
    void relabelEdge(const size_type a, const size_type b) const
        { std::cout << "relabeling edge " << a << ". new label is " << b << std::endl; }
};

} // namespace graph
} // namespace andres

#endif // #ifndef ANDRES_GRAPH_VISITOR_HXX
