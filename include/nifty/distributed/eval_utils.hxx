#pragma once

#include "boost/functional/hash.hpp"

#include "z5/dataset_factory.hxx"
#include "nifty/xtensor/xtensor.hxx"
#include "nifty/tools/for_each_coordinate.hxx"

namespace fs = boost::filesystem;

namespace nifty {
namespace distributed {

    typedef std::pair<uint64_t, uint64_t> Element;
    typedef std::unordered_map<Element, std::size_t, boost::hash<Element>> ContingencyTable;

    template<class SEG>
    inline void computeContingencyTable(const SEG & segA, const SEG & segB, ContingencyTable & table,
                                        const uint64_t ignoreA=0, const uint64_t ignoreB=0) {

        typedef nifty::array::StaticArray<int64_t, 3> Coord;
        const auto & shape = segA.shape();
        Coord iterShape = {shape[0], shape[1], shape[2]};

        // compute the contingency matrix
        tools::forEachCoordinate(iterShape, [&](const Coord & coord){
            const uint64_t labelA = xtensor::read(segA, coord.asStdArray());
            const uint64_t labelB = xtensor::read(segB, coord.asStdArray());

            if(labelA == ignoreA || labelB == ignoreB) {
                return;
            }

            Element elem = std::make_pair(labelA, labelB);
            auto elemIt = table.find(elem);
            if(elemIt == table.end()) {
                table[elem] = 1;
            } else {
                ++elemIt->second;
            }
        });
    }


    inline void serializeContingencyTable(const ContingencyTable & table,
                                          const std::string & path,
                                          const std::vector<std::size_t> & chunkId) {

        const std::size_t serSize = 3 * table.size();
        std::vector<uint64_t> serialization(serSize);

        std::size_t index = 0;
        for(const auto elem: table) {
            serialization[index] = elem.first.first;
            ++index;
            serialization[index] = elem.first.second;
            ++index;
            serialization[index] = elem.second;
            ++index;
        }

        // FIXME not parallelization save ???
        auto ds = z5::openDataset(path);
        ds->writeChunk(chunkId, &serialization[0], true, serSize);
    }


    template<class SEG>
    inline void computeAndSerializeContingecyTable(const SEG & segA, const SEG & segB,
                                                   const std::string & path, const std::vector<std::size_t> & chunkId,
                                                   const uint64_t ignoreA=0, const uint64_t ignoreB=0) {
        ContingencyTable table;
        computeContingencyTable(segA, segB, table, ignoreA, ignoreB);
        serializeContingencyTable(table, path, chunkId);
    }


    // TODO
    inline void mergeContingencyTables() {

    }


}
}
