#pragma once

#include <cmath>
#include "boost/functional/hash.hpp"

#include "z5/dataset_factory.hxx"
#include "z5/util/for_each.hxx"

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


    inline void deserializeContingencyTable(const std::vector<uint64_t> & serialization,
                                            ContingencyTable & table) {
        std::size_t index = 0;
        while(index < serialization.size()) {
            const uint64_t lA = serialization[index];
            ++index;
            const uint64_t lB = serialization[index];
            ++index;
            const uint64_t count = serialization[index];
            ++index;

            Element elem = std::make_pair(lA, lB);
            auto elemIt = table.find(elem);
            if(elemIt == table.end()) {
                table[elem] = count;
            } else {
                elemIt->second += count;
            }
        }
    }


    // TODO could try to paralleilze this
    inline void mergeContingencyTables(const std::string & inputPath,
                                       ContingencyTable & table) {
        auto inputDs = z5::openDataset(inputPath);
        z5::util::parallel_for_each_chunk(*inputDs, 1,
                                          [&table](const int tId,
                                                     const z5::Dataset & ds,
                                                     const z5::types::ShapeType & chunkCoord){
            // read this chunk's data (if present)
            z5::handle::Chunk chunk(ds.handle(), chunkCoord, ds.isZarr());
            if(!chunk.exists()) {
                return;
            }
            bool isVarlen;
            const std::size_t chunkSize = ds.getDiscChunkSize(chunkCoord, isVarlen);
            std::vector<uint64_t> serialization(chunkSize);
            ds.readChunk(chunkCoord, &serialization[0]);
            deserializeContingencyTable(serialization, table);
        });
    }


    inline void computeRandPrimitives(const ContingencyTable & table,
                                      const std::vector<uint64_t> & rowSum,
                                      const std::vector<uint64_t> & colSum,
                                      const double aux,
                                      std::map<std::string, double> & primitives) {
        const std::size_t nLabelsA = rowSum.size();
        const std::size_t nLabelsB = colSum.size();

        double randA = 0;
        double randB = 0;
        double randAB = 0;

        // sum of square of rows
        for(size_t i = 0; i < rowSum.size(); ++i){
            randA += rowSum[i] * rowSum[i];
        }

        // sum of square of cols
        for( size_t j = 0; j < colSum.size(); ++j){
            randB += colSum[j] * colSum[j];
        }
        randB += aux;

        for(size_t i = 1; i < nLabelsA; ++i){
            for(size_t j = 1; j < nLabelsB; ++j){
                auto elemIt = table.find(std::make_pair(i, j));
                if(elemIt != table.end()) {
                    randAB += elemIt->second * elemIt->second;
                }
            }
        }
        randAB += aux;

        primitives["randA"] = randA;
        primitives["randB"] = randB;
        primitives["randAB"] = randAB;
    }


    inline void computeViPrimitives(const ContingencyTable & table,
                                    const std::vector<uint64_t> & rowSum,
                                    const std::vector<uint64_t> & colSum,
                                    const std::size_t nPoints,
                                    const double aux,
                                    std::map<std::string, double> & primitives) {
        const std::size_t nLabelsA = rowSum.size();
        const std::size_t nLabelsB = colSum.size();

        double viA = 0;
        double viB = 0;
        double viAB = 0;

        // sum of square of rows
        for(size_t i = 0; i < rowSum.size(); ++i){
            if(rowSum[i] == 0) {
                continue;
            }
            const double val = static_cast<double>(rowSum[i]) / nPoints;
            viA += val * log(val);
        }

        // sum of square of cols
        for(size_t j = 0; j < colSum.size(); ++j){
            if(colSum[j] == 0) {
                continue;
            }
            const double val = static_cast<double>(colSum[j]) / nPoints;
            viB += val * log(val);
        }
        viB -= aux * log(nPoints);

        for(size_t i = 1; i < nLabelsA; ++i){
            for(size_t j = 1; j < nLabelsB; ++j){

                const auto elemIt = table.find(std::make_pair(i, j));
                if(elemIt == table.end()) {
                    continue;
                }

                const double val = static_cast<double>(elemIt->second) / nPoints;
                viAB += val * log(val);
            }
        }
        viAB -= aux / log(nPoints);

        primitives["viA"] = viA;
        primitives["viB"] = viB;
        primitives["viAB"] = viAB;
    }


    inline void computeEvalPrimitives(const std::string & inputPath,
                                      const std::size_t nPoints,
                                      const std::size_t nLabelsA,
                                      const std::size_t nLabelsB,
                                      std::map<std::string, double> & primitives,
                                      const int nThreads) {
        // merge the contingency tables
        ContingencyTable table;
        mergeContingencyTables(inputPath, table);

        // compute row and column sums
        std::vector<uint64_t> rowSum(nLabelsA);
        std::vector<uint64_t> colSum(nLabelsB);

        for(size_t i = 0; i < nLabelsA; ++i){
            for(size_t j = 0; j < nLabelsB; ++j){
                auto elemIt = table.find(std::make_pair(i, j));
                if(elemIt != table.end()) {
                    rowSum[i] += elemIt->second;
                }
            }
        }

        for(size_t j = 0; j < nLabelsB; ++j){
            for(size_t i = 0; i < nLabelsA; ++i){
                auto elemIt = table.find(std::make_pair(i, j));
                if(elemIt != table.end()) {
                    colSum[j] += elemIt->second;
                }
            }
        }

        double aux = 0.;
        for( size_t i = 0; i < nLabelsA ; ++i) {
            auto elemIt = table.find(std::make_pair(i, 0));
            if(elemIt != table.end()) {
                aux += elemIt->second;
            }
        }
        aux /= nPoints;

        // compute primitives for rand
        computeRandPrimitives(table, rowSum, colSum, aux, primitives);

        // compute primitives for vi
        computeViPrimitives(table, rowSum, colSum, nPoints, aux, primitives);

        // serialize nPoints
        primitives["nPoints"] = nPoints;
    }


}
}
