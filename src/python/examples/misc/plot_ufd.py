"""
Union-Find Data Structure
================================

The wikipedia definition of union find data structure is excellent
therefore we will just cite it here:

    In computer science, 
    a disjoint-set data structure, 
    also called a union–find data structure or merge–find set, 
    is a data structure that keeps track of a set of elements 
    partitioned into a number of disjoint (non-overlapping) subsets. 
    It provides near-constant-time operations (bounded by the inverse Ackermann function) to add new sets, 
    to merge existing sets, 
    and to determine whether elements are in the same set. In addition to many other uses (see the Applications section), 
    disjoint-sets play a key role in Kruskal's algorithm for finding the minimum spanning tree of a graph.

In this example we show how to use a fast UFD implementation.


"""
import numpy
import nifty.ufd



# create a ufd with 10 elements
ufd = nifty.ufd.ufd(size=10)

# initially all elements are in their own set
assert ufd.numberOfSets == ufd.numberOfElements
print("NumberOfSets", ufd.numberOfSets)
print("NumberOfElements", ufd.numberOfElements)


# 
for element in range(10):
    print("find %d  = "%element, ufd.find(element))


# do a merge
print("merge 2 and 3")
ufd.merge(2,3)
print("NumberOfSets", ufd.numberOfSets)


# element 2 and 3 are now in the same set
assert ufd.find(2) == ufd.find(3)

# find can also be called 
# in a vectorized fashion
# this is way faster for 
# if a large number of elements
# should be found
toFind = numpy.array([2,3])
print("find [2, 3] = ",ufd.find(toFind))


# elment merging can also be done in a 
# a vectorized fashion
# here we merge 4 with 5  and also
# 7 with 
print("merge 2 with 3 and 7 with 8")
toMerge  = numpy.array([[4,5], [7,8]], dtype='uint64')
ufd.merge(toMerge)
assert ufd.find(4) == ufd.find(5)
assert ufd.find(7) == ufd.find(8)
print("NumberOfSets", ufd.numberOfSets)

# find all elements
setIds = ufd.find(numpy.arange(ufd.numberOfElements))
for element in range(10):
    print("find %d  = "%element, ufd.find(setIds[element]))


# if we now merge 3 and 4
# 2 and 5 are merged transitively
ufd.merge(3,4)
assert ufd.find(3) == ufd.find(4)
assert ufd.find(2) == ufd.find(5)


print(ufd.find(numpy.arange(ufd.numberOfElements)))