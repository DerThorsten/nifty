cmake_minimum_required(VERSION 2.8)

SET(QPBO_SRC_FILES
  QPBO_extra.cpp
  QPBO.cpp
  QPBO_maxflow.cpp
  QPBO_postprocessing.cpp
)

ADD_LIBRARY(inferno-external-qpbo ${QPBO_SRC_FILES})
ADD_LIBRARY(inferno-external-qpbo-shared SHARED ${QPBO_SRC_FILES})

install (TARGETS inferno-external-qpbo DESTINATION lib/nifty_externals/)
install (TARGETS inferno-external-qpbo-shared DESTINATION lib/nifty_externals/)

install (FILES QPBO.h DESTINATION include/inferno_externals/qpbo)
install (FILES block.h DESTINATION include/inferno_externals/qpbo)
