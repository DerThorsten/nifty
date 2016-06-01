add_executable(test_undirected_graph test_undirected_graph.cxx )
target_link_libraries(test_undirected_graph ${TEST_LIBS})
add_test(test_undirected_graph test_undirected_graph)

add_executable(test_shortest_path_dijkstra test_shortest_path_dijkstra.cxx )
target_link_libraries(test_shortest_path_dijkstra ${TEST_LIBS})
add_test(test_shortest_path_dijkstra test_shortest_path_dijkstra)

add_executable(test_shortest_path_bellman_ford test_shortest_path_bellman_ford.cxx )
target_link_libraries(test_shortest_path_bellman_ford ${TEST_LIBS})
add_test(test_shortest_path_bellman_ford test_shortest_path_bellman_ford)

add_executable(test_breadth_first_search test_breadth_first_search.cxx )
target_link_libraries(test_breadth_first_search ${TEST_LIBS})
add_test(test_breadth_first_search test_breadth_first_search)

add_executable(test_depth_first_search test_depth_first_search.cxx )
target_link_libraries(test_depth_first_search ${TEST_LIBS})
add_test(test_depth_first_search test_depth_first_search)

add_executable(test_multicut test_multicut.cxx )
target_link_libraries(test_multicut ${TEST_LIBS})
add_test(test_multicut test_multicut)

if(WITH_GUROBI)
    target_link_libraries(test_multicut ${GUROBI_LIBRARIES})
endif(WITH_GUROBI)

if(WITH_CPLEX)
    target_link_libraries(test_multicut ${CPLEX_LIBRARIES})
endif(WITH_CPLEX)
