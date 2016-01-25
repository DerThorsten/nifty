add_executable(test_undirected_graph test_undirected_graph.cxx )
target_link_libraries(test_undirected_graph ${TEST_LIBS})
add_test(test_undirected_graph test_undirected_graph)


add_executable(test_shortest_path test_shortest_path.cxx )
target_link_libraries(test_shortest_path ${TEST_LIBS})
add_test(test_shortest_path test_shortest_path)


add_executable(test_breadth_first_search test_breadth_first_search.cxx )
target_link_libraries(test_breadth_first_search ${TEST_LIBS})
add_test(test_breadth_first_search test_breadth_first_search)


add_executable(test_depth_first_search test_depth_first_search.cxx )
target_link_libraries(test_depth_first_search ${TEST_LIBS})
add_test(test_depth_first_search test_depth_first_search)
