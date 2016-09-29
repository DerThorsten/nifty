
#-------------------------------------------------------------------------------------------------------------------
# examples
#-------------------------------------------------------------------------------------------------------------------
enable_testing()
macro(add_autorun_example target_name target_file)
    add_executable(${target_name} ${target_file} )
    target_link_libraries(${target_name} ${ARGV2})
    add_test(example_${target_name} ${target_name})
endmacro()


include_directories( ${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(graph)


