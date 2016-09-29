



# Create the binding library
pybind11_add_module(_gala 
   gala.cxx
   gala_main_class.cxx
   gala_feature_base.cxx
)



# Don't add a 'lib' prefix to the shared library
set_target_properties(_gala PROPERTIES PREFIX "")

# Link against the Python shared library
target_link_libraries(_gala ${PYTHON_LIBRARIES})




file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/python/nifty/graph/gala)

ADD_CUSTOM_COMMAND(
        TARGET _gala
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} ARGS -E copy_if_different $<TARGET_FILE:_gala> ${CMAKE_BINARY_DIR}/python/nifty/graph/gala
        COMMENT "Copying target gala to temporary module directory")




add_custom_target(copy_pythongala__init__ ALL#POST_BUILD
    COMMAND cmake -E copy_if_different 
    ${CMAKE_SOURCE_DIR}/src/python/module/graph/gala/__init__.py 
    ${CMAKE_BINARY_DIR}/python/nifty/graph/gala/__init__.py)



