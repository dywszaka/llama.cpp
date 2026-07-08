# C100 GGML Backend CMakeLists.txt

ggml_add_backend_library(ggml-c100
    ggml-c100.c
)

target_include_directories(ggml-c100 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# Define GGML version macros
target_compile_definitions(ggml-c100 PRIVATE
    GGML_VERSION="0.9.8"
    GGML_COMMIT="c100-integration"
)

if (TARGET c100_runtime)
    target_link_libraries(ggml-c100 PRIVATE $<LINK_LIBRARY:WHOLE_ARCHIVE,c100_runtime>)
endif()

message(STATUS "C100 backend enabled")
