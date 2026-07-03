# C100 GGML Backend CMakeLists.txt

ggml_add_backend_library(ggml-c100
    ggml-c100.c
)

target_include_directories(ggml-c100 PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(ggml-c100 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# Define GGML version macros
target_compile_definitions(ggml-c100 PRIVATE
    GGML_VERSION="0.9.8"
    GGML_COMMIT="c100-integration"
)

if (LLAMA_C100_RUNTIME)
    if (NOT TARGET c100_runtime)
        message(FATAL_ERROR "LLAMA_C100_RUNTIME is ON but c100_runtime target is not available")
    endif()
    target_link_libraries(ggml-c100 PRIVATE
        "$<LINK_LIBRARY:WHOLE_ARCHIVE,c100_runtime>")
    if (TARGET ggml)
        target_link_libraries(ggml PUBLIC c100_runtime)
    endif()
endif()

message(STATUS "C100 backend enabled")
