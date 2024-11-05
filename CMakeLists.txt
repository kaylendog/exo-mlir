cmake_minimum_required(VERSION 3.12)
project(exolang-mlir VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror -O3")

set(EXO_OUTPUT_DIR "${CMAKE_BINARY_DIR}/generated_exo_sources")
file(MAKE_DIRECTORY ${EXO_OUTPUT_DIR})

set(BENCHMARK_PYTHON_DIR "${CMAKE_SOURCE_DIR}/benchmarks")
file(GLOB BENCHMARK_PYTHON_FILES "${BENCHMARK_PYTHON_DIR}/*.py")

# add benchmark
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Disable testing of the benchmark library" FORCE)
add_subdirectory(${CMAKE_SOURCE_DIR}/submodules/benchmark)
include_directories(${CMAKE_SOURCE_DIR}/submodules/benchmark/include)

# invoke exocc
function(compile_exo input_py)
    get_filename_component(filename ${input_py} NAME_WE)
    set(stem ${filename})
    set(output_c "${EXO_OUTPUT_DIR}/${stem}.c")
    set(output_h "${EXO_OUTPUT_DIR}/${stem}.h")

    add_custom_command(
        OUTPUT ${output_c} ${output_h}
        COMMAND exocc -o ${EXO_OUTPUT_DIR} --stem ${stem} ${input_py}
        DEPENDS ${input_py}
        COMMENT "Compiling ${input_py} to ${output_c} and ${output_h}"
        VERBATIM
    )

    set_property(DIRECTORY APPEND PROPERTY EXO_C_FILES ${output_c})
endfunction()

# compile all benchmark python files
foreach(py_file ${BENCHMARK_PYTHON_FILES})
    compile_exo(${py_file})
endforeach()

add_executable(benchmarks ${CMAKE_SOURCE_DIR}/benchmarks/entrypoint.cpp)
target_link_libraries(benchmarks PRIVATE benchmark::benchmark)

# print c files at build time
add_custom_target(print_c_files
    COMMAND ${CMAKE_COMMAND} -E echo "C files: $<JOIN:${EXO_C_FILES}, >"
    DEPENDS ${EXO_C_FILES}
    VERBATIM
)

add_dependencies(benchmarks print_c_files)
