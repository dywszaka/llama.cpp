# Optional C100 simulator runtime bridge for llama.cpp.
#
# This intentionally avoids add_subdirectory(C100_SIM_ROOT): the simulator
# project assumes CMAKE_SOURCE_DIR is its repository root and uses generic target
# names that would collide with llama.cpp targets.

if (NOT LLAMA_C100_RUNTIME)
    return()
endif()

set(C100_DOCKER_RISCV_TOOLCHAIN_DIR "/opt/riscv/master-v20251230")
set(C100_RISCV_TOOLCHAIN_DIR_DEFAULT "")
foreach(env_name RISCV_PATH RISCV RISCV_TOOLCHAIN)
    if (NOT C100_RISCV_TOOLCHAIN_DIR_DEFAULT AND NOT "$ENV{${env_name}}" STREQUAL "")
        set(C100_RISCV_TOOLCHAIN_DIR_DEFAULT "$ENV{${env_name}}")
    endif()
endforeach()
if (NOT C100_RISCV_TOOLCHAIN_DIR_DEFAULT)
    set(C100_RISCV_TOOLCHAIN_DIR_DEFAULT "${C100_DOCKER_RISCV_TOOLCHAIN_DIR}")
endif()

set(C100_RISCV_TOOLCHAIN_DIR "${C100_RISCV_TOOLCHAIN_DIR_DEFAULT}" CACHE PATH
    "RISC-V bare-metal toolchain root for C100 firmware, for example /opt/riscv/master-v20251230")
set(C100_RISCV_PREFIX "$ENV{RISCV_PREFIX}" CACHE STRING
    "RISC-V bare-metal tool prefix for C100 firmware, for example /opt/riscv/master-v20251230/bin/riscv64-unknown-elf-")

find_program(C100_DTC_EXECUTABLE NAMES dtc DOC "Device tree compiler used by C100 Spike configure")
if (NOT C100_DTC_EXECUTABLE)
    message(FATAL_ERROR
        "LLAMA_C100_RUNTIME requires dtc from the device-tree-compiler package before configuring the C100 Spike dependency. "
        "Install device-tree-compiler or use the C100 CI Docker image (${C100_SIM_ROOT}/ci/Dockerfile).")
endif()

find_program(C100_MAKE_EXECUTABLE NAMES gmake make DOC "make executable used by C100 Spike and firmware builds")
if (NOT C100_MAKE_EXECUTABLE)
    message(FATAL_ERROR
        "LLAMA_C100_RUNTIME requires make to build the C100 Spike and firmware dependencies. "
        "Install make or use the C100 CI Docker image (${C100_SIM_ROOT}/ci/Dockerfile).")
endif()

if (NOT C100_SIM_ROOT)
    message(FATAL_ERROR "LLAMA_C100_RUNTIME requires C100_SIM_ROOT")
endif()

get_filename_component(C100_SIM_ROOT "${C100_SIM_ROOT}" ABSOLUTE)
if (NOT EXISTS "${C100_SIM_ROOT}/src/top/llama_cpp.cpp")
    message(FATAL_ERROR "C100_SIM_ROOT does not look like a C100 simulator tree: ${C100_SIM_ROOT}")
endif()

set(C100_SIM_SRC_DIR "${C100_SIM_ROOT}/src")
set(C100_SIM_EXT_DIR "${C100_SIM_ROOT}/ext/riscv-isa-sim-lib-demo/src/top")
set(C100_SIM_SPIKE_ROOT "${C100_SIM_ROOT}/ext/riscv-isa-sim-lib-demo/riscv-isa-sim")
set(C100_SIM_SPIKE_BUILD_DIR "${CMAKE_BINARY_DIR}/c100-spike")
set(C100_SIM_SPIKE_INSTALL_DIR "${CMAKE_BINARY_DIR}/c100-spike-install")
set(C100_SIM_FIRMWARE_BUILD_DIR "${CMAKE_BINARY_DIR}/c100-firmware")
set(C100_RUNTIME_FIRMWARE_DIR "${CMAKE_BINARY_DIR}/firmware/llama.cpp")
set(C100_SIM_SPIKE_CONFIGURE_INPUT "${CMAKE_BINARY_DIR}/c100-spike-configure-input.txt")

file(MAKE_DIRECTORY
    "${C100_SIM_SPIKE_BUILD_DIR}"
    "${C100_SIM_SPIKE_INSTALL_DIR}"
    "${C100_SIM_FIRMWARE_BUILD_DIR}"
    "${C100_RUNTIME_FIRMWARE_DIR}")

if (NOT EXISTS "${C100_SIM_SPIKE_ROOT}/configure")
    message(FATAL_ERROR "C100 Spike source tree is missing configure script: ${C100_SIM_SPIKE_ROOT}")
endif()

if (C100_RISCV_PREFIX)
    set(C100_RESOLVED_RISCV_PREFIX "${C100_RISCV_PREFIX}")
elseif (C100_RISCV_TOOLCHAIN_DIR)
    set(C100_RESOLVED_RISCV_PREFIX "${C100_RISCV_TOOLCHAIN_DIR}/bin/riscv64-unknown-elf-")
else()
    message(FATAL_ERROR
        "LLAMA_C100_RUNTIME requires a RISC-V bare-metal toolchain prefix. "
        "Set -DC100_RISCV_TOOLCHAIN_DIR=${C100_DOCKER_RISCV_TOOLCHAIN_DIR} or "
        "-DC100_RISCV_PREFIX=/path/to/bin/riscv64-unknown-elf-.")
endif()

if (C100_RISCV_TOOLCHAIN_DIR)
    get_filename_component(C100_RESOLVED_RISCV_TOOLCHAIN_DIR "${C100_RISCV_TOOLCHAIN_DIR}" ABSOLUTE)
elseif (IS_ABSOLUTE "${C100_RESOLVED_RISCV_PREFIX}")
    get_filename_component(C100_RISCV_PREFIX_BIN_DIR "${C100_RESOLVED_RISCV_PREFIX}" DIRECTORY)
    get_filename_component(C100_RESOLVED_RISCV_TOOLCHAIN_DIR "${C100_RISCV_PREFIX_BIN_DIR}/.." ABSOLUTE)
else()
    set(C100_RESOLVED_RISCV_TOOLCHAIN_DIR "")
endif()

function(c100_check_program_runs program description)
    execute_process(
        COMMAND "${program}" --version
        RESULT_VARIABLE c100_program_result
        OUTPUT_QUIET
        ERROR_QUIET)
    if (NOT c100_program_result EQUAL 0)
        message(FATAL_ERROR
            "LLAMA_C100_RUNTIME found ${description} at ${program}, but it did not run successfully with --version.")
    endif()
endfunction()

function(c100_find_riscv_tool out_var tool_name)
    set(candidate "${C100_RESOLVED_RISCV_PREFIX}${tool_name}")
    if (IS_ABSOLUTE "${candidate}")
        if (NOT EXISTS "${candidate}")
            message(FATAL_ERROR
                "LLAMA_C100_RUNTIME requires the RISC-V bare-metal tool ${candidate}. "
                "Set -DC100_RISCV_TOOLCHAIN_DIR=${C100_DOCKER_RISCV_TOOLCHAIN_DIR} or "
                "-DC100_RISCV_PREFIX=/path/to/bin/riscv64-unknown-elf-. The C100 CI Docker image installs "
                "ci/toolchains/riscv-toolchain-master-v20251230.tar.gz to ${C100_DOCKER_RISCV_TOOLCHAIN_DIR} "
                "and exports RISCV, RISCV_PATH, RISCV_TOOLCHAIN, and RISCV_PREFIX.")
        endif()
        set(${out_var} "${candidate}" PARENT_SCOPE)
    else()
        unset(found_tool CACHE)
        find_program(found_tool NAMES "${candidate}")
        if (NOT found_tool)
            message(FATAL_ERROR
                "LLAMA_C100_RUNTIME requires the RISC-V bare-metal tool ${candidate} on PATH. "
                "Set -DC100_RISCV_TOOLCHAIN_DIR=${C100_DOCKER_RISCV_TOOLCHAIN_DIR} or "
                "-DC100_RISCV_PREFIX=/path/to/bin/riscv64-unknown-elf-. The C100 CI Docker image installs "
                "ci/toolchains/riscv-toolchain-master-v20251230.tar.gz to ${C100_DOCKER_RISCV_TOOLCHAIN_DIR} "
                "and exports RISCV, RISCV_PATH, RISCV_TOOLCHAIN, and RISCV_PREFIX.")
        endif()
        set(${out_var} "${found_tool}" PARENT_SCOPE)
    endif()
endfunction()

c100_check_program_runs("${C100_DTC_EXECUTABLE}" "dtc")
c100_check_program_runs("${C100_MAKE_EXECUTABLE}" "make")
c100_find_riscv_tool(C100_RISCV_GCC_EXECUTABLE gcc)
c100_find_riscv_tool(C100_RISCV_AR_EXECUTABLE ar)
c100_find_riscv_tool(C100_RISCV_OBJCOPY_EXECUTABLE objcopy)
c100_find_riscv_tool(C100_RISCV_OBJDUMP_EXECUTABLE objdump)
c100_find_riscv_tool(C100_RISCV_SIZE_EXECUTABLE size)
c100_check_program_runs("${C100_RISCV_GCC_EXECUTABLE}" "RISC-V gcc")
c100_check_program_runs("${C100_RISCV_AR_EXECUTABLE}" "RISC-V ar")
c100_check_program_runs("${C100_RISCV_OBJCOPY_EXECUTABLE}" "RISC-V objcopy")
c100_check_program_runs("${C100_RISCV_OBJDUMP_EXECUTABLE}" "RISC-V objdump")
c100_check_program_runs("${C100_RISCV_SIZE_EXECUTABLE}" "RISC-V size")

if (NOT EXISTS "${C100_SIM_ROOT}/scripts/elf2hex-split")
    message(FATAL_ERROR
        "LLAMA_C100_RUNTIME requires the C100 firmware helper script: ${C100_SIM_ROOT}/scripts/elf2hex-split")
endif()

set(C100_FIRMWARE_TOOLCHAIN_DIR "${CMAKE_BINARY_DIR}/c100-riscv-toolchain")
set(C100_FIRMWARE_TOOLCHAIN_BIN_DIR "${C100_FIRMWARE_TOOLCHAIN_DIR}/bin")
file(MAKE_DIRECTORY "${C100_FIRMWARE_TOOLCHAIN_BIN_DIR}")

foreach(c100_riscv_tool gcc ar objcopy objdump size as ld)
    set(c100_riscv_tool_link "${C100_FIRMWARE_TOOLCHAIN_BIN_DIR}/riscv64-unknown-elf-${c100_riscv_tool}")
    set(c100_riscv_tool_target "${C100_RESOLVED_RISCV_PREFIX}${c100_riscv_tool}")
    if (EXISTS "${c100_riscv_tool_link}" OR IS_SYMLINK "${c100_riscv_tool_link}")
        file(REMOVE "${c100_riscv_tool_link}")
    endif()
    file(CREATE_LINK "${c100_riscv_tool_target}" "${c100_riscv_tool_link}" SYMBOLIC)
endforeach()

set(C100_FIRMWARE_BIN2HEX "${C100_FIRMWARE_TOOLCHAIN_BIN_DIR}/riscv64-unknown-elf-bin2hex")
file(WRITE "${C100_FIRMWARE_BIN2HEX}" [=[
#!/usr/bin/env python3
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("-w", "--width", type=int, required=True)
parser.add_argument("input")
args = parser.parse_args()

if args.width <= 0 or args.width % 8 != 0:
    sys.stderr.write("bin2hex width must be a positive multiple of 8\n")
    sys.exit(1)

step = args.width // 8
with open(args.input, "rb") as f:
    while True:
        chunk = f.read(step)
        if not chunk:
            break
        sys.stdout.write(chunk[::-1].hex() + "\n")
]=])
file(CHMOD "${C100_FIRMWARE_BIN2HEX}"
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)

set(C100_FIRMWARE_ENV
    "RISCV=${C100_FIRMWARE_TOOLCHAIN_DIR}"
    "RISCV_PATH=${C100_FIRMWARE_TOOLCHAIN_DIR}"
    "RISCV_TOOLCHAIN=${C100_FIRMWARE_TOOLCHAIN_DIR}"
    "RISCV_PREFIX=${C100_FIRMWARE_TOOLCHAIN_BIN_DIR}/riscv64-unknown-elf-")

set(C100_SIM_COMMON_INCLUDES
    "${C100_SIM_SRC_DIR}"
    "${C100_SIM_SRC_DIR}/barrier"
    "${C100_SIM_SRC_DIR}/common"
    "${C100_SIM_SRC_DIR}/semaphore"
    "${C100_SIM_EXT_DIR}/runtime"
    "${C100_SIM_EXT_DIR}/extensions"
    "${C100_SIM_EXT_DIR}/common"
    "${C100_SIM_SPIKE_ROOT}"
    "${C100_SIM_SPIKE_ROOT}/riscv"
    "${C100_SIM_SPIKE_ROOT}/fesvr"
    "${C100_SIM_SPIKE_ROOT}/disasm"
    "${C100_SIM_SPIKE_ROOT}/softfloat"
    "${C100_SIM_SPIKE_BUILD_DIR}"
    "${C100_SIM_ROOT}/ext/spdlog/include")

set(C100_SIM_SPIKE_LIBS
    "${C100_SIM_SPIKE_BUILD_DIR}/libriscv.a"
    "${C100_SIM_SPIKE_BUILD_DIR}/libsoftfloat.a"
    "${C100_SIM_SPIKE_BUILD_DIR}/libdisasm.a"
    "${C100_SIM_SPIKE_BUILD_DIR}/libfesvr.a"
    "${C100_SIM_SPIKE_BUILD_DIR}/libspike_main.a"
    "${C100_SIM_SPIKE_BUILD_DIR}/libfdt.a")

configure_file("${CMAKE_CURRENT_LIST_DIR}/c100-spike-configure-input.in"
    "${C100_SIM_SPIKE_CONFIGURE_INPUT}" @ONLY)

add_custom_command(
    OUTPUT "${C100_SIM_SPIKE_BUILD_DIR}/c100-spike-configured.stamp"
    COMMAND "${CMAKE_COMMAND}" -E remove_directory "${C100_SIM_SPIKE_BUILD_DIR}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${C100_SIM_SPIKE_BUILD_DIR}" "${C100_SIM_SPIKE_INSTALL_DIR}"
    COMMAND "${CMAKE_COMMAND}" -E chdir "${C100_SIM_SPIKE_BUILD_DIR}"
            "${CMAKE_COMMAND}" -E env
            "CC=${CMAKE_C_COMPILER}"
            "CXX=${CMAKE_CXX_COMPILER}"
            "CFLAGS=-fPIC -O2"
            "CXXFLAGS=-fPIC -O2 -std=c++17"
            "${C100_SIM_SPIKE_ROOT}/configure"
            "--prefix=${C100_SIM_SPIKE_INSTALL_DIR}"
            "--with-boost=no"
            "--with-boost-asio=no"
            "--with-boost-regex=no"
    COMMAND "${CMAKE_COMMAND}" -E touch "${C100_SIM_SPIKE_BUILD_DIR}/c100-spike-configured.stamp"
    DEPENDS "${C100_SIM_SPIKE_CONFIGURE_INPUT}"
    COMMENT "Configuring C100 Spike simulator dependency")

add_custom_command(
    OUTPUT ${C100_SIM_SPIKE_LIBS}
    COMMAND "${C100_MAKE_EXECUTABLE}" -j8
    WORKING_DIRECTORY "${C100_SIM_SPIKE_BUILD_DIR}"
    DEPENDS "${C100_SIM_SPIKE_BUILD_DIR}/c100-spike-configured.stamp"
    COMMENT "Building C100 Spike simulator dependency")

add_custom_target(c100_sim_spike_lib DEPENDS ${C100_SIM_SPIKE_LIBS})

function(c100_sim_target_defaults target)
    target_include_directories(${target} PUBLIC ${C100_SIM_COMMON_INCLUDES})
    target_compile_features(${target} PUBLIC cxx_std_17)
    target_compile_definitions(${target} PUBLIC SPDLOG_HEADER_ONLY)
    set_target_properties(${target} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    add_dependencies(${target} c100_sim_spike_lib)
endfunction()

set(C100_SIM_CUSTOM_RISCV_SOURCES
    "${C100_SIM_EXT_DIR}/custom/riscv/custom_expp.cpp"
    "${C100_SIM_EXT_DIR}/custom/riscv/MxFp8ActQuant.cpp"
    "${C100_SIM_EXT_DIR}/custom/riscv/QuantBF16_NvFp4.cpp"
    "${C100_SIM_EXT_DIR}/custom/riscv/BF16.cpp"
    "${C100_SIM_EXT_DIR}/custom/riscv/SoftmaxCore.cpp")

add_library(c100_sim_xperiv STATIC "${C100_SIM_EXT_DIR}/extensions/xperiv.cc")
add_library(c100_sim_xperia STATIC "${C100_SIM_EXT_DIR}/extensions/xperia.cc")
add_library(c100_sim_custom_csr STATIC "${C100_SIM_EXT_DIR}/extensions/custom_csr.cc")
add_library(c100_sim_nuclei_bf16 STATIC "${C100_SIM_EXT_DIR}/extensions/nuclei_bf16.cc")
add_library(c100_sim_mailbox STATIC "${C100_SIM_EXT_DIR}/extensions/mailbox.cc")
add_library(c100_sim_spike_wrapper STATIC
    "${C100_SIM_EXT_DIR}/runtime/spike_wrapper.cc"
    "${C100_SIM_EXT_DIR}/extensions/spike_memory.cc")

foreach(target
        c100_sim_xperiv
        c100_sim_xperia
        c100_sim_custom_csr
        c100_sim_nuclei_bf16
        c100_sim_mailbox
        c100_sim_spike_wrapper)
    c100_sim_target_defaults(${target})
endforeach()
target_link_libraries(c100_sim_spike_wrapper PRIVATE c100_sim_nuclei_bf16)

add_library(c100_sim_common STATIC
    "${C100_SIM_SRC_DIR}/common/bus_device.cpp"
    "${C100_SIM_SRC_DIR}/common/spike_adapter.cpp"
    "${C100_SIM_SRC_DIR}/common/simple_bus.cpp"
    "${C100_SIM_SRC_DIR}/common/spike_csr_device.cpp"
    "${C100_SIM_SRC_DIR}/common/spike_mmio_device.cpp"
    "${C100_SIM_SRC_DIR}/common/unified_memory_device.cpp"
    "${C100_SIM_SRC_DIR}/common/round_robin_arbiter.cpp"
    "${C100_SIM_SRC_DIR}/common/priority_arbiter.cpp"
    "${C100_SIM_SRC_DIR}/common/matrix_arbiter.cpp"
    "${C100_SIM_SRC_DIR}/common/logger.cpp"
    "${C100_SIM_SRC_DIR}/common/fifo.cpp"
    "${C100_SIM_SRC_DIR}/common/spdlog_wrapper.cpp")
c100_sim_target_defaults(c100_sim_common)
target_link_libraries(c100_sim_common PUBLIC fmt)

add_library(c100_sim_barrier STATIC
    "${C100_SIM_SRC_DIR}/barrier/barrier_model.cpp"
    "${C100_SIM_SRC_DIR}/barrier/event_queue.cpp")
c100_sim_target_defaults(c100_sim_barrier)

add_library(c100_sim_semaphore STATIC "${C100_SIM_SRC_DIR}/semaphore/sema_manager.cpp")
c100_sim_target_defaults(c100_sim_semaphore)

add_library(c100_sim_tensor_common STATIC
    "${C100_SIM_SRC_DIR}/tensor_common/util.cpp"
    "${C100_SIM_SRC_DIR}/tensor_common/tdma_core.cpp"
    "${C100_SIM_SRC_DIR}/tensor_common/tdma.cpp"
    "${C100_SIM_SRC_DIR}/tensor_common/transpose.cpp"
    "${C100_SIM_SRC_DIR}/tensor_common/add_tree.cpp"
    "${C100_SIM_SRC_DIR}/tensor_common/acc_buffer.cpp")
c100_sim_target_defaults(c100_sim_tensor_common)
target_link_libraries(c100_sim_tensor_common PUBLIC c100_sim_common)

add_library(c100_sim_dma STATIC "${C100_SIM_SRC_DIR}/dma/su_dma_out.cpp")
c100_sim_target_defaults(c100_sim_dma)

add_library(c100_sim_m2s STATIC "${C100_SIM_SRC_DIR}/m2s/m2s_dma.cpp")
c100_sim_target_defaults(c100_sim_m2s)
target_link_libraries(c100_sim_m2s PUBLIC c100_sim_tensor_common c100_sim_common)

add_library(c100_sim_s2m STATIC "${C100_SIM_SRC_DIR}/s2m/s2m_dma.cpp")
c100_sim_target_defaults(c100_sim_s2m)
target_link_libraries(c100_sim_s2m PUBLIC c100_sim_tensor_common c100_sim_common)

add_library(c100_sim_ring_buffer STATIC
    "${C100_SIM_SRC_DIR}/ring_buffer/channel.cpp"
    "${C100_SIM_SRC_DIR}/ring_buffer/ring_buffer.cpp"
    "${C100_SIM_SRC_DIR}/ring_buffer/sram.cpp")
c100_sim_target_defaults(c100_sim_ring_buffer)

add_library(c100_sim_copy_engine STATIC "${C100_SIM_SRC_DIR}/copy_engine/copy_engine_top.cpp")
c100_sim_target_defaults(c100_sim_copy_engine)
target_link_libraries(c100_sim_copy_engine PUBLIC c100_sim_tensor_common c100_sim_common)

add_library(c100_sim_gemm STATIC
    "${C100_SIM_SRC_DIR}/gemm/gemm_top.cpp"
    "${C100_SIM_SRC_DIR}/gemm/gemm_core.cpp"
    "${C100_SIM_SRC_DIR}/gemm/gemm_dma_engine.cpp"
    "${C100_SIM_SRC_DIR}/gemm/gemm_shape.cpp"
    "${C100_SIM_SRC_DIR}/gemm/gemm_format.cpp"
    "${C100_SIM_SRC_DIR}/gemm/gemm_stream_layout.cpp"
    "${C100_SIM_SRC_DIR}/gemm/gemm_result_layout.cpp"
    "${C100_SIM_SRC_DIR}/gemm/gemm_nvfp4_unit.cpp"
    "${C100_SIM_SRC_DIR}/gemm/gemm_nvfp4_engine.cpp"
    "${C100_SIM_SRC_DIR}/gemm/gemm_mxfp8_engine.cpp")
c100_sim_target_defaults(c100_sim_gemm)
target_link_libraries(c100_sim_gemm PUBLIC c100_sim_tensor_common c100_sim_common)

add_library(c100_sim_tools STATIC "${C100_SIM_SRC_DIR}/tools/memory_image_manager.cpp")
c100_sim_target_defaults(c100_sim_tools)
target_link_libraries(c100_sim_tools PUBLIC c100_sim_common)

add_library(c100_sim_su INTERFACE)
target_include_directories(c100_sim_su INTERFACE ${C100_SIM_COMMON_INCLUDES})
target_compile_definitions(c100_sim_su INTERFACE SPDLOG_HEADER_ONLY)
target_link_libraries(c100_sim_su INTERFACE c100_sim_common)

add_library(c100_sim_ve INTERFACE)
target_include_directories(c100_sim_ve INTERFACE ${C100_SIM_COMMON_INCLUDES})
target_compile_definitions(c100_sim_ve INTERFACE SPDLOG_HEADER_ONLY)
target_link_libraries(c100_sim_ve INTERFACE c100_sim_common)

add_library(c100_sim_ctrl_cpu INTERFACE)
target_include_directories(c100_sim_ctrl_cpu INTERFACE ${C100_SIM_COMMON_INCLUDES})
target_compile_definitions(c100_sim_ctrl_cpu INTERFACE SPDLOG_HEADER_ONLY)
target_link_libraries(c100_sim_ctrl_cpu INTERFACE c100_sim_common)

add_library(c100_sim_simulator STATIC
    "${C100_SIM_SRC_DIR}/top/c100_address_router.cpp"
    "${C100_SIM_SRC_DIR}/top/chip.cpp"
    "${C100_SIM_SRC_DIR}/top/doorbell_int.cpp"
    "${C100_SIM_SRC_DIR}/top/hcu.cpp"
    "${C100_SIM_SRC_DIR}/top/m2s_dma_bus_device.cpp"
    "${C100_SIM_SRC_DIR}/top/pcie_top.cpp"
    "${C100_SIM_SRC_DIR}/top/ring_buffer_bus_device.cpp"
    "${C100_SIM_SRC_DIR}/top/chip_noc.cpp")
c100_sim_target_defaults(c100_sim_simulator)
target_include_directories(c100_sim_simulator PUBLIC
    "${C100_SIM_SRC_DIR}/top"
    "${C100_SIM_ROOT}/firmware/llama.cpp/common")
target_link_libraries(c100_sim_simulator PUBLIC
    c100_sim_common
    c100_sim_barrier
    c100_sim_semaphore
    c100_sim_m2s
    c100_sim_s2m
    c100_sim_dma
    c100_sim_ring_buffer
    c100_sim_su
    c100_sim_ve
    c100_sim_ctrl_cpu
    c100_sim_copy_engine
    c100_sim_gemm
    c100_sim_tensor_common
    c100_sim_tools)

add_library(c100_runtime STATIC
    "${C100_SIM_SRC_DIR}/top/llama_cpp.cpp"
    ${C100_SIM_CUSTOM_RISCV_SOURCES})
c100_sim_target_defaults(c100_runtime)
target_compile_definitions(c100_runtime PRIVATE
    C100_LLAMA_FIRMWARE_DIR="${C100_RUNTIME_FIRMWARE_DIR}")
target_include_directories(c100_runtime PUBLIC
    "${C100_SIM_SRC_DIR}/top"
    "${C100_SIM_ROOT}/firmware/llama.cpp/common"
    "${CMAKE_CURRENT_SOURCE_DIR}/ggml/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/ggml/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/ggml/src/ggml-c100")
target_link_libraries(c100_runtime PUBLIC
    c100_sim_simulator
    c100_sim_su
    c100_sim_ve
    c100_sim_common
    c100_sim_spike_wrapper
    c100_sim_custom_csr
    c100_sim_mailbox
    "$<LINK_LIBRARY:WHOLE_ARCHIVE,c100_sim_xperia,c100_sim_xperiv>"
    ${C100_SIM_SPIKE_LIBS}
    pthread
    dl
    z
    boost_regex
    boost_system)

function(c100_add_firmware_target name cpu_type)
    set(src_dir "${C100_SIM_ROOT}/firmware/llama.cpp/${cpu_type}")
    set(build_dir "${C100_SIM_FIRMWARE_BUILD_DIR}/llama.cpp/${cpu_type}")
    set(out_elf "${build_dir}/firmware.elf")
    set(copied_elf "${C100_RUNTIME_FIRMWARE_DIR}/${cpu_type}.elf")
    file(GLOB_RECURSE firmware_deps CONFIGURE_DEPENDS
        "${src_dir}/*"
        "${C100_SIM_ROOT}/firmware/llama.cpp/common/*"
        "${C100_SIM_ROOT}/firmware/llama.cpp/operators/*"
        "${C100_SIM_ROOT}/firmware/common/*")

    add_custom_command(
        OUTPUT "${copied_elf}"
        COMMAND "${CMAKE_COMMAND}" -E env ${C100_FIRMWARE_ENV}
                "${C100_MAKE_EXECUTABLE}"
                "BUILD_DIR=${build_dir}"
                "RISCV=${C100_FIRMWARE_TOOLCHAIN_DIR}"
                "RISCV_PATH=${C100_FIRMWARE_TOOLCHAIN_DIR}"
                "RISCV_TOOLCHAIN=${C100_FIRMWARE_TOOLCHAIN_DIR}"
                "RISCV_PREFIX=${C100_FIRMWARE_TOOLCHAIN_BIN_DIR}/riscv64-unknown-elf-"
                all
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${C100_RUNTIME_FIRMWARE_DIR}"
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${out_elf}" "${copied_elf}"
        DEPENDS ${firmware_deps}
        WORKING_DIRECTORY "${src_dir}"
        COMMENT "Building C100 ${name} firmware")
endfunction()

c100_add_firmware_target(SU su)
c100_add_firmware_target(VE ve)

add_custom_target(c100_firmware
    DEPENDS
        "${C100_RUNTIME_FIRMWARE_DIR}/su.elf"
        "${C100_RUNTIME_FIRMWARE_DIR}/ve.elf")

add_dependencies(c100_runtime c100_firmware)

message(STATUS "C100 runtime integration enabled")
message(STATUS "  C100_SIM_ROOT: ${C100_SIM_ROOT}")
message(STATUS "  C100_RISCV_PREFIX: ${C100_RESOLVED_RISCV_PREFIX}")
