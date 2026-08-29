# ShaderToolchain.cmake
#
# Locates the GLSL -> SPIR-V compiler (glslc) and the SPIR-V validator
# (spirv-val) and exposes a helper for compiling shaders at build time.
#
# glslc is provided by the Vulkan SDK as an IMPORTED executable target
# `Vulkan::glslc` (via `find_package(Vulkan COMPONENTS glslc)`). If that target
# is not available we fall back to `find_program`. `spirv-val` has no CMake
# FindVulkan component, so it is always located via `find_program`.
#
# No shaders are compiled during M0 (scaffold). Kernels land in M1+, where
# callers will do:
#
#     add_shader(vulkan_runtime_shaders my_kernel.comp)
#
# which produces `${binary_dir}/shaders/my_kernel.spv`, validates it with
# spirv-val, and adds the output to the `vulkan_runtime_shaders` INTERFACE
# target so it can be embedded (via the `embed_shaders` helper below).

if(NOT TARGET Vulkan::glslc)
    find_program(VULKAN_GLSL_COMPILER NAMES glslc)
    if(NOT VULKAN_GLSL_COMPILER)
        message(WARNING "glslc not found: Vulkan::glslc target absent and 'glslc' not on PATH. "
                        "Shader compilation will be disabled (M0 does not need it).")
    endif()
endif()

find_program(VULKAN_SPIRV_VALIDATOR NAMES spirv-val)

# Returns the glslc invocation as a CMake command string in `out_glslc`.
function(_vulkan_runtime_glslc_command out_glslc)
    if(TARGET Vulkan::glslc)
        set("${out_glslc}" "$<TARGET_FILE:Vulkan::glslc>" PARENT_SCOPE)
    else()
        set("${out_glslc}" "${VULKAN_GLSL_COMPILER}" PARENT_SCOPE)
    endif()
endfunction()

# add_shader(<target> <source.glsl>)
#
# Compiles a GLSL source to SPIR-V with `-O` (mandatory: omitting -O costs
# ~1000x on GCN), validates the result with spirv-val, and appends the output
# .spv to the INTERFACE target <target>.
function(add_shader target source)
    if(NOT TARGET Vulkan::glslc AND NOT VULKAN_GLSL_COMPILER)
        message(FATAL_ERROR "add_shader('${source}'): glslc is required but not found.")
    endif()

    # The target must exist before add_dependencies below can reference it.
    if(NOT TARGET "${target}")
        add_library("${target}" INTERFACE)
    endif()

    get_filename_component(shader_name "${source}" NAME_WE)
    set(spv_output "${CMAKE_CURRENT_BINARY_DIR}/shaders/${shader_name}.spv")

    _vulkan_runtime_glslc_command(glslc_cmd)
    add_custom_command(
        OUTPUT "${spv_output}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/shaders"
        COMMAND ${glslc_cmd} -O -fshader-stage=compute
                -I "${CMAKE_CURRENT_SOURCE_DIR}/shaders"
                -o "${spv_output}"
                "${CMAKE_CURRENT_SOURCE_DIR}/${source}"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${source}"
        COMMENT "glslc: ${source} -> ${shader_name}.spv"
        VERBATIM
    )

    # Validate the produced SPIR-V at build time (spirv-val may be optional;
    # warn loudly but don't hard-fail the scaffold if it's missing).
    if(VULKAN_SPIRV_VALIDATOR)
        add_custom_command(
            OUTPUT "${spv_output}.validated"
            COMMAND "${VULKAN_SPIRV_VALIDATOR}" "${spv_output}"
            COMMAND "${CMAKE_COMMAND}" -E touch "${spv_output}.validated"
            DEPENDS "${spv_output}"
            COMMENT "spirv-val: ${shader_name}.spv"
            VERBATIM
        )
        add_custom_target("${shader_name}_validate" DEPENDS "${spv_output}.validated")
        add_dependencies("${target}" "${shader_name}_validate")
    endif()

    target_sources("${target}" INTERFACE "${spv_output}")
endfunction()

# embed_shaders(<target>)
#
# Generates a C++ header (embedded_shaders.hpp) at build time that embeds every
# .spv file attached to <target> as a byte array. Consumers include the header
# and call `vulkan_runtime::shaders::get("<name>.spv")` to fetch a blob.
#
# The header is produced by a CMake script (embed_shaders_impl.cmake) that hex-
# encodes the SPIR-V at build time — after glslc has produced it. The generated
# directory is added to <target>'s INTERFACE include dirs, and a build-order
# dependency on the generation step is propagated so consumers never compile
# against a stale/missing header.
function(embed_shaders target)
    if(NOT TARGET "${target}")
        message(FATAL_ERROR "embed_shaders('${target}'): target does not exist.")
    endif()

    get_target_property(shader_sources "${target}" INTERFACE_SOURCES)
    if(NOT shader_sources)
        message(WARNING "embed_shaders('${target}'): no shaders attached; nothing to embed.")
        return()
    endif()

    set(gen_dir "${CMAKE_CURRENT_BINARY_DIR}/generated")
    set(gen_hpp "${gen_dir}/embedded_shaders.hpp")

    # Pair each .spv with its basename (the registry key).
    set(shader_names "")
    foreach(spv IN LISTS shader_sources)
        get_filename_component(spv_base "${spv}" NAME)
        list(APPEND shader_names "${spv_base}")
    endforeach()

    add_custom_command(
        OUTPUT "${gen_hpp}"
        COMMAND "${CMAKE_COMMAND}"
                "-DOUTPUT=${gen_hpp}"
                "-DSHADER_NAMES=${shader_names}"
                "-DSHADER_FILES=${shader_sources}"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_shaders_impl.cmake"
        DEPENDS ${shader_sources}
        COMMENT "embed_shaders: generating embedded_shaders.hpp"
        VERBATIM
    )

    add_custom_target("${target}_embed" DEPENDS "${gen_hpp}")

    target_include_directories("${target}" INTERFACE "${gen_dir}")
    target_sources("${target}" INTERFACE "${gen_hpp}")
    # Build-order dependency so consumers compile after the header is generated.
    # (A custom target must NOT go into INTERFACE_LINK_LIBRARIES — that would
    # make the linker look for -l<target>.) add_dependencies on an INTERFACE
    # library propagates transitively to its consumers since CMake 3.19.
    add_dependencies("${target}" "${target}_embed")
endfunction()
