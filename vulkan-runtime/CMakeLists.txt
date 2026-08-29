cmake_minimum_required(VERSION 3.25)

# ---------------------------------------------------------------------------
# vulkan-runtime — headless Vulkan compute runtime for AMD RX580 (RADV/gfx803)
#
# M0 = project scaffold: CMake skeleton, vendored VMA + Vulkan-Hpp, VVL in
# debug, lavapipe fallback, and a smoke target that proves instance + device +
# VMA allocator creation. No kernels, shaders, or exec engine yet (M1+).
# ---------------------------------------------------------------------------
project(vulkan-runtime LANGUAGES CXX)

# ---------------------------------------------------------------------------
# C++20
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# clangd: export compile_commands.json (the presets also set this).
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---------------------------------------------------------------------------
# Warnings. -Werror is opt-in via VR_WARNINGS_AS_ERRORS so the scaffold stays
# green even if a future toolchain emits a benign diagnostic we haven't seen.
# ---------------------------------------------------------------------------
option(VR_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
add_compile_options(-Wall -Wextra -Wpedantic)
if(VR_WARNINGS_AS_ERRORS)
    add_compile_options(-Werror)
endif()

# ---------------------------------------------------------------------------
# Vulkan headers + loader, and the glslc shader compiler.
#   find_package(Vulkan COMPONENTS glslc) yields:
#     Vulkan::Vulkan  (headers + libvulkan loader)
#     Vulkan::glslc   (IMPORTED executable, if glslc is on PATH)
# ---------------------------------------------------------------------------
find_package(Vulkan REQUIRED COMPONENTS glslc)

# spirv-val has no FindVulkan component; locate it directly.
find_program(VULKAN_SPIRV_VALIDATOR NAMES spirv-val)

# ---------------------------------------------------------------------------
# Vendored headers (preferred over system Vulkan-Hpp / VMA).
#
# Layout: third_party/vulkan-hpp/vulkan/*.hpp and third_party/vma/*.h, so
#   #include <vulkan/vulkan.hpp>  -> vendored v1.4.309
#   #include <vk_mem_alloc.h>     -> vendored v3.4.0
# The Vulkan C headers (vulkan.h / vulkan_core.h) still come from the SDK via
# Vulkan::Vulkan (i.e. /usr/include). This INTERFACE target is linked *before*
# Vulkan::Vulkan so the vendored headers win the include search.
# ---------------------------------------------------------------------------
add_library(vulkan_runtime_headers INTERFACE)
target_include_directories(vulkan_runtime_headers SYSTEM INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/vulkan-hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/vma"
)

# ---------------------------------------------------------------------------
# Shader toolchain helper (glslc -> SPIR-V -> spirv-val). M0 compiles no
# shaders; the helper is wired for M1+.
# ---------------------------------------------------------------------------
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(ShaderToolchain)

# ---------------------------------------------------------------------------
# Core library — shared headless-compute setup (instance / device / VMA), used
# by the smoke target and every M1+ entry point. VMA configuration macros live
# in src/core/vk_setup.h (single source of truth); VMA_IMPLEMENTATION is in
# src/core/vk_setup.cpp.
# ---------------------------------------------------------------------------
add_library(vulkan_runtime_core STATIC src/core/vk_setup.cpp)
set_target_properties(vulkan_runtime_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(vulkan_runtime_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(vulkan_runtime_core PUBLIC vulkan_runtime_headers Vulkan::Vulkan)

# C ABI vector_add integration target. The implementation reuses the canonical
# validated GPU path and returns the device-produced output buffer.
add_library(vulkan_runtime_api SHARED
    src/api/vector_add_api.cpp
    src/api/sgemm_api.cpp
    src/api/attention_api.cpp
    src/api/attention_causal_api.cpp
    src/api/lora_delta_api.cpp
    src/api/lora_gradients_api.cpp
    src/api/lora_sgd_api.cpp
    src/api/lora_train_api.cpp
    src/api/lora_session_api.cpp
    src/api/causal_loss_api.cpp
    src/api/attention_causal_backward_api.cpp
    src/api/lm_head_backward_api.cpp
    src/api/projection_backward_api.cpp
    src/api/lora_gradients_multi_api.cpp
    src/api/lora_sgd_multi_api.cpp
    src/api/tiny_forward_persistent.cpp
    src/api/tiny_immutable_prototype.cpp
    src/api/dataset_batch_buffer.cpp
    src/api/reduced_precision_storage.cpp
    src/api/tiny_base_training_api.cpp
    src/api/output_projection_training_api.cpp
    src/api/qkv_projection_training_api.cpp
    src/api/embedding_training_api.cpp
)
target_include_directories(vulkan_runtime_api PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(vulkan_runtime_api PUBLIC vulkan_runtime_core vulkan_runtime_shaders vulkan_runtime_headers Vulkan::Vulkan)

add_executable(attention_causal_api tests/test_attention_causal_api.cpp)
target_link_libraries(attention_causal_api PRIVATE vulkan_runtime_api)
add_executable(lora_delta_api tests/test_lora_delta_api.cpp)
target_link_libraries(lora_delta_api PRIVATE vulkan_runtime_api)
add_executable(lora_gradients_api tests/test_lora_gradients_api.cpp)
target_link_libraries(lora_gradients_api PRIVATE vulkan_runtime_api)
add_executable(lora_sgd_api tests/test_lora_sgd_api.cpp)
target_link_libraries(lora_sgd_api PRIVATE vulkan_runtime_api)
add_executable(lora_train_api tests/test_lora_train_api.cpp)
target_link_libraries(lora_train_api PRIVATE vulkan_runtime_api)
add_executable(lora_session_api tests/test_lora_session_api.cpp)
target_link_libraries(lora_session_api PRIVATE vulkan_runtime_api)
add_executable(causal_loss_api tests/test_causal_loss_api.cpp)
target_link_libraries(causal_loss_api PRIVATE vulkan_runtime_api)
add_executable(attention_causal_backward_api tests/test_attention_causal_backward_api.cpp)
target_link_libraries(attention_causal_backward_api PRIVATE vulkan_runtime_api)
add_executable(lm_head_backward_api tests/test_lm_head_backward_api.cpp)
target_link_libraries(lm_head_backward_api PRIVATE vulkan_runtime_api)
add_executable(tiny_base_training tests/test_tiny_base_training.cpp)
target_link_libraries(tiny_base_training PRIVATE vulkan_runtime_api)
add_executable(output_projection_training tests/test_output_projection_training.cpp)
target_link_libraries(output_projection_training PRIVATE vulkan_runtime_api)
add_executable(qkv_projection_training tests/test_qkv_projection_training.cpp)
target_link_libraries(qkv_projection_training PRIVATE vulkan_runtime_api)
add_executable(embedding_training tests/test_embedding_training.cpp)
target_link_libraries(embedding_training PRIVATE vulkan_runtime_api)
add_executable(embedding_training_rmsnorm_dx tests/test_embedding_training_rmsnorm_dx.cpp)
add_executable(position_training_rmsnorm_dx tests/test_position_training_rmsnorm_dx.cpp)
target_link_libraries(position_training_rmsnorm_dx PRIVATE vulkan_runtime_core vulkan_runtime_shaders vulkan_runtime_headers Vulkan::Vulkan)
add_dependencies(position_training_rmsnorm_dx vulkan_runtime_shaders_embed)
target_link_libraries(embedding_training_rmsnorm_dx PRIVATE vulkan_runtime_core vulkan_runtime_shaders vulkan_runtime_headers Vulkan::Vulkan)
add_dependencies(embedding_training_rmsnorm_dx vulkan_runtime_shaders_embed)
add_executable(projection_backward_api tests/test_projection_backward_api.cpp)
target_link_libraries(projection_backward_api PRIVATE vulkan_runtime_api)
add_executable(lora_gradients_multi_api tests/test_lora_gradients_multi_api.cpp)
target_link_libraries(lora_gradients_multi_api PRIVATE vulkan_runtime_api)
add_executable(lora_sgd_multi_api tests/test_lora_sgd_multi_api.cpp)
target_link_libraries(lora_sgd_multi_api PRIVATE vulkan_runtime_api)

# ---------------------------------------------------------------------------
# Shaders — compiled at build time (glslc -O) -> SPIR-V -> spirv-val, then
# embedded into a generated header consumed by kernel entry points.
# ---------------------------------------------------------------------------
add_shader(vulkan_runtime_shaders shaders/vector_add.comp)
add_shader(vulkan_runtime_shaders shaders/sgemm.comp)
add_shader(vulkan_runtime_shaders shaders/sgemm_bk48.comp)
add_shader(vulkan_runtime_shaders shaders/sgemm_vec4a.comp)
add_shader(vulkan_runtime_shaders shaders/sgemm_8x2.comp)
add_shader(vulkan_runtime_shaders shaders/transpose.comp)
add_shader(vulkan_runtime_shaders shaders/rmsnorm.comp)
add_shader(vulkan_runtime_shaders shaders/rmsnorm_backward.comp)
add_shader(vulkan_runtime_shaders shaders/rmsnorm_backward_direct.comp)
add_shader(vulkan_runtime_shaders shaders/rmsnorm_dgamma.comp)
add_shader(vulkan_runtime_shaders shaders/rmsnorm_backward_masked.comp)
add_shader(vulkan_runtime_shaders shaders/rmsnorm_backward_state.comp)
add_shader(vulkan_runtime_shaders shaders/rmsnorm_dgamma_masked.comp)
add_shader(vulkan_runtime_shaders shaders/rmsnorm_dgamma_state.comp)
add_shader(vulkan_runtime_shaders shaders/rmsnorm_forward_state.comp)
add_shader(vulkan_runtime_shaders shaders/rmsnorm_forward_fixed_profile.comp)
add_shader(vulkan_runtime_shaders shaders/scale_dgamma_masked.comp)

add_shader(vulkan_runtime_shaders shaders/rmsnorm_gamma_adamw.comp)
add_shader(vulkan_runtime_shaders shaders/ffn_gelu_forward.comp)
add_shader(vulkan_runtime_shaders shaders/lm_head_forward.comp)
add_shader(vulkan_runtime_shaders shaders/ffn_gelu_forward_unmasked.comp)
add_shader(vulkan_runtime_shaders shaders/ffn_gelu_backward.comp)
add_shader(vulkan_runtime_shaders shaders/ffn_param_grads.comp)
add_shader(vulkan_runtime_shaders shaders/ffn_param_grads_masked.comp)
add_shader(vulkan_runtime_shaders shaders/ffn_param_grads_rmsnorm.comp)
add_shader(vulkan_runtime_shaders shaders/ffn_adamw.comp)
add_shader(vulkan_runtime_shaders shaders/rope.comp)
add_shader(vulkan_runtime_shaders shaders/q4gemm.comp)
add_shader(vulkan_runtime_shaders shaders/attention.comp)
add_shader(vulkan_runtime_shaders shaders/attention_causal.comp)
add_shader(vulkan_runtime_shaders shaders/lora_delta.comp)
add_shader(vulkan_runtime_shaders shaders/lora_gradients.comp)
add_shader(vulkan_runtime_shaders shaders/lora_sgd.comp)
add_shader(vulkan_runtime_shaders shaders/causal_loss.comp)
add_shader(vulkan_runtime_shaders shaders/metrics_reduce.comp)
add_shader(vulkan_runtime_shaders shaders/lm_head_backward.comp)
add_shader(vulkan_runtime_shaders shaders/lm_head_weight_grad.comp)
add_shader(vulkan_runtime_shaders shaders/lm_head_sgd.comp)
add_shader(vulkan_runtime_shaders shaders/lm_head_adamw.comp)
add_shader(vulkan_runtime_shaders shaders/output_projection_weight_grad.comp)
add_shader(vulkan_runtime_shaders shaders/output_projection_sgd.comp)
add_shader(vulkan_runtime_shaders shaders/output_projection_adamw.comp)
add_shader(vulkan_runtime_shaders shaders/qkv_projection_grad.comp)
add_shader(vulkan_runtime_shaders shaders/qkv_projection_weight_grad.comp)
add_shader(vulkan_runtime_shaders shaders/qkv_projection_sgd.comp)
add_shader(vulkan_runtime_shaders shaders/qkv_projection_adamw.comp)
add_shader(vulkan_runtime_shaders shaders/embedding_training_grad.comp)
add_shader(vulkan_runtime_shaders shaders/embedding_training_grad_rmsnorm.comp)
add_shader(vulkan_runtime_shaders shaders/embedding_training_sgd.comp)
add_shader(vulkan_runtime_shaders shaders/position_training_grad.comp)
add_shader(vulkan_runtime_shaders shaders/position_training_grad_rmsnorm.comp)
add_shader(vulkan_runtime_shaders shaders/position_training_sgd.comp)
add_shader(vulkan_runtime_shaders shaders/position_training_adamw.comp)
add_shader(vulkan_runtime_shaders shaders/projection_backward.comp)
add_shader(vulkan_runtime_shaders shaders/attention_causal_backward.comp)
add_shader(vulkan_runtime_shaders shaders/lora_gradients_multi.comp)
add_shader(vulkan_runtime_shaders shaders/lora_gradients_persistent.comp)
add_shader(vulkan_runtime_shaders shaders/lora_sgd_multi.comp)
add_shader(vulkan_runtime_shaders shaders/lora_accumulate_clear.comp)
add_shader(vulkan_runtime_shaders shaders/lora_finalize_sgd_multi.comp)
add_shader(vulkan_runtime_shaders shaders/lora_adamw_multi.comp)
add_shader(vulkan_runtime_shaders shaders/tiny_forward_logits.comp)
add_shader(vulkan_runtime_shaders shaders/tiny_train_forward_rmsnorm.comp)
add_shader(vulkan_runtime_shaders shaders/tiny_immutable_prototype.comp)
add_shader(vulkan_runtime_shaders shaders/dataset_batch_buffer.comp)
# M6c-1: cactus GPU kernel library (fp16 I/O, fp32 compute, CQ4 decomposed).
add_shader(vulkan_runtime_shaders shaders/cast.comp)
add_shader(vulkan_runtime_shaders shaders/unary.comp)
add_shader(vulkan_runtime_shaders shaders/sgemm_f16.comp)
add_shader(vulkan_runtime_shaders shaders/rmsnorm_f16.comp)
add_shader(vulkan_runtime_shaders shaders/kvquantize.comp)
add_shader(vulkan_runtime_shaders shaders/attention_f16.comp)
add_shader(vulkan_runtime_shaders shaders/attention_i8kv.comp)
add_shader(vulkan_runtime_shaders shaders/transform_act.comp)
add_shader(vulkan_runtime_shaders shaders/dequant_w.comp)
add_shader(vulkan_runtime_shaders shaders/embed_cq4.comp)
# M6c-2: arbitrary-M GEMM (decode M==1) + packed<->word layout bridging.
add_shader(vulkan_runtime_shaders shaders/sgemm_f16_am.comp)
add_shader(vulkan_runtime_shaders shaders/fp16pack.comp)
embed_shaders(vulkan_runtime_shaders)

# ---------------------------------------------------------------------------
# Smoke target — the M0 functional proof.
# Creates an instance (VVL if enumerated), a compute device, and a VMA
# allocator, prints the device name, and tears everything down.
# ---------------------------------------------------------------------------
add_executable(smoke src/main.cpp)
target_link_libraries(smoke PRIVATE vulkan_runtime_core vulkan_runtime_headers Vulkan::Vulkan)

# ---------------------------------------------------------------------------
# vector_add test — the M1 functional proof (correct vs CPU on RADV + lavapipe).
# ---------------------------------------------------------------------------
add_executable(vector_add tests/test_vector_add.cpp)
target_link_libraries(vector_add PRIVATE
    vulkan_runtime_core
    vulkan_runtime_shaders
    vulkan_runtime_headers
    Vulkan::Vulkan
)

# ---------------------------------------------------------------------------
# sgemm test — the M2 functional proof (correct vs CPU on RADV + lavapipe).
# ---------------------------------------------------------------------------
add_executable(sgemm tests/test_sgemm.cpp)
target_link_libraries(sgemm PRIVATE
    vulkan_runtime_core
    vulkan_runtime_shaders
    vulkan_runtime_headers
    Vulkan::Vulkan
)

# ---------------------------------------------------------------------------
# M4a kernel library tests — transpose / RMSNorm / RoPE, correct vs CPU on
# RADV + lavapipe. (int4-dequant-GEMM and attention are M4b, not here.)
# ---------------------------------------------------------------------------
add_executable(transpose tests/test_transpose.cpp)
target_link_libraries(transpose PRIVATE
    vulkan_runtime_core
    vulkan_runtime_shaders
    vulkan_runtime_headers
    Vulkan::Vulkan
)

add_executable(rmsnorm tests/test_rmsnorm.cpp)
add_executable(combined_training_continuation tests/test_combined_training_continuation.cpp)
add_executable(combined_loss_decrease tests/test_combined_loss_decrease.cpp)
add_executable(combined_loss_trace tests/test_combined_loss_trace.cpp)
add_executable(combined_gradient_readback tests/test_combined_gradient_readback.cpp)
add_executable(gamma_finite_difference tests/test_gamma_finite_difference.cpp)
add_executable(position_finite_difference tests/test_position_finite_difference.cpp)
add_executable(base_checkpoint_cabi tests/test_base_checkpoint_cabi.cpp)
add_executable(training_gates tests/test_training_gates.cpp)
add_executable(bounded_training_orchestration tests/test_bounded_training_orchestration.cpp)
add_executable(bounded_full_graph_cabi tests/test_bounded_full_graph_cabi.cpp)
add_executable(tiny_rmsnorm_staged_state tests/test_tiny_rmsnorm_staged_state.cpp)
target_link_libraries(tiny_rmsnorm_staged_state PRIVATE vulkan_runtime_api)
add_dependencies(tiny_rmsnorm_staged_state vulkan_runtime_shaders_embed)
add_executable(sync_validation_gate tests/test_sync_validation_gate.cpp)
add_executable(gamma_forward_effect tests/test_gamma_forward_effect.cpp)
target_link_libraries(gamma_forward_effect PRIVATE vulkan_runtime_api vulkan_runtime_core vulkan_runtime_headers Vulkan::Vulkan)
add_executable(loss_decrease_bounded tests/test_loss_decrease_bounded.cpp)

target_link_libraries(loss_decrease_bounded PRIVATE vulkan_runtime_api vulkan_runtime_core vulkan_runtime_headers Vulkan::Vulkan)
target_link_libraries(combined_training_continuation PRIVATE vulkan_runtime_api vulkan_runtime_core vulkan_runtime_headers Vulkan::Vulkan)
target_link_libraries(combined_loss_decrease PRIVATE vulkan_runtime_api vulkan_runtime_core vulkan_runtime_headers Vulkan::Vulkan)
target_link_libraries(combined_loss_trace PRIVATE vulkan_runtime_api vulkan_runtime_core vulkan_runtime_headers Vulkan::Vulkan)
target_link_libraries(combined_gradient_readback PRIVATE vulkan_runtime_api vulkan_runtime_core vulkan_runtime_headers Vulkan::Vulkan)
target_link_libraries(gamma_finite_difference PRIVATE vulkan_runtime_api vulkan_runtime_core vulkan_runtime_headers Vulkan::Vulkan)
target_link_libraries(position_finite_difference PRIVATE vulkan_runtime_api vulkan_runtime_core vulkan_runtime_headers Vulkan::Vulkan)
target_link_libraries(base_checkpoint_cabi PRIVATE vulkan_runtime_api vulkan_runtime_core vulkan_runtime_headers Vulkan::Vulkan)
target_link_libraries(training_gates PRIVATE vulkan_runtime_api vulkan_runtime_core vulkan_runtime_headers Vulkan::Vulkan)
target_link_libraries(bounded_training_orchestration PRIVATE vulkan_runtime_api vulkan_runtime_core vulkan_runtime_headers Vulkan::Vulkan)
target_link_libraries(bounded_full_graph_cabi PRIVATE vulkan_runtime_api vulkan_runtime_core vulkan_runtime_headers Vulkan::Vulkan)
target_link_libraries(sync_validation_gate PRIVATE vulkan_runtime_api vulkan_runtime_core vulkan_runtime_headers Vulkan::Vulkan)
add_executable(combined_training_gating tests/test_combined_training_gating.cpp)
target_link_libraries(combined_training_gating PRIVATE vulkan_runtime_api vulkan_runtime_core vulkan_runtime_headers Vulkan::Vulkan)
add_executable(gamma_state_roundtrip tests/test_gamma_state_roundtrip.cpp)
target_link_libraries(gamma_state_roundtrip PRIVATE vulkan_runtime_api vulkan_runtime_core vulkan_runtime_headers Vulkan::Vulkan)
add_executable(ffn_state_roundtrip tests/test_ffn_state_roundtrip.cpp)
target_link_libraries(ffn_state_roundtrip PRIVATE vulkan_runtime_api vulkan_runtime_core vulkan_runtime_headers Vulkan::Vulkan)
add_executable(ffn_state_transfer tests/test_ffn_state_transfer.cpp)
target_link_libraries(ffn_state_transfer PRIVATE vulkan_runtime_api vulkan_runtime_core vulkan_runtime_headers Vulkan::Vulkan)
add_executable(ffn_state_pack tests/test_ffn_state_pack.cpp)
target_link_libraries(ffn_state_pack PRIVATE vulkan_runtime_api vulkan_runtime_core vulkan_runtime_headers Vulkan::Vulkan)
add_executable(gamma_state_abi tests/test_gamma_state_abi.cpp)
target_link_libraries(gamma_state_abi PRIVATE vulkan_runtime_api vulkan_runtime_core vulkan_runtime_headers Vulkan::Vulkan)
add_executable(ffn_c_abi_gating tests/test_ffn_c_abi_gating.cpp)
target_link_libraries(ffn_c_abi_gating PRIVATE vulkan_runtime_api vulkan_runtime_core vulkan_runtime_headers Vulkan::Vulkan)
add_executable(ffn_checkpoint_layout tests/test_ffn_checkpoint_layout.cpp)
target_link_libraries(ffn_checkpoint_layout PRIVATE vulkan_runtime_api vulkan_runtime_core vulkan_runtime_headers Vulkan::Vulkan)
add_executable(ffn_adamw_gpu tests/test_ffn_adamw_gpu.cpp)
target_link_libraries(ffn_adamw_gpu PRIVATE vulkan_runtime_core vulkan_runtime_shaders vulkan_runtime_headers Vulkan::Vulkan)
add_dependencies(ffn_adamw_gpu vulkan_runtime_shaders_embed)
add_executable(ffn_state_gating tests/test_ffn_state_gating.cpp)
target_link_libraries(ffn_state_gating PRIVATE vulkan_runtime_core vulkan_runtime_headers Vulkan::Vulkan)
add_executable(ffn_adamw tests/test_ffn_adamw.cpp)
add_executable(ffn_finite_difference tests/test_ffn_finite_difference.cpp)
add_executable(ffn_param_grads_gpu tests/test_ffn_param_grads_gpu.cpp)
target_link_libraries(ffn_param_grads_gpu PRIVATE vulkan_runtime_core vulkan_runtime_shaders vulkan_runtime_headers Vulkan::Vulkan)
add_dependencies(ffn_param_grads_gpu vulkan_runtime_shaders_embed)
add_executable(ffn_param_grads_cpu tests/test_ffn_param_grads_cpu.cpp)
add_executable(ffn_backward_gpu tests/test_ffn_backward_gpu.cpp)
target_link_libraries(ffn_backward_gpu PRIVATE vulkan_runtime_core vulkan_runtime_shaders vulkan_runtime_headers Vulkan::Vulkan)
add_dependencies(ffn_backward_gpu vulkan_runtime_shaders_embed)
add_executable(ffn_forward_masked_gpu tests/test_ffn_forward_masked_gpu.cpp)
target_link_libraries(ffn_forward_masked_gpu PRIVATE vulkan_runtime_core vulkan_runtime_shaders vulkan_runtime_headers Vulkan::Vulkan)
add_dependencies(ffn_forward_masked_gpu vulkan_runtime_shaders_embed)
add_executable(ffn_forward_gpu tests/test_ffn_forward_gpu.cpp)
target_link_libraries(ffn_forward_gpu PRIVATE vulkan_runtime_core vulkan_runtime_shaders vulkan_runtime_headers Vulkan::Vulkan)
add_dependencies(ffn_forward_gpu vulkan_runtime_shaders_embed)
add_executable(ffn_cpu_reference tests/test_ffn_cpu_reference.cpp)
add_executable(rmsnorm_gamma_adamw_gpu tests/test_rmsnorm_gamma_adamw_gpu.cpp)
target_link_libraries(rmsnorm_gamma_adamw_gpu PRIVATE vulkan_runtime_core vulkan_runtime_shaders vulkan_runtime_headers Vulkan::Vulkan)
add_dependencies(rmsnorm_gamma_adamw_gpu vulkan_runtime_shaders_embed)
add_executable(rmsnorm_gamma_adamw tests/test_rmsnorm_gamma_adamw.cpp)
add_executable(rmsnorm_backward_state tests/test_rmsnorm_backward_state.cpp)
target_link_libraries(rmsnorm_backward_state PRIVATE vulkan_runtime_core vulkan_runtime_shaders vulkan_runtime_headers Vulkan::Vulkan)
add_dependencies(rmsnorm_backward_state vulkan_runtime_shaders_embed)
add_executable(rmsnorm_forward_state tests/test_rmsnorm_forward_state.cpp)
target_link_libraries(rmsnorm_forward_state PRIVATE vulkan_runtime_core vulkan_runtime_shaders vulkan_runtime_headers Vulkan::Vulkan)
add_dependencies(rmsnorm_forward_state vulkan_runtime_shaders_embed)
add_executable(rmsnorm_dgamma_masked tests/test_rmsnorm_dgamma_masked.cpp)
target_link_libraries(rmsnorm_dgamma_masked PRIVATE vulkan_runtime_core vulkan_runtime_shaders vulkan_runtime_headers Vulkan::Vulkan)
add_dependencies(rmsnorm_dgamma_masked vulkan_runtime_shaders_embed)
add_executable(rmsnorm_masked tests/test_rmsnorm_masked.cpp)
target_link_libraries(rmsnorm_masked PRIVATE vulkan_runtime_core vulkan_runtime_shaders vulkan_runtime_headers Vulkan::Vulkan)
add_dependencies(rmsnorm_masked vulkan_runtime_shaders_embed)
add_executable(rmsnorm_dgamma tests/test_rmsnorm_dgamma.cpp)
target_link_libraries(rmsnorm_dgamma PRIVATE vulkan_runtime_core vulkan_runtime_shaders vulkan_runtime_headers Vulkan::Vulkan)
add_dependencies(rmsnorm_dgamma vulkan_runtime_shaders_embed)
add_executable(rmsnorm_backward tests/test_rmsnorm_backward.cpp)
target_link_libraries(rmsnorm_backward PRIVATE vulkan_runtime_core vulkan_runtime_shaders vulkan_runtime_headers Vulkan::Vulkan)
add_dependencies(rmsnorm_backward vulkan_runtime_shaders_embed)
target_link_libraries(rmsnorm PRIVATE
    vulkan_runtime_core
    vulkan_runtime_shaders
    vulkan_runtime_headers
    Vulkan::Vulkan
)

add_executable(rope tests/test_rope.cpp)
target_link_libraries(rope PRIVATE
    vulkan_runtime_core
    vulkan_runtime_shaders
    vulkan_runtime_headers
    Vulkan::Vulkan
)

# ---------------------------------------------------------------------------
# M4b kernel library tests — int4-dequant GEMM (Q4_0-style) and flash-style
# attention (online softmax), correct vs CPU on RADV + lavapipe.
# ---------------------------------------------------------------------------
add_executable(q4gemm tests/test_q4gemm.cpp)
target_link_libraries(q4gemm PRIVATE
    vulkan_runtime_core
    vulkan_runtime_shaders
    vulkan_runtime_headers
    Vulkan::Vulkan
)

add_executable(attention tests/test_attention.cpp)
target_link_libraries(attention PRIVATE
    vulkan_runtime_core
    vulkan_runtime_shaders
    vulkan_runtime_headers
    Vulkan::Vulkan
)

# ---------------------------------------------------------------------------
# Exec engine — the M5a N-slot ring + timeline-semaphore core. Depends on the
# core library (PUBLIC so consumers get core include dirs + Vulkan
# transitively). src/exec/engine.cpp is the engine; tests/exec_engine.cpp is
# its functional proof.
# ---------------------------------------------------------------------------
add_library(vulkan_runtime_exec STATIC src/exec/engine.cpp)
set_target_properties(vulkan_runtime_exec PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(vulkan_runtime_exec PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(vulkan_runtime_exec PUBLIC vulkan_runtime_core)
target_link_libraries(vulkan_runtime_api PUBLIC vulkan_runtime_exec)

add_executable(exec_engine tests/test_exec_engine.cpp)
target_link_libraries(exec_engine PRIVATE vulkan_runtime_exec vulkan_runtime_shaders vulkan_runtime_headers Vulkan::Vulkan)
add_executable(immutable_command_buffer tests/test_immutable_command_buffer.cpp)
target_link_libraries(immutable_command_buffer PRIVATE vulkan_runtime_exec vulkan_runtime_headers Vulkan::Vulkan)
add_executable(tiny_forward_persistent tests/test_tiny_forward_persistent.cpp)
target_link_libraries(tiny_forward_persistent PRIVATE vulkan_runtime_api)
add_executable(tiny_graph_dstate tests/test_tiny_graph_dstate.cpp)
target_link_libraries(tiny_graph_dstate PRIVATE vulkan_runtime_api)
add_executable(tiny_metrics_api tests/test_tiny_metrics_api.cpp)
target_link_libraries(tiny_metrics_api PRIVATE vulkan_runtime_api)
add_executable(tiny_profile_api tests/test_tiny_profile_api.cpp)
target_link_libraries(tiny_profile_api PRIVATE vulkan_runtime_api)
add_executable(tiny_immutable_prototype tests/test_tiny_immutable_prototype.cpp)
target_link_libraries(tiny_immutable_prototype PRIVATE vulkan_runtime_api)
add_executable(dataset_batch_buffer tests/test_dataset_batch_buffer.cpp)
target_link_libraries(dataset_batch_buffer PRIVATE vulkan_runtime_api)
add_executable(reduced_precision_storage tests/test_reduced_precision_storage.cpp)
target_link_libraries(reduced_precision_storage PRIVATE vulkan_runtime_api)
target_link_libraries(exec_engine PRIVATE
    vulkan_runtime_exec
    vulkan_runtime_shaders
    vulkan_runtime_headers
    Vulkan::Vulkan
)

# ---------------------------------------------------------------------------
# Bench harness — reusable timestamp-query timing (median/min/max), shared by
# the M3 SGEMM driver and every M4+ benchmark. Depends on the core library
# (PUBLIC so consumers get the core include dirs + Vulkan transitively).
# ---------------------------------------------------------------------------
add_library(vulkan_runtime_bench STATIC
    bench/bench_common.cpp
)
target_include_directories(vulkan_runtime_bench PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}" # for bench/bench_common.h
)
target_link_libraries(vulkan_runtime_bench PUBLIC
    vulkan_runtime_core
    vulkan_runtime_headers
    Vulkan::Vulkan
)

# ---------------------------------------------------------------------------
# sgemm benchmark — the M3 tuning harness: size sweep + variant ranking
# (timestamp queries, median-of-runs, >=1.0 TFLOPS gate @ 1024^3, variants
# ranked against the M2 3.19 TFLOPS baseline).
# ---------------------------------------------------------------------------
add_executable(bench_sgemm bench/bench_sgemm.cpp)
target_link_libraries(bench_sgemm PRIVATE
    vulkan_runtime_bench
    vulkan_runtime_shaders
)

# ---------------------------------------------------------------------------
# q4gemm benchmark — the M4b "GEMM perf measured" gate: int4-dequant GEMM at
# 1024^3, timestamp-queried median, reported vs the sgemm 3.21 TFLOPS baseline.
# ---------------------------------------------------------------------------
add_executable(bench_q4gemm bench/bench_q4gemm.cpp)
target_link_libraries(bench_q4gemm PRIVATE
    vulkan_runtime_bench
    vulkan_runtime_shaders
)

# ---------------------------------------------------------------------------
# cq4gemm benchmark — the M7 KPI tool: the full CQ4 chain (transform_act ->
# dequant_w -> sgemm_f16_am, packed fp16) end-to-end, 1024^3 anchor plus
# needle-realistic M=1 and M=1024 prefill shapes, median reported vs the sgemm
# 3.21 TFLOPS baseline.
# ---------------------------------------------------------------------------
add_executable(bench_cq4gemm bench/bench_cq4gemm.cpp)
target_link_libraries(bench_cq4gemm PRIVATE
    vulkan_runtime_bench
    vulkan_runtime_shaders
)

# ---------------------------------------------------------------------------
# exec-engine benchmark — the M5b cross-queue ACE overlap demo: async copies on
# queue 1 overlapping sgemm dispatches on queue 0, timed serial vs pipelined.
# Depends on the exec engine (cross-queue timeline waits) + shaders + bench.
# ---------------------------------------------------------------------------
add_executable(bench_tiny_training bench/bench_tiny_training.cpp)
target_link_libraries(bench_tiny_training PRIVATE
    vulkan_runtime_api
    vulkan_runtime_bench
)

add_executable(bench_exec_engine bench/bench_exec_engine.cpp)
target_link_libraries(bench_exec_engine PRIVATE
    vulkan_runtime_bench
    vulkan_runtime_exec
    vulkan_runtime_shaders
)

# ---------------------------------------------------------------------------
# The benchmarks are perf tools whose *host-side* work (input generation, the
# double-precision CPU reference in bench_exec_engine, median/timestamp timing)
# must not be left at the -O0 debug default: an unoptimized 32x1024^3 double
# GEMM reference takes minutes and would make ctest look hung. -O2 optimizes
# only this harness code; the GPU shaders are unaffected (compiled by glslc -O
# independently, and GPU timings come from timestamp queries, not host clocks).
# ---------------------------------------------------------------------------
target_compile_options(bench_sgemm PRIVATE -O2)
target_compile_options(bench_q4gemm PRIVATE -O2)
target_compile_options(bench_cq4gemm PRIVATE -O2)
target_compile_options(bench_exec_engine PRIVATE -O2)
target_compile_options(bench_tiny_training PRIVATE -O2)

# ---------------------------------------------------------------------------
# cactus_x86 — self-contained x86 CPU Cactus kernel library (M6b-1).
#
# Compiled with clang++ (NOT the project-default g++): the Cactus ABI uses the
# ARM ACLE __fp16 storage type, which gcc rejects on x86_64 but clang accepts
# (arithmetic promotes to float). It is built via custom commands so the rest of
# the project keeps gcc. find_program falls back clang++ -> clang++-21 (only the
# -21 suffix binary is installed here); configure fails loudly if absent.
# ---------------------------------------------------------------------------
find_program(CACTUS_CLANG_CXX NAMES clang++ clang++-21 clang++-20 clang++-19)
if(NOT CACTUS_CLANG_CXX)
    message(FATAL_ERROR
        "cactus_x86 requires clang++ (for __fp16); found neither clang++ nor clang++-21 on PATH.")
endif()

set(CACTUS_X86_DIR "${CMAKE_CURRENT_BINARY_DIR}/cactus_x86")
file(MAKE_DIRECTORY "${CACTUS_X86_DIR}")
set(CACTUS_X86_OBJ "${CACTUS_X86_DIR}/cactus_x86.cpp.o")
set(CACTUS_X86_BRIDGE_OBJ "${CACTUS_X86_DIR}/cactus_x86_bridge.cpp.o")
set(CACTUS_X86_LIB "${CACTUS_X86_DIR}/libcactus_x86.a")
set(CACTUS_X86_TEST_BIN "${CACTUS_X86_DIR}/test_cactus_x86")

add_custom_command(
    OUTPUT ${CACTUS_X86_OBJ}
    COMMAND ${CACTUS_CLANG_CXX} -std=c++20 -O2 -fPIC
            -I "${CMAKE_CURRENT_SOURCE_DIR}/src"
            -Wall -Wextra -Wpedantic
            -c "${CMAKE_CURRENT_SOURCE_DIR}/src/cactus/cactus_x86.cpp"
            -o ${CACTUS_X86_OBJ}
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/cactus/cactus_x86.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/cactus/cactus_x86.h"
    VERBATIM
    COMMENT "Compiling cactus_x86.cpp with clang++ (__fp16)"
)

# M6c-1: gcc-compatible bridge (also clang++ — it includes cactus_x86.h) so the
# gcc-compiled GPU kernel tests can call the __fp16 CPU reference.
add_custom_command(
    OUTPUT ${CACTUS_X86_BRIDGE_OBJ}
    COMMAND ${CACTUS_CLANG_CXX} -std=c++20 -O2 -fPIC
            -I "${CMAKE_CURRENT_SOURCE_DIR}/src"
            -Wall -Wextra -Wpedantic
            -c "${CMAKE_CURRENT_SOURCE_DIR}/src/cactus/cactus_x86_bridge.cpp"
            -o ${CACTUS_X86_BRIDGE_OBJ}
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/cactus/cactus_x86_bridge.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/cactus/cactus_x86_bridge.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/cactus/cactus_x86.h"
    VERBATIM
    COMMENT "Compiling cactus_x86_bridge.cpp with clang++"
)

add_custom_command(
    OUTPUT ${CACTUS_X86_LIB}
    COMMAND "${CMAKE_AR}" rcs ${CACTUS_X86_LIB} ${CACTUS_X86_OBJ} ${CACTUS_X86_BRIDGE_OBJ}
    DEPENDS ${CACTUS_X86_OBJ} ${CACTUS_X86_BRIDGE_OBJ}
    VERBATIM
    COMMENT "Archiving libcactus_x86.a"
)

add_custom_target(cactus_x86_build ALL DEPENDS ${CACTUS_X86_LIB})

# Linkable imported target for M6b-2 (engine integration) to consume.
add_library(cactus_x86 STATIC IMPORTED GLOBAL)
set_target_properties(cactus_x86 PROPERTIES IMPORTED_LOCATION ${CACTUS_X86_LIB})
add_dependencies(cactus_x86 cactus_x86_build)

# CPU-only test (also clang++ because it uses __fp16). No Vulkan, no GPU: it
# runs identically on RADV and lavapipe.
add_custom_command(
    OUTPUT ${CACTUS_X86_TEST_BIN}
    COMMAND ${CACTUS_CLANG_CXX} -std=c++20 -O2
            -I "${CMAKE_CURRENT_SOURCE_DIR}/src"
            -Wall -Wextra -Wpedantic
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_cactus_x86.cpp"
            ${CACTUS_X86_LIB}
            -o ${CACTUS_X86_TEST_BIN}
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_cactus_x86.cpp" ${CACTUS_X86_LIB}
    VERBATIM
    COMMENT "Building test_cactus_x86 with clang++"
)
add_custom_target(test_cactus_x86_build ALL DEPENDS ${CACTUS_X86_TEST_BIN})

# ---------------------------------------------------------------------------
# M6c-1 kernel library tests — GPU kernels vs the cactus_x86 CPU reference.
#
# Each is a self-contained gcc-compiled test (house pattern) that links the
# imported cactus_x86 library (which now also contains the clang++ bridge) so
# it can call the __fp16 CPU reference via uint16_t fp16-bit arguments.
# ---------------------------------------------------------------------------
set(M6C1_TEST_LIBS
    vulkan_runtime_core
    vulkan_runtime_shaders
    vulkan_runtime_headers
    Vulkan::Vulkan
    cactus_x86
)

add_executable(cast tests/test_cast.cpp)
target_link_libraries(cast PRIVATE ${M6C1_TEST_LIBS})

add_executable(unary tests/test_unary.cpp)
target_link_libraries(unary PRIVATE ${M6C1_TEST_LIBS})

add_executable(sgemm_f16 tests/test_sgemm_f16.cpp)
target_link_libraries(sgemm_f16 PRIVATE ${M6C1_TEST_LIBS})

add_executable(rmsnorm_f16 tests/test_rmsnorm_f16.cpp)
target_link_libraries(rmsnorm_f16 PRIVATE ${M6C1_TEST_LIBS})

add_executable(kvquantize tests/test_kvquantize.cpp)
target_link_libraries(kvquantize PRIVATE ${M6C1_TEST_LIBS})

add_executable(attention_f16 tests/test_attention_f16.cpp)
target_link_libraries(attention_f16 PRIVATE ${M6C1_TEST_LIBS})

add_executable(attention_i8kv tests/test_attention_i8kv.cpp)
target_link_libraries(attention_i8kv PRIVATE ${M6C1_TEST_LIBS})

add_executable(cq4 tests/test_cq4.cpp)
target_link_libraries(cq4 PRIVATE ${M6C1_TEST_LIBS})

add_executable(embed_cq4 tests/test_embed_cq4.cpp)
target_link_libraries(embed_cq4 PRIVATE ${M6C1_TEST_LIBS})

add_executable(sgemm_f16_am tests/test_sgemm_f16_am.cpp)
target_link_libraries(sgemm_f16_am PRIVATE ${M6C1_TEST_LIBS})

add_executable(fp16pack tests/test_fp16pack.cpp)
target_link_libraries(fp16pack PRIVATE ${M6C1_TEST_LIBS})

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
enable_testing()
add_test(NAME smoke COMMAND smoke)
add_test(NAME vector_add COMMAND vector_add)
add_test(NAME sgemm COMMAND sgemm)
add_test(NAME bench_sgemm COMMAND bench_sgemm)
add_test(NAME transpose COMMAND transpose)
add_test(NAME rmsnorm COMMAND rmsnorm)
add_test(NAME combined_training_continuation COMMAND combined_training_continuation)
add_test(NAME combined_loss_decrease COMMAND combined_loss_decrease)
add_test(NAME combined_loss_trace COMMAND combined_loss_trace)
add_test(NAME combined_gradient_readback COMMAND combined_gradient_readback)
add_test(NAME gamma_finite_difference COMMAND gamma_finite_difference)
add_test(NAME position_finite_difference COMMAND position_finite_difference)
add_test(NAME base_checkpoint_cabi COMMAND base_checkpoint_cabi)
add_test(NAME training_gates COMMAND training_gates)
add_test(NAME bounded_training_orchestration COMMAND bounded_training_orchestration)
add_test(NAME bounded_full_graph_cabi COMMAND bounded_full_graph_cabi)
add_test(NAME sync_validation_gate COMMAND sync_validation_gate)
add_test(NAME gamma_forward_effect COMMAND gamma_forward_effect)
add_test(NAME loss_decrease_bounded COMMAND loss_decrease_bounded)
add_test(NAME combined_training_gating COMMAND combined_training_gating)
add_test(NAME gamma_state_roundtrip COMMAND gamma_state_roundtrip)
add_test(NAME ffn_state_roundtrip COMMAND ffn_state_roundtrip)
add_test(NAME ffn_state_transfer COMMAND ffn_state_transfer)
add_test(NAME ffn_state_pack COMMAND ffn_state_pack)
add_test(NAME gamma_state_abi COMMAND gamma_state_abi)
add_test(NAME ffn_c_abi_gating COMMAND ffn_c_abi_gating)
add_test(NAME ffn_checkpoint_layout COMMAND ffn_checkpoint_layout)
add_test(NAME ffn_adamw_gpu COMMAND ffn_adamw_gpu)
add_test(NAME ffn_state_gating COMMAND ffn_state_gating)
add_test(NAME ffn_adamw COMMAND ffn_adamw)
add_test(NAME ffn_finite_difference COMMAND ffn_finite_difference)
add_test(NAME ffn_param_grads_gpu COMMAND ffn_param_grads_gpu)
add_test(NAME ffn_param_grads_cpu COMMAND ffn_param_grads_cpu)
add_test(NAME ffn_backward_gpu COMMAND ffn_backward_gpu)
add_test(NAME ffn_forward_masked_gpu COMMAND ffn_forward_masked_gpu)
add_test(NAME ffn_forward_gpu COMMAND ffn_forward_gpu)
add_test(NAME ffn_cpu_reference COMMAND ffn_cpu_reference)
add_test(NAME rmsnorm_backward_state COMMAND rmsnorm_backward_state)
add_test(NAME rmsnorm_forward_state COMMAND rmsnorm_forward_state)
add_test(NAME embedding_training_rmsnorm_dx COMMAND embedding_training_rmsnorm_dx)
add_test(NAME position_training_rmsnorm_dx COMMAND position_training_rmsnorm_dx)
add_test(NAME rmsnorm_gamma_adamw_gpu COMMAND rmsnorm_gamma_adamw_gpu)
add_test(NAME rmsnorm_gamma_adamw COMMAND rmsnorm_gamma_adamw)
add_test(NAME rmsnorm_dgamma_masked COMMAND rmsnorm_dgamma_masked)
add_test(NAME rmsnorm_masked COMMAND rmsnorm_masked)
add_test(NAME rmsnorm_dgamma COMMAND rmsnorm_dgamma)
add_test(NAME rmsnorm_backward COMMAND rmsnorm_backward)
add_test(NAME rope COMMAND rope)
add_test(NAME q4gemm COMMAND q4gemm)
add_test(NAME attention COMMAND attention)
add_test(NAME attention_causal_api COMMAND attention_causal_api)
add_test(NAME lora_delta_api COMMAND lora_delta_api)
add_test(NAME lora_gradients_api COMMAND lora_gradients_api)
add_test(NAME lora_sgd_api COMMAND lora_sgd_api)
add_test(NAME lora_train_api COMMAND lora_train_api)
add_test(NAME lora_session_api COMMAND lora_session_api)
add_test(NAME causal_loss_api COMMAND causal_loss_api)
add_test(NAME attention_causal_backward_api COMMAND attention_causal_backward_api)
add_test(NAME lm_head_backward_api COMMAND lm_head_backward_api)
add_test(NAME tiny_base_training COMMAND tiny_base_training)
add_test(NAME output_projection_training COMMAND output_projection_training)
add_test(NAME qkv_projection_training COMMAND qkv_projection_training)
add_test(NAME embedding_training COMMAND embedding_training)
add_test(NAME projection_backward_api COMMAND projection_backward_api)
add_test(NAME lora_gradients_multi_api COMMAND lora_gradients_multi_api)
add_test(NAME lora_sgd_multi_api COMMAND lora_sgd_multi_api)
add_test(NAME exec_engine COMMAND exec_engine)
add_test(NAME immutable_command_buffer COMMAND immutable_command_buffer)
add_test(NAME tiny_forward_persistent COMMAND tiny_forward_persistent)
add_test(NAME tiny_graph_dstate COMMAND tiny_graph_dstate)
add_test(NAME tiny_metrics_api COMMAND tiny_metrics_api)
add_test(NAME tiny_profile_api COMMAND tiny_profile_api)
add_test(NAME tiny_immutable_prototype COMMAND tiny_immutable_prototype)
add_test(NAME dataset_batch_buffer COMMAND dataset_batch_buffer)
add_test(NAME reduced_precision_storage COMMAND reduced_precision_storage)
add_test(NAME bench_q4gemm COMMAND bench_q4gemm)
add_test(NAME bench_cq4gemm COMMAND bench_cq4gemm)
add_test(NAME bench_exec_engine COMMAND bench_exec_engine)
add_test(NAME cactus_x86 COMMAND ${CACTUS_X86_TEST_BIN})
add_test(NAME cast COMMAND cast)
add_test(NAME unary COMMAND unary)
add_test(NAME sgemm_f16 COMMAND sgemm_f16)
add_test(NAME rmsnorm_f16 COMMAND rmsnorm_f16)
add_test(NAME kvquantize COMMAND kvquantize)
add_test(NAME attention_f16 COMMAND attention_f16)
add_test(NAME attention_i8kv COMMAND attention_i8kv)
add_test(NAME cq4 COMMAND cq4)
add_test(NAME embed_cq4 COMMAND embed_cq4)
add_test(NAME sgemm_f16_am COMMAND sgemm_f16_am)
add_test(NAME fp16pack COMMAND fp16pack)
set_tests_properties(bench_sgemm PROPERTIES LABELS "bench")
set_tests_properties(bench_q4gemm PROPERTIES LABELS "bench")
set_tests_properties(bench_cq4gemm PROPERTIES LABELS "bench")
set_tests_properties(bench_exec_engine PROPERTIES LABELS "bench")
