#include "api/tiny_forward_persistent.h"
#include "core/vk_setup.h"
#include <algorithm>
#include <iostream>
#include <vector>
#include <cstring>
int main(){using namespace vulkan_runtime::tiny;constexpr std::size_t n=3*(H*4*H+4*H+4*H*H+H);std::vector<float>state(n,.125f);if(spaceslug_tiny_forward_readback_ffn_state(nullptr,state.data(),n,nullptr)!=-1||spaceslug_tiny_forward_update_ffn_state(nullptr,state.data(),n,7)!=-1){return 1;}if(std::strcmp(spaceslug_tiny_forward_ffn_capability(),"ffn_graph_state_allocated_forward_backward_w1_adamw_only")!=0)return 1;std::cout<<"FFN state transfer gate: PASS\n";return 0;}
