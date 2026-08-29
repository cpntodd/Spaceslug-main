#include "api/tiny_forward_persistent.h"
#include <iostream>
int main(){using namespace vulkan_runtime::tiny;constexpr std::size_t n=3*(H*4*H+4*H+4*H*H+H);if(ForwardResourceGraph::trainable_ffn_supported||ForwardResourceGraph::trainable_ffn_unsupported!=-6||n==0)return 1;std::cout<<"FFN state gating: PASS\n";return 0;}
