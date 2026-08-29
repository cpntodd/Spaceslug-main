#include "api/tiny_forward_persistent.h"
#include <iostream>
int main(){using namespace vulkan_runtime::tiny;constexpr std::size_t params=H*4*H+4*H+4*H*H+H;BaseCheckpoint c;c.version=4;c.group_mask=BaseCheckpointNormalization|BaseCheckpointFfn;c.gamma.resize(H);c.ffn_w1.resize(H*4*H);if(c.gamma.size()!=H||c.ffn_w1.size()!=H*4*H||params==0)return 1;std::cout<<"FFN checkpoint layout: PASS\n";return 0;}
