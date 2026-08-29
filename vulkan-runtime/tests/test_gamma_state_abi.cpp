#include "api/tiny_forward_persistent.h"
#include <iostream>
int main(){if(spaceslug_tiny_forward_readback_gamma_state(nullptr,nullptr,nullptr,nullptr,nullptr)!=-1)return 1;if(spaceslug_tiny_forward_update_gamma_state(nullptr,nullptr,nullptr,nullptr,0)!=-1)return 1;std::cout<<"Gamma state ABI gating: PASS\n";return 0;}
