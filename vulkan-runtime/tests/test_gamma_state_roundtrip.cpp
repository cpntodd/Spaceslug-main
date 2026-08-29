#include "api/tiny_forward_persistent.h"
#include "core/vk_setup.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>
int main(){using namespace vulkan_runtime::tiny;std::vector<float> e(V*H,0.f),p(Tcap*H,0.f),lm(H*Vp,0.f),g(H),m(H),v(H),rg(H),rm(H),rv(H);for(std::size_t i=0;i<H;++i){g[i]=1.f+.01f*i;m[i]=-.02f*i;v[i]=.03f*i;}auto ctx=vulkan_runtime::core::create_context("gamma-state-roundtrip");ForwardResourceGraph graph(ctx,e.data(),p.data(),lm.data());if(graph.update_gamma_state(g.data(),m.data(),v.data(),23)!=0)return 1;std::uint64_t step=0;if(graph.readback_gamma_state(rg.data(),rm.data(),rv.data(),&step)!=0||step!=23)return 1;double err=0;for(std::size_t i=0;i<H;++i){err=std::max(err,std::abs(double(rg[i])-g[i]));err=std::max(err,std::abs(double(rm[i])-m[i]));err=std::max(err,std::abs(double(rv[i])-v[i]));}if(err>1e-6)return 1;std::cout<<"Gamma state roundtrip: PASS max="<<err<<" step="<<step<<"\n";return 0;}
