#include "api/tiny_forward_persistent.h"
#include "core/vk_setup.h"
#include <cmath>
#include <iostream>
#include <vector>
int main(){using namespace vulkan_runtime::tiny;std::vector<float>e(V*H,.01f),p(Tcap*H,0.f),q(H*H,0.f),k(H*H,0.f),v(H*H,0.f),o(H*H,0.f),lm(H*Vp,0.f);for(unsigned i=0;i<H;++i){q[i*H+i]=k[i*H+i]=v[i*H+i]=o[i*H+i]=1.f;lm[i*Vp+(i%V)]=.02f;}auto c=vulkan_runtime::core::create_context("loss-decrease-bounded");ForwardResourceGraph g(c,e.data(),p.data(),q.data(),k.data(),v.data(),o.data(),lm.data());std::vector<unsigned>t(Tcap,1),y(Tcap,2),m(Tcap,0);m[0]=1;float before=0,after=0;unsigned n=0;g.forward_loss_fixed_metrics(t.data(),y.data(),m.data(),&before,&n);if(n!=1||!std::isfinite(before))return 1;if(g.train_lm_head_adamw(t.data(),y.data(),m.data(),1,.05f,.9f,.999f,1.e-8f,0.f)!=0)return 1;g.forward_loss_fixed_metrics(t.data(),y.data(),m.data(),&after,&n);if(n!=1||!std::isfinite(after)||!(after<before)){std::cerr<<"bounded loss did not decrease: before="<<before<<" after="<<after<<" count="<<n<<"\n";return 1;}std::cout<<"Bounded loss decrease: PASS before="<<before<<" after="<<after<<"\n";return 0;}
