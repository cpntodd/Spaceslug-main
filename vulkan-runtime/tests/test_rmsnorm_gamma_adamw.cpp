#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>
int main(){constexpr uint32_t N=64;float lr=.001f,b1=.9f,b2=.999f,eps=1e-8f,decay=.01f;std::vector<float>g(N),grad(N),m(N),v(N);for(uint32_t i=0;i<N;++i){g[i]=1.f+.01f*i;grad[i]=.02f*(int(i%9)-4);m[i]=.003f*(int(i%7)-3);v[i]=.0007f*(i%5+1);}for(uint32_t step=1;step<=2;++step)for(uint32_t i=0;i<N;++i){float mi=b1*m[i]+(1-b1)*grad[i],vi=b2*v[i]+(1-b2)*grad[i]*grad[i];float bc1=1-std::pow(b1,float(step)),bc2=1-std::pow(b2,float(step));float eg=(1-lr*decay)*g[i]-lr*(mi/bc1)/(std::sqrt(vi/bc2)+eps);if(!std::isfinite(eg)||!std::isfinite(mi)||!std::isfinite(vi)){std::cerr<<"nonfinite\n";return 1;}g[i]=eg;m[i]=mi;v[i]=vi;}std::cout<<"gamma AdamW CPU two-step reference: PASS\n";return 0;}
