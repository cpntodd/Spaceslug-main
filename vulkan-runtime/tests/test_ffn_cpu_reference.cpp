#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>
static float gelu(float z){constexpr float a=.7978845608028654f;return .5f*z*(1.f+std::tanh(a*(z+.044715f*z*z*z)));}
int main(){constexpr uint32_t R=3,H=4,I=16;std::vector<float>x(R*H),w1(H*I),b1(I),w2(I*H),b2(H),y(R*H),dy(R*H,.1f),dx(R*H);for(size_t i=0;i<x.size();++i)x[i]=.03f*float(int(i%7)-3);for(size_t i=0;i<w1.size();++i)w1[i]=.02f*float(int(i%5)-2);for(size_t i=0;i<w2.size();++i)w2[i]=.02f*float(int(i%3)-1);for(uint32_t r=0;r<R;++r)for(uint32_t o=0;o<H;++o){float z=b2[o]+x[r*H+o];for(uint32_t j=0;j<I;++j){float q=b1[j];for(uint32_t k=0;k<H;++k)q+=x[r*H+k]*w1[k*I+j];z+=gelu(q)*w2[j*H+o];}y[r*H+o]=z;}for(uint32_t r=0;r<R;++r)for(uint32_t o=0;o<H;++o){float q=dy[r*H+o];for(uint32_t j=0;j<I;++j){float z=b1[j];for(uint32_t k=0;k<H;++k)z+=x[r*H+k]*w1[k*I+j];float u=0;for(uint32_t k=0;k<H;++k)u+=dy[r*H+k]*w2[j*H+k];float t=std::tanh(.7978845608f*(z+.044715f*z*z*z));float gd=.5f*(1+t)+.5f*z*(1-t*t)*.7978845608f*(1+3*.044715f*z*z);q+=gd*u*w1[o*I+j];}dx[r*H+o]=q;}for(float v:y)if(!std::isfinite(v))return 1;for(float v:dx)if(!std::isfinite(v))return 1;std::cout<<"FFN CPU reference: PASS\n";return 0;}
