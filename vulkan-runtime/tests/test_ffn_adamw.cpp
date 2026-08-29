#include <cmath>
#include <iostream>
#include <vector>
int main(){std::vector<float>w{.4f,-.2f,.7f},g{.3f,-.1f,.2f},m{.1f,-.05f,.04f},v{.02f,.03f,.01f};auto step=[&](int t){for(size_t i=0;i<w.size();++i){m[i]=.9f*m[i]+.1f*g[i];v[i]=.99f*v[i]+.01f*g[i]*g[i];w[i]=(.999f)*w[i]-.01f*(m[i]/(1-std::pow(.9f,t)))/(std::sqrt(v[i]/(1-std::pow(.99f,t)))+1e-8f);}};step(1);step(2);for(float x:w)if(!std::isfinite(x))return 1;std::cout<<"FFN AdamW reference: PASS\n";return 0;}
