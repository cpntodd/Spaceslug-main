#include "api/lm_head_backward_api.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>
int main(){constexpr std::uint32_t rows=3,vocab=8,hidden=4;std::vector<float>dl(rows*vocab),w(hidden*vocab),out(rows*hidden),ref(rows*hidden,0);for(std::size_t i=0;i<dl.size();++i)dl[i]=std::sin(i*.07f);for(std::size_t i=0;i<w.size();++i)w[i]=std::cos(i*.11f);for(std::uint32_t r=0;r<rows;++r)for(std::uint32_t h=0;h<hidden;++h)for(std::uint32_t token=0;token<vocab;++token)ref[r*hidden+h]+=dl[r*vocab+token]*w[h*vocab+token];if(spaceslug_lm_head_backward(dl.data(),w.data(),out.data(),rows,vocab,hidden)!=0)return 1;float e=0;for(std::size_t i=0;i<out.size();++i)e=std::max(e,std::abs(out[i]-ref[i]));if(e>3e-5f){std::cerr<<"LM backward mismatch "<<e<<'\n';return 1;}std::cout<<"LM backward: PASS max_abs_error="<<e<<'\n';}
