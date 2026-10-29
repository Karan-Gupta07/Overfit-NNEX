#pragma once
// Experimental reference coder, not a public format or portable numeric spec.
// Encode and decode use the same CPU inference and arithmetic coding.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using Bytes=std::vector<uint8_t>;
static void require(bool v,const char* m){if(!v)throw std::runtime_error(m);}
static Bytes load(const char* p,size_t maximum){
    std::ifstream f(p,std::ios::binary|std::ios::ate);
    require(bool(f),"cannot open input");auto size=f.tellg();
    require(size>=0&&uint64_t(size)<=maximum,"oversized input");
    Bytes b(static_cast<size_t>(size));f.seekg(0);
    f.read(reinterpret_cast<char*>(b.data()),b.size());
    require(bool(f)||b.empty(),"truncated input");return b;
}
static void save(const char* p,const Bytes& b){
    std::ofstream f(p,std::ios::binary);
    f.write(reinterpret_cast<const char*>(b.data()),b.size());
    require(bool(f),"cannot write output");
}
constexpr uint32_t TOTAL=1<<20;
struct BitWriter{
    Bytes b;int used=0;
    void put(int value){
        if(used==0)b.push_back(0);
        b.back()|=(value&1)<<(7-used);used=(used+1)&7;
    }
};
struct BitReader{
    const Bytes& b;size_t position=0;
    uint32_t get(){require(position<b.size()*8,"truncated arithmetic stream");
        uint32_t bit=(b[position/8]>>(7-position%8))&1;position++;return bit;}
};
struct ArithmeticEncoder{
    uint32_t low=0,high=0xffffffffU;size_t pending=0;BitWriter bits;
    void emit(int b){bits.put(b);while(pending){bits.put(1-b);pending--;}}
    void put(uint32_t a,uint32_t b){
        require(a<b&&b<=TOTAL,"invalid cumulative probabilities");
        uint64_t range=uint64_t(high)-low+1;
        high=uint32_t(low+range*b/TOTAL-1);low=uint32_t(low+range*a/TOTAL);
        while(true){
            if(high<0x80000000U)emit(0);
            else if(low>=0x80000000U){emit(1);low-=0x80000000U;high-=0x80000000U;}
            else if(low>=0x40000000U&&high<0xc0000000U){
                pending++;low-=0x40000000U;high-=0x40000000U;
            }else break;
            low<<=1;high=(high<<1)|1;
        }
    }
    Bytes finish(){pending++;emit(low<0x40000000U?0:1);
        bits.b.insert(bits.b.end(),4,0);return bits.b;}
};
struct ArithmeticDecoder{
    uint32_t low=0,high=0xffffffffU,code=0;BitReader bits;
    explicit ArithmeticDecoder(const Bytes& b):bits{b}{
        for(int i=0;i<32;i++)code=(code<<1)|bits.get();
    }
    uint32_t target()const{
        uint64_t range=uint64_t(high)-low+1;
        return uint32_t(((uint64_t(code)-low+1)*TOTAL-1)/range);
    }
    void take(uint32_t a,uint32_t b){
        uint64_t range=uint64_t(high)-low+1;
        high=uint32_t(low+range*b/TOTAL-1);low=uint32_t(low+range*a/TOTAL);
        while(true){
            if(high<0x80000000U){}
            else if(low>=0x80000000U){code-=0x80000000U;low-=0x80000000U;high-=0x80000000U;}
            else if(low>=0x40000000U&&high<0xc0000000U){
                code-=0x40000000U;low-=0x40000000U;high-=0x40000000U;
            }else break;
            low<<=1;high=(high<<1)|1;code=(code<<1)|bits.get();
        }
    }
};
