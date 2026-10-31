#pragma once
// NNEX v5 context mixing. Each plane is coded bit by bit, most significant bit
// first. The network's stretch for every bit is one mixer input next to hashed
// context nodes and their bit histories; gated mixers, a second mixer layer
// and adaptive probability maps follow. Everything is integer arithmetic with
// wrapping 32-bit sums, so encoder and decoder compute the same probabilities
// on any conforming platform; NEON only speeds up the same arithmetic.
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <new>
#include <vector>
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace v5 {

constexpr int ST_LIM=4095;
struct Tables{
    std::array<uint16_t,8192> squash{};std::array<int32_t,1025> log2frac{};std::array<int16_t,4096> stretch{};
    std::array<int32_t,1024> rate{};
    int64_t log2q(uint32_t n)const{
        const int e=31-__builtin_clz(n);
        const uint32_t m=e>=10?(n>>(e-10))&1023:(n<<(10-e))&1023;
        return int64_t(e)*65536+log2frac[m];
    }
    Tables(){
        // 2^(-k/256) in Q62 from an exact integer recurrence; C is 2^(-1/256)*2^62 rounded.
        constexpr uint64_t C=4599216278082138665ULL;
        std::vector<uint64_t> e(4097);e[0]=uint64_t(1)<<62;
        for(int k=0;k<4096;k++)e[k+1]=uint64_t((__uint128_t(e[k])*C+(__uint128_t(1)<<61))>>62);
        auto up=[&](int s){const __uint128_t den=(__uint128_t(1)<<62)+e[size_t(s)];
            return int64_t(((__uint128_t(1)<<78)+den/2)/den);};
        for(int s=-4096;s<4096;s++)
            squash[size_t(s+4096)]=uint16_t(std::clamp<int64_t>(s>=0?up(s):65536-up(-s),1,65535));
        // 65536 log2(1 + i/1024): 24 bits by repeated squaring in Q62, then rounded.
        for(int i=0;i<1024;i++){
            __uint128_t x=__uint128_t(1024+i)<<52;uint32_t y=0;
            for(int b=0;b<24;b++){
                x=(x*x+(__uint128_t(1)<<61))>>62;y<<=1;
                if(x>=(__uint128_t(2)<<62)){x>>=1;y|=1;}
            }
            log2frac[size_t(i)]=int32_t((y+128)>>8);
        }
        log2frac[1024]=65536;
        for(int i=0;i<4096;i++)
            stretch[size_t(i)]=int16_t(std::clamp<int64_t>(
                (log2q(uint32_t(2*i+1))-log2q(uint32_t(8191-2*i)))>>8,-ST_LIM,ST_LIM));
        for(int i=0;i<1024;i++)rate[size_t(i)]=int32_t(655360/(10*i+16));
    }
};
inline const Tables TB;
static inline int squash(int st){return TB.squash[size_t(std::clamp(st,-4096,4095)+4096)];}
static inline int64_t log2q(uint32_t n){return TB.log2q(n);}
static inline int clampst(int64_t s){return int(std::clamp<int64_t>(s,-ST_LIM,ST_LIM));}
// Stretch of a 16-bit probability from exact logarithms, finer than st16 near 0 and 1.
static inline int stretch16(int p){
    p=std::clamp(p,1,65535);return clampst((log2q(uint32_t(p))-log2q(uint32_t(65536-p)))>>8);
}
// Stretch of a 16-bit or 22-bit probability of a one, in 1/256 log2 units.
static inline int st16(uint32_t p){return TB.stretch[p>>4];}
static inline int st22(uint32_t p){return TB.stretch[p>>10];}
static inline uint64_t mix64(uint64_t x){
    x^=x>>31;x*=0x7fb5d329728ea185ULL;x^=x>>27;x*=0x81dadef4bc2dd44dULL;x^=x>>33;return x;
}
static inline uint64_t hc(uint64_t id,uint64_t a,uint64_t b=0,uint64_t c=0){
    return mix64((id+1)*0x9E3779B97F4A7C15ULL+a*0xC2B2AE3D27D4EB4FULL+b*0x165667B19E3779F9ULL+c*0xD6E8FEB86659FD93ULL);
}
static inline uint64_t next_hash(uint64_t h,int nib){return (h^(uint64_t(nib+1)*0x9E3779B97F4A7C15ULL))*0xD6E8FEB86659FD93ULL;}
static inline int qe(int e){
    const int a=std::abs(e);const int q=a==0?0:a==1?1:a==2?2:a<=4?3:a<=8?4:a<=16?5:a<=32?6:7;
    return 8+(e<0?-q:q);
}

// AArch64 SQRDMULH: round(2ab / 2^32), saturating only for INT32_MIN squared.
static inline int32_t rdmulh(int32_t a,int32_t b){
    if(a==std::numeric_limits<int32_t>::min()&&b==a)return std::numeric_limits<int32_t>::max();
    return int32_t((int64_t(a)*b+(int64_t(1)<<30))>>31);
}
static inline int32_t wrap(uint32_t v){return int32_t(v);}
// in << 15 and in << 13 as wrapping 32-bit values, n a multiple of 4.
static inline void shifted(const int32_t* in,int n,int32_t* x15,int32_t* x13){
#if defined(__ARM_NEON)
    for(int i=0;i<n;i+=4){
        const int32x4_t v=vld1q_s32(in+i);vst1q_s32(x15+i,vshlq_n_s32(v,15));vst1q_s32(x13+i,vshlq_n_s32(v,13));
    }
#else
    for(int i=0;i<n;i++){x15[i]=wrap(uint32_t(in[i])<<15);x13[i]=wrap(uint32_t(in[i])<<13);}
#endif
}

// The mixing streams' carryless binary arithmetic coder, 32 bits wide with
// bytewise output. p is the probability of a one in 1/65536, 1 to 65535; a one
// takes the lower part of the interval. finish() appends the fewest bytes that,
// followed by zeros, lie in the final interval, and a decoder reads zeros past
// the end of its bytes.
struct Encoder{
    uint32_t x1=0,x2=0xffffffffu;std::vector<uint8_t> out;
    void put(int bit,uint32_t p){
        const uint32_t xmid=x1+uint32_t((uint64_t(x2-x1)*p)>>16);
        if(bit)x2=xmid;else x1=xmid+1;
        while(((x1^x2)>>24)==0){out.push_back(uint8_t(x2>>24));x1<<=8;x2=(x2<<8)|255;}
    }
    std::vector<uint8_t> finish(){
        for(int n=0;n<=4;n++){
            const int s=32-8*n;const uint64_t v=((uint64_t(x1)+((uint64_t(1)<<s)-1))>>s)<<s;
            if(v<=x2){for(int k=0;k<n;k++)out.push_back(uint8_t(v>>(24-8*k)));break;}
        }
        return std::move(out);
    }
};
// A decoder takes its bytes from Source::next().
struct ByteSource{
    const uint8_t* in;size_t size,pos=0;
    uint32_t next(){return pos<size?in[pos++]:0u;}
};
template<class Source>
struct SourceDecoder{
    Source source;uint32_t x1=0,x2=0xffffffffu,x=0;
    explicit SourceDecoder(Source s):source(s){for(int k=0;k<4;k++)x=(x<<8)|source.next();}
    int get(uint32_t p){
        const uint32_t xmid=x1+uint32_t((uint64_t(x2-x1)*p)>>16);
        const int bit=x<=xmid;
        if(bit)x2=xmid;else x1=xmid+1;
        while(((x1^x2)>>24)==0){x1<<=8;x2=(x2<<8)|255;x=(x<<8)|source.next();}
        return bit;
    }
};
struct Decoder:SourceDecoder<ByteSource>{
    Decoder(const uint8_t* data,size_t n):SourceDecoder<ByteSource>(ByteSource{data,n}){}
};

// Weights in 1/65536. Updates skip errors under DEAD_ZONE; the rate of a weight
// set decays with its use: lr * (1 + 2560 / (256 + uses)), in steps of 16 uses.
// lr is MIXER_RATE for mixers over many inputs and SECOND_RATE for the second layer.
constexpr int DEAD_ZONE=25,MIXER_RATE=6,SECOND_RATE=8;
struct RateTable{
    std::array<std::array<int32_t,1024>,2> r{};
    RateTable(){
        for(int k=0;k<2;k++){
            const int lr=k?MIXER_RATE:SECOND_RATE;
            for(int b=0;b<1024;b++)r[size_t(k)][size_t(b)]=lr+int32_t(int64_t(10*lr)*256/(256+16*b));
        }
    }
};
inline const RateTable RT;
struct Mixer{
    int stride=0;const int32_t* rates=nullptr;std::vector<int32_t> w;std::vector<uint16_t> uses;
    void init(int inputs,int sets,int rate){
        stride=(inputs+3)&~3;rates=RT.r[rate==MIXER_RATE?1:0].data();
        w.assign(size_t(stride)*sets,0);uses.assign(size_t(sets),0u);
    }
    int32_t* at(int s){return &w[size_t(s)*stride];}
    void prefetch(int s)const{
        const char* p=reinterpret_cast<const char*>(&w[size_t(s)*stride]);
        for(int k=0;k<stride*4;k+=128)__builtin_prefetch(p+k);
    }
    // x15 = inputs << 15: the wrapping sum of rdmulh(w, x15) per term.
    int dot(const int32_t* __restrict x15,int s)const{
        const int32_t* __restrict ww=&w[size_t(s)*stride];
#if defined(__ARM_NEON)
        int32x4_t a0=vdupq_n_s32(0),a1=vdupq_n_s32(0);int i=0;
        for(;i+8<=stride;i+=8){
            a0=vaddq_s32(a0,vqrdmulhq_s32(vld1q_s32(ww+i),vld1q_s32(x15+i)));
            a1=vaddq_s32(a1,vqrdmulhq_s32(vld1q_s32(ww+i+4),vld1q_s32(x15+i+4)));
        }
        if(i<stride)a0=vaddq_s32(a0,vqrdmulhq_s32(vld1q_s32(ww+i),vld1q_s32(x15+i)));
        return clampst(vaddvq_s32(vaddq_s32(a0,a1)));
#else
        uint32_t sum=0;
        for(int i=0;i<stride;i++)sum+=uint32_t(rdmulh(ww[i],x15[i]));
        return clampst(wrap(sum));
#endif
    }
    // out[k] = mixers[k].dot(x15, sets[k]) for mixers of one stride, in one pass
    // over x15. Wrapping sums do not depend on the order of their terms.
    template<size_t N>
    static void dots(const std::array<Mixer,N>& mixers,const std::array<int,N>& sets,
                     const int32_t* __restrict x15,int32_t* out){
#if defined(__ARM_NEON)
        const int32_t* ww[N];int32x4_t acc[N];
        for(size_t k=0;k<N;k++){ww[k]=&mixers[k].w[size_t(sets[k])*mixers[k].stride];acc[k]=vdupq_n_s32(0);}
        const int stride=mixers[0].stride;
        for(int i=0;i<stride;i+=4){
            const int32x4_t xv=vld1q_s32(x15+i);
            for(size_t k=0;k<N;k++)acc[k]=vaddq_s32(acc[k],vqrdmulhq_s32(vld1q_s32(ww[k]+i),xv));
        }
        for(size_t k=0;k<N;k++)out[k]=clampst(vaddvq_s32(acc[k]));
#else
        for(size_t k=0;k<N;k++)out[k]=mixers[k].dot(x15,sets[k]);
#endif
    }
    // update(x13, sets[k], errors[k]) for mixers of one stride, in one pass over x13.
    template<size_t N>
    static void updates(std::array<Mixer,N>& mixers,const std::array<int,N>& sets,
                        const int32_t* __restrict x13,const int* errors){
#if defined(__ARM_NEON)
        int32_t* ww[N];int32x4_t ev[N];int n=0;
        for(size_t k=0;k<N;k++){
            const int err=errors[k];if(std::abs(err)<DEAD_ZONE)continue;
            Mixer& m=mixers[k];const size_t s=size_t(sets[k]);
            const uint32_t c=m.uses[s];if(c<16383)m.uses[s]=uint16_t(c+1);
            ww[n]=&m.w[s*m.stride];ev[n]=vdupq_n_s32(err*m.rates[c>>4]);n++;
        }
        const int stride=mixers[0].stride;
        for(int i=0;i<stride;i+=4){
            const int32x4_t xv=vld1q_s32(x13+i);
            for(int k=0;k<n;k++)vst1q_s32(ww[k]+i,vaddq_s32(vld1q_s32(ww[k]+i),vqrdmulhq_s32(xv,ev[k])));
        }
#else
        for(size_t k=0;k<N;k++)mixers[k].update(x13,sets[k],errors[k]);
#endif
    }
    // x13 = inputs << 13: w += rdmulh(x13, err * rate), wrapping.
    void update(const int32_t* __restrict x13,int s,int err){
        if(std::abs(err)<DEAD_ZONE)return;
        int32_t* __restrict ww=&w[size_t(s)*stride];
        const uint32_t c=uses[size_t(s)];if(c<16383)uses[size_t(s)]=uint16_t(c+1);
        const int32_t e=err*rates[c>>4];
#if defined(__ARM_NEON)
        const int32x4_t ev=vdupq_n_s32(e);
        for(int i=0;i<stride;i+=4)vst1q_s32(ww+i,vaddq_s32(vld1q_s32(ww+i),vqrdmulhq_s32(vld1q_s32(x13+i),ev)));
#else
        for(int i=0;i<stride;i++)ww[i]=wrap(uint32_t(ww[i])+uint32_t(rdmulh(x13[i],e)));
#endif
    }
};
struct APM{
    std::vector<uint16_t> t;int index=0,weight=0;
    void init(int contexts){
        std::array<uint16_t,33> row;for(int j=0;j<33;j++)row[size_t(j)]=uint16_t(squash((j-16)*256));
        t.resize(size_t(contexts)*33);
        for(int c=0;c<contexts;c++)std::copy(row.begin(),row.end(),t.begin()+long(c)*33);
    }
    int pp(int st,int ctx){
        const int s=std::clamp(st,-4095,4095)+4096;const int j=s>>8;weight=s&255;index=ctx*33+j;
        return (t[size_t(index)]*(256-weight)+t[size_t(index)+1]*weight)>>8;
    }
    // A row's 66 bytes span at most two 64-byte lines.
    void prefetch(int ctx)const{const char* p=reinterpret_cast<const char*>(&t[size_t(ctx)*33]);__builtin_prefetch(p);__builtin_prefetch(p+65);}
    void update(int bit){
        const int g=bit?65535:0;uint16_t* e=&t[size_t(index)];
        e[0]=uint16_t(e[0]+((((g-e[0])*(256-weight))>>8)>>7));
        e[1]=uint16_t(e[1]+((((g-e[1])*weight)>>8)>>7));
    }
};
// Node: bits 0-7 bit history (zeros 0-2, ones 3-5, last two bits 6-7), 8-15
// update count, 16-31 probability of a one. The count reaches 255, so a node's
// rate falls to about 1/256: images are close to stationary.
constexpr uint32_t NODE_INIT=32768u<<16;
constexpr int HISTORIES=256,COUNT_LIMIT=255;
static inline uint32_t history_step(uint32_t s,int bit){
    int n0=s&7,n1=(s>>3)&7,l=(s>>6)&3;
    if(bit){if(n1<7)n1++;if(n0>2)n0=n0/2+1;}else{if(n0<7)n0++;if(n1>2)n1=n1/2+1;}
    l=((l<<1)|bit)&3;return uint32_t(n0|(n1<<3)|(l<<6));
}
struct HistoryTable{
    std::array<uint16_t,2*HISTORIES> next{};
    HistoryTable(){for(uint32_t s=0;s<HISTORIES;s++)for(int b=0;b<2;b++)next[s*2+uint32_t(b)]=uint16_t(history_step(s,b));}
};
inline const HistoryTable HT;
static inline void bump(uint32_t& z,int bit){
    int64_t p=z>>16;int n=(z>>8)&255;
    p+=(((bit?65535:0)-p)*TB.rate[size_t(n)]+32768)>>16;p=std::clamp<int64_t>(p,1,65535);
    if(n<COUNT_LIMIT)n++;
    z=(uint32_t(p)<<16)|(uint32_t(n)<<8)|HT.next[(z&255)*2+uint32_t(bit)];
}
// History map entry: bits 0-9 count, 10-31 probability (22 bits).
static inline void bump32(uint32_t& z,int bit){
    int64_t p=z>>10;int n=z&1023;const int64_t t=bit?(1<<22)-1:0;
    p+=((t-p)*TB.rate[size_t(n)])>>16;p=std::clamp<int64_t>(p,1,(1<<22)-1);if(n<1023)n++;
    z=uint32_t(n)|(uint32_t(p)<<10);
}
// A bucket is one 128-byte line holding two slots; a slot holds a nibble's 15
// nodes after a header word (bits 16-31 check, 0-15 priority). A context
// hashes to an aligned pair of buckets and may take any of its WAYS slots; a
// miss replaces the first slot of lowest priority.
struct alignas(128) Bucket{uint32_t s[2][16];};
constexpr int WAYS=4;
struct Table{
    Bucket* b=nullptr;int shift=64;
    std::unique_ptr<void,void(*)(void*)> mem{nullptr,std::free};
    void init(int bits){
        const size_t count=size_t(1)<<bits;void* p=std::calloc(count*sizeof(Bucket)+128,1);
        if(!p)throw std::bad_alloc();
        mem.reset(p);
        b=reinterpret_cast<Bucket*>((reinterpret_cast<uintptr_t>(p)+127)&~uintptr_t(127));shift=64-bits;
    }
    Bucket* pair(uint64_t h)const{return &b[(h>>shift)&~uint64_t(1)];}
    void prefetch(uint64_t h)const{const Bucket* P=pair(h);__builtin_prefetch(P);__builtin_prefetch(P+1);}
    uint32_t* claim(uint64_t h){
        Bucket* P=pair(h);const uint32_t c=(uint32_t(h>>16)&0xffff)|1;
        uint32_t* s[WAYS];for(int k=0;k<WAYS;k++)s[k]=P[k>>1].s[k&1];
        uint32_t* v=nullptr;
        const unsigned hit=unsigned((s[0][0]>>16)==c)|unsigned((s[1][0]>>16)==c)<<1|
                           unsigned((s[2][0]>>16)==c)<<2|unsigned((s[3][0]>>16)==c)<<3;
        if(hit)v=s[__builtin_ctz(hit)];
        else{
            int m=0;for(int k=1;k<WAYS;k++)if((s[k][0]&0xffff)<(s[m][0]&0xffff))m=k;
            v=s[m];v[0]=c<<16;for(int j=1;j<16;j++)v[j]=NODE_INIT;
        }
        if((v[0]&0xffff)<0xffff)v[0]++;
        return v;
    }
};

// Contexts of G, and of R and B: plane contexts, then SHARED_CONTEXTS of every plane.
constexpr int SHARED_CONTEXTS=7,CONTEXTS=16+SHARED_CONTEXTS,CHROMA_CONTEXTS=19+SHARED_CONTEXTS;
constexpr int MAX_CONTEXTS=CHROMA_CONTEXTS;
constexpr int MAX_INPUTS=2+2*MAX_CONTEXTS+2;
// Mixer weight-set selectors, see Coder::code.
constexpr std::array<int,6> SELECTOR_SETS={256,64,256,128,256,256};
constexpr int MIXERS=int(SELECTOR_SETS.size()),MIXER_PAD=(MIXERS+1+3)&~3;
// The final mixer's stretched inputs: the second layer's output and the six
// adaptive probability maps.
constexpr int BLEND_INPUTS=7;
// Confidence gate (gated models): a (bit, |stretch| >> 8) bucket codes with the network alone
// while the mixed model's recent cost there is within GATE_RATIO permille of
// the network's, except every (GATE_PROBE + 1)th bit, which keeps both costs
// fresh. Costs are exponential sums decaying by 2^-GATE_DECAY per mixed bit.
constexpr int GATE_BUCKETS=17,GATE_MIN_BUCKET=2,GATE_RATIO=5,GATE_DECAY=12;
constexpr uint32_t GATE_PROBE=7,GATE_WARMUP=256;
// Table bits of a band model: 2^bits buckets of 128 bytes, at least one per sample.
constexpr int MIN_TABLE_BITS=16,MAX_TABLE_BITS=22;
static int table_bits(uint64_t samples){
    int bits=MIN_TABLE_BITS;while(bits<MAX_TABLE_BITS&&(uint64_t(1)<<bits)<samples)bits++;return bits;
}

// What a plane's model reads. P holds samples, Pc coded samples: G as is, R
// and B as (v - G + 128) & 255. mode is the network's most likely coded
// sample, scale its scale bucket and cost the network's cost of each coded
// sample in 1/16 bits; across and down count the coded samples, up to 255,
// in the run of equal ones that ends at each sample from the left and from
// above. Coder::code writes cost, across and down.
struct Planes{
    int w=0,h=0;
    std::array<const uint8_t*,3> P{},Pc{},mode{},scale{};
    std::array<uint16_t*,3> cost{};
    std::array<uint8_t*,3> across{},down{};
};
// Run lengths in eleven buckets.
static inline int run_bucket(int r){return r<4?r:r<6?4:r<8?5:r<12?6:r<16?7:r<24?8:r<32?9:10;}

struct Model{
    int ch,na,nin;bool gated;Table T;std::vector<uint32_t> histories;
    std::array<Mixer,MIXERS> mx,mid;Mixer fin,blend;APM a1,a2;std::array<APM,4> ax;
    int64_t gate_nn[8][GATE_BUCKETS]{},gate_cm[8][GATE_BUCKETS]{};uint32_t gate_seen[8][GATE_BUCKETS]{},gate_count[8][GATE_BUCKETS]{};
    Model(int plane,int bits,bool gate):ch(plane),gated(gate){
        na=ch==1?CONTEXTS:CHROMA_CONTEXTS;nin=2+2*na;
        T.init(bits);histories.resize(size_t(na)*8*HISTORIES);
        for(int a=0;a<na;a++)for(int q=0;q<8;q++)for(int s=0;s<HISTORIES;s++){
            const int n0=s&7,n1=(s>>3)&7;const uint32_t p=uint32_t(((2*n1+1)<<22)/(2*(n0+n1)+2));
            histories[(size_t(a)*8+q)*HISTORIES+size_t(s)]=std::clamp<uint32_t>(p,1,(1u<<22)-1)<<10;
        }
        for(int k=0;k<MIXERS;k++){
            const int sets=SELECTOR_SETS[size_t(k)];
            mx[size_t(k)].init(nin,sets,MIXER_RATE);for(int s=0;s<sets;s++)mx[size_t(k)].at(s)[0]=65536;
            mid[size_t(k)].init(MIXERS+1,sets,SECOND_RATE);
            for(int s=0;s<sets;s++)for(int t=0;t<MIXERS;t++)mid[size_t(k)].at(s)[t]=65536/MIXERS;
        }
        fin.init(MIXERS,8,MIXER_RATE);for(int q=0;q<8;q++)for(int t=0;t<MIXERS;t++)fin.at(q)[t]=65536/MIXERS;
        blend.init(BLEND_INPUTS+1,256,MIXER_RATE);
        for(int q=0;q<256;q++)for(int t=0;t<BLEND_INPUTS;t++)blend.at(q)[t]=65536/BLEND_INPUTS;
        a1.init(256*256);a2.init(16*256);for(int t=0;t<3;t++)ax[size_t(t)].init(256*256);ax[3].init(256*64);
    }
};

// Codes one sample of plane ch at (x, y) with model M. nn() returns the
// network's stretch for the next bit, or sets nn.forced to a bit that is not
// coded, and nn.take(bit) follows the bit; io(p, bp) codes bit bp (0 the most
// significant) with probability p/65536 of a one and returns it. Reads only
// samples before (x, y) in plane ch, and planes coded before it through row
// y + 1; writes the sample's network cost.
struct Coder{
    const Planes& S;int ch;
    Coder(const Planes& planes,int plane):S(planes),ch(plane){}
    int px(int pl,int x,int y)const{
        if(x<0||y<0||x>=S.w||y>=S.h)return 0;const size_t j=size_t(y)*S.w+x;
        if(pl==ch||(ch==2&&pl==0))return S.Pc[pl][j];return S.P[pl][j];
    }
    int er(int pl,int x,int y)const{
        if(x<0||y<0||x>=S.w||y>=S.h)return 0;const size_t i=size_t(y)*S.w+x;return int(S.Pc[pl][i])-int(S.mode[pl][i]);
    }
    // across is the run length of W (see Planes).
    int contexts(int x,int y,int across,uint64_t* b)const{
        const size_t i=size_t(y)*S.w+x;
        const int Wv=px(ch,x-1,y),Nv=px(ch,x,y-1),NW=px(ch,x-1,y-1),NE=px(ch,x+1,y-1),WW=px(ch,x-2,y),NN=px(ch,x,y-2);
        const int NEE=px(ch,x+2,y-1),NNN=px(ch,x,y-3),NNE=px(ch,x+1,y-2);
        const int mo=S.mode[ch][i];
        const int big=(Wv<<16)|(Nv<<8)|NW,big2=(NE<<16)|(WW<<8)|NN;
        // Shared by all planes, after the plane's own contexts: the run of W, four
        // linear predictions with the samples coded before this one at (x, y), and
        // the network's mode with its errors (coded sample - mode) around the sample.
        {
            uint64_t* s=b+(ch==1?CONTEXTS:CHROMA_CONTEXTS)-SHARED_CONTEXTS;
            const uint64_t E=(uint64_t(ch)<<6)|32;
            const int cr=ch==1?0:ch==0?px(1,x,y)+1:((px(1,x,y)<<8)|px(0,x,y))+1;
            auto cl=[](int v){return std::clamp(v,0,255);};
            const int g=Wv+Nv-NW,ga=std::abs(g-Wv),gb=std::abs(g-Nv),gc=std::abs(g-NW);
            const int paeth=ga<=gb&&ga<=gc?Wv:gb<=gc?Nv:NW;
            const int eW=er(ch,x-1,y),eN=er(ch,x,y-1),eNW=er(ch,x-1,y-1),eNE=er(ch,x+1,y-1);
            s[0]=hc(E|21,Wv,run_bucket(across),(Nv==Wv)|((NE==Wv)<<1));
            s[1]=hc(E|2,paeth>>1,cr);s[2]=hc(E|3,cl(Nv+NE-NNE)>>1,cr);
            s[3]=hc(E|4,cl(2*Wv-WW)>>1,cr);s[4]=hc(E|5,cl(2*Nv-NN)>>1,cr);
            s[5]=hc(E|29,mo,qe(eW),qe(eN));s[6]=hc(uint64_t(1041+100*ch),mo,qe(eW+eN+(eNW+eNE)/2));
        }
        if(ch==1){
            b[0]=hc(0,0);b[1]=hc(1,Wv);b[2]=hc(2,Nv);b[3]=hc(3,NE);b[4]=hc(4,NW);
            b[5]=hc(5,Wv,Nv);b[6]=hc(6,Wv,NW);b[7]=hc(7,Nv,NE);b[8]=hc(9,Wv,Nv,NE);b[9]=hc(10,big,big2);
            b[10]=hc(14,x);b[11]=hc(15,std::clamp(Wv+Nv-NW,0,255));
            b[12]=hc(16,std::clamp(2*Nv-NN,0,255),std::clamp(2*Wv-WW,0,255));b[13]=hc(17,Nv,NN,NNN);
            b[14]=hc(18,Wv,WW);b[15]=hc(19,NE,NEE);
            return std::abs(Wv-Nv);
        }
        const uint64_t C=uint64_t(ch)<<6;
        const int o1=px(1,x,y),Wo1=px(1,x-1,y),No1=px(1,x,y-1);
        const int pW=std::clamp(Wv+o1-Wo1,0,255),pN=std::clamp(Nv+o1-No1,0,255);
        const int GE=px(1,x+1,y),GS=px(1,x,y+1);
        if(ch==0){
            b[0]=hc(C|0,0);b[1]=hc(C|1,Wv);b[2]=hc(C|2,Nv);b[3]=hc(C|3,o1);b[4]=hc(C|4,o1,Wv);b[5]=hc(C|5,o1,Nv);
            b[6]=hc(C|6,o1,NE);b[7]=hc(C|7,o1,NW);b[8]=hc(C|9,o1,pW);b[9]=hc(C|10,o1,pN);
            b[10]=hc(C|14,mo,o1);b[11]=hc(C|15,mo,Wv);b[12]=hc(C|16,x);
            b[13]=hc(C|18,o1,GE,GS);b[14]=hc(C|20,x,o1>>2);
        }else{
            const int o2=px(0,x,y),Wo2=px(0,x-1,y);
            const int qW=std::clamp(Wv+o2-Wo2,0,255);
            const int RE=px(0,x+1,y),RS=px(0,x,y+1);
            b[0]=hc(C|0,0);b[1]=hc(C|1,Wv);b[2]=hc(C|2,Nv);b[3]=hc(C|3,o1);b[4]=hc(C|4,o2);b[5]=hc(C|5,o1,o2);
            b[6]=hc(C|6,o1,o2,Wv);b[7]=hc(C|7,o1,o2,Nv);b[8]=hc(C|9,o2,qW);b[9]=hc(C|10,o1,pW);
            b[10]=hc(C|14,mo,o1,o2);b[11]=hc(C|15,mo,Wv);
            b[12]=hc(C|16,x);b[13]=hc(C|18,o2,RE,RS);b[14]=hc(C|20,x,o2>>2);
        }
        // Raw (uncoded) own samples around the sample, with G.
        const uint8_t* own=S.P[ch];
        auto rr=[&](int dx,int dy)->int{const int xx=x+dx,yy=y+dy;if(xx<0||yy<0||xx>=S.w)return 0;return own[size_t(yy)*S.w+xx];};
        const int rW=rr(-1,0),rN=rr(0,-1),rNW=rr(-1,-1),rNE=rr(1,-1);const int g=S.P[1][i];
        b[15]=hc(C|22,rW,rN,g);
        b[16]=hc(C|23,(rW==0)|((rW==255)<<1)|((rN==0)<<2)|((rN==255)<<3)|((rNW==0)<<4)|((rNW==255)<<5)|
                ((rNE==0)<<6)|((rNE==255)<<7),g);
        b[17]=hc(C|24,rW,g,Wv);b[18]=hc(C|25,rN,g,Nv);
        return std::abs(Wv-Nv);
    }
    // The network's cost of the six nearest coded samples, in eight buckets.
    int loss(int x,int y)const{
        int64_t t=0;
        auto a=[&](int dx,int dy){const int xx=x+dx,yy=y+dy;if(xx<0||yy<0||xx>=S.w)return;t+=S.cost[ch][size_t(yy)*S.w+xx];};
        a(-1,0);a(0,-1);a(-2,0);a(0,-2);a(-1,-1);a(1,-1);const int bits=int(t>>4);
        return bits<2?0:bits<4?1:bits<7?2:bits<11?3:bits<16?4:bits<23?5:bits<32?6:7;
    }
    template<class NN,class IO>
    int code(Model& M,int x,int y,NN& nn,IO&& io)const{
        const int W=S.w;const size_t i=size_t(y)*W+x;const int na=M.na;
        const int across=x>0?std::min<int>(32,S.across[ch][i-1]):0,down=y>0?std::min<int>(32,S.down[ch][i-size_t(W)]):0;
        uint64_t h[MAX_CONTEXTS];const int grad=contexts(x,y,across,h);
        for(int a=0;a<na;a++)M.T.prefetch(h[a]);
        const int lq=loss(x,y);
        const int mo=S.mode[ch][i],sb=S.scale[ch][i];
        const int Wv=px(ch,x-1,y),Nv=px(ch,x,y-1),NWv=px(ch,x-1,y-1);
        const int grW=std::clamp(Wv+Nv-NWv,0,255);
        const int o1=ch==1?0:px(1,x,y),o2=ch==2?px(0,x,y):0;
        // Weight-set selectors, one per mixer (SELECTOR_SETS sizes).
        std::array<int,MIXERS> fixed{};
        fixed[1]=int(int64_t(x)*64/W);fixed[4]=run_bucket(across)*16+run_bucket(down);
        const int chroma=ch==1?-1:((o1>>5)*8+(o2>>5))*4;
        const int errors=(qe(er(ch,x-1,y))<<4)|qe(er(ch,x,y-1));
        uint32_t* cur[MAX_CONTEXTS];
        for(int a=0;a<na;a++)cur[a]=M.T.claim(h[a]);
        uint64_t nx0[MAX_CONTEXTS],nx1[MAX_CONTEXTS];
        int c0=1,j=1;int64_t nncost=0;
        alignas(16) int32_t xin[MAX_INPUTS],x15[MAX_INPUTS],x13[MAX_INPUTS];
        alignas(16) int32_t fx[MIXER_PAD],gx[MIXER_PAD],hx[MIXER_PAD],h15[MIXER_PAD],h13[MIXER_PAD],g15[MIXER_PAD],g13[MIXER_PAD];
        for(int bp=0;bp<8;bp++){
            if(bp==3){
                const int n0=(c0<<1)&15,n1=n0|1;
                for(int a=0;a<na;a++){
                    nx0[a]=next_hash(h[a],n0);nx1[a]=next_hash(h[a],n1);M.T.prefetch(nx0[a]);M.T.prefetch(nx1[a]);
                }
            }else if(bp==4){
                j=1;const bool one=c0&1;for(int a=0;a<na;a++)cur[a]=M.T.claim(one?nx1[a]:nx0[a]);
            }
            const int st=nn();
            if(nn.forced>=0){const int bit=nn.forced;nn.take(bit);c0=(c0<<1)|bit;j=(j<<1)|bit;continue;}
            const int gq=std::min(GATE_BUCKETS-1,std::abs(st)>>8);
            if(M.gated&&gq>=GATE_MIN_BUCKET&&M.gate_seen[bp][gq]>=GATE_WARMUP&&
               M.gate_cm[bp][gq]*1000>=M.gate_nn[bp][gq]*(1000-GATE_RATIO)&&((M.gate_count[bp][gq]++)&GATE_PROBE)!=0){
                const int pn=squash(st);const int bit=io(pn,bp);
                nncost+=int64_t(16)*65536-log2q(uint32_t(bit?pn:65536-pn));
                nn.take(bit);c0=(c0<<1)|bit;j=(j<<1)|bit;continue;
            }
            // Map rows hold one context's bit paths (context * 256 + c0), so a
            // sample's rows share pages; fetch this bit's rows while it mixes.
            const int conf=std::min(7,std::abs(st)>>9);
            const std::array<int,6> rows={errors*256+c0,std::min(15,grad)*256+c0,Wv*256+c0,Nv*256+c0,mo*256+c0,
                                          (lq*8+conf)*256+c0};
            M.a1.prefetch(rows[0]);M.a2.prefetch(rows[1]);for(int t=0;t<4;t++)M.ax[size_t(t)].prefetch(rows[size_t(t)+2]);
            xin[0]=st;xin[1]=256;int ni=2;
            for(int a=0;a<na;a++){const uint32_t z=cur[a][j];xin[ni++]=(z&0xff00)?st16(z>>16):0;}
            uint32_t* bix[MAX_CONTEXTS];
            for(int a=0;a<na;a++){
                uint32_t* e=&M.histories[(size_t(a)*8+size_t(bp))*HISTORIES+(cur[a][j]&255)];bix[a]=e;xin[ni++]=st22(*e>>10);
            }
            while(ni&3)xin[ni++]=0;
            // Within a nibble the next bit's nodes are j's children: fetch their
            // history entries and the next weight sets while this bit mixes.
            if((bp&3)!=3){
                const uint32_t* next=&M.histories[size_t(bp+1)*HISTORIES];
                for(int a=0;a<na;a++){
                    const uint32_t* row=next+size_t(a)*8*HISTORIES;
                    __builtin_prefetch(row+(cur[a][2*j]&255));__builtin_prefetch(row+(cur[a][2*j+1]&255));
                }
            }
            if(bp<7){
                M.mx[0].prefetch(2*c0);M.mx[0].prefetch(2*c0+1);M.mid[0].prefetch(2*c0);M.mid[0].prefetch(2*c0+1);
                M.mx[5].prefetch((bp+1)*32+std::min(31,sb));
            }
            shifted(xin,ni,x15,x13);
            const int pre=c0-(1<<bp);
            auto match=[&](int v){return int((v>>(8-bp))==pre);};
            std::array<int,MIXERS> sidx=fixed;
            sidx[0]=c0;sidx[2]=chroma<0?bp:chroma+(bp>>1);
            sidx[3]=((match(mo)<<3|match(Wv)<<2|match(Nv)<<1|match(grW))<<3)|bp;sidx[5]=bp*32+std::min(31,sb);
            Mixer::dots(M.mx,sidx,x15,fx);
            for(int k=0;k<MIXERS;k++)hx[k]=fx[k];
            hx[MIXERS]=st;for(int k=MIXERS+1;k<MIXER_PAD;k++)hx[k]=0;
            shifted(hx,MIXER_PAD,h15,h13);
            Mixer::dots(M.mid,sidx,h15,gx);
            for(int k=MIXERS;k<MIXER_PAD;k++)gx[k]=0;
            shifted(gx,MIXER_PAD,g15,g13);
            const int fo=M.fin.dot(g15,bp);const int pm=squash(fo);
            const std::array<int,6> q={M.a1.pp(fo,rows[0]),M.a2.pp(fo,rows[1]),M.ax[0].pp(fo,rows[2]),
                                       M.ax[1].pp(fo,rows[3]),M.ax[2].pp(fo,rows[4]),M.ax[3].pp(fo,rows[5])};
            const int sum=pm+q[0]+2*q[1]+q[2]+q[3]+q[4]+q[5];
            // The fixed average of the maps, and a mixer of their stretches selected by
            // the bit, the network's recent loss and its confidence.
            alignas(16) int32_t bx[8]={fo,0,0,0,0,0,0,256},b15[8],b13[8];
            for(int t=0;t<6;t++)bx[t+1]=stretch16(q[size_t(t)]);
            shifted(bx,8,b15,b13);
            const int bsel=(lq*8+bp)*4+std::min(3,conf>>1);
            const int bo=M.blend.dot(b15,bsel),pb=squash(bo);
            const int p=std::clamp((std::clamp((sum+4)/8,1,65535)+pb+1)/2,1,65535);
            const int bit=io(p,bp);
            {const int pn=squash(st);const int64_t cn=int64_t(16)*65536-log2q(uint32_t(bit?pn:65536-pn));nncost+=cn;
             const int64_t cm=int64_t(16)*65536-log2q(uint32_t(bit?p:65536-p));
             M.gate_nn[bp][gq]+=cn-(M.gate_nn[bp][gq]>>GATE_DECAY);M.gate_cm[bp][gq]+=cm-(M.gate_cm[bp][gq]>>GATE_DECAY);
             if(M.gate_seen[bp][gq]<(1u<<30))M.gate_seen[bp][gq]++;}
            nn.take(bit);
            int em[MIXERS],ex[MIXERS];
            for(int k=0;k<MIXERS;k++){em[k]=((bit<<16)-squash(gx[k]))>>4;ex[k]=((bit<<16)-squash(fx[k]))>>4;}
            Mixer::updates(M.mid,sidx,h13,em);Mixer::updates(M.mx,sidx,x13,ex);
            M.fin.update(g13,bp,((bit<<16)-pm)>>4);M.blend.update(b13,bsel,((bit<<16)-pb)>>4);
            M.a1.update(bit);M.a2.update(bit);for(auto& a:M.ax)a.update(bit);
            for(int a=0;a<na;a++){bump(cur[a][j],bit);bump32(*bix[a],bit);}
            c0=(c0<<1)|bit;j=(j<<1)|bit;
        }
        S.cost[ch][i]=uint16_t(std::min<int64_t>(nncost>>12,65535));
        const int v=c0&255;
        S.across[ch][i]=uint8_t(x>0&&v==S.Pc[ch][i-1]?std::min(255,S.across[ch][i-1]+1):1);
        S.down[ch][i]=uint8_t(y>0&&v==S.Pc[ch][i-size_t(W)]?std::min(255,S.down[ch][i-size_t(W)]+1):1);
        return v;
    }
};

}  // namespace v5
