// Integer numerical reference, C++17 with 128-bit integer support. Feature sets 3
// and 4 add binary64 least-squares steps (see LocalFit).
#include "arithmetic.hpp"

#include <cfloat>

#include <atomic>
#include <exception>
#include <memory>
#include <optional>
#include <mutex>
#include <thread>
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif
#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif
#if defined(__clang__)
#pragma STDC FP_CONTRACT OFF
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#endif

constexpr int64_t Q=65536, GUIDE_Q=16777216, LIMIT=512*Q;
static int64_t floor_div(int64_t n,int64_t d){return n/d-(n%d<0);}
static int64_t rounded(int64_t n,int64_t d){return floor_div(n+d/2,d);}
struct Reader{
    const Bytes& b;size_t pos=0;
    uint32_t u32(){
        require(pos+4<=b.size(),"truncated configuration");
        uint32_t v=uint32_t(b[pos])|(uint32_t(b[pos+1])<<8)|
                   (uint32_t(b[pos+2])<<16)|(uint32_t(b[pos+3])<<24);
        pos+=4;return v;
    }
    std::vector<int32_t> i32(size_t n){
        require(n<=16384,"invalid parameter count");std::vector<int32_t> a(n);
        for(auto& value:a){
            uint32_t v=u32();int64_t s=(v>>31)?-1-int64_t(UINT32_MAX-v):int64_t(v);
            require(s>-10000*Q&&s<10000*Q,"invalid parameter");value=int32_t(s);
        }
        return a;
    }
    std::vector<int64_t> i64(size_t n){
        require(n<=16384,"invalid parameter count");std::vector<int64_t> a(n);
        for(auto& value:a){
            uint64_t v=u32();v|=uint64_t(u32())<<32;
            value=(v>>63)?-1-int64_t(UINT64_MAX-v):int64_t(v);
            require(value>-10000*GUIDE_Q&&value<10000*GUIDE_Q,"invalid parameter");
        }
        return a;
    }
};
const std::array<std::array<int,2>,12> far_taps={{{0,-3},{-3,0},{-1,-3},{-1,3},
    {-3,-1},{-3,1},{-2,-3},{-2,3},{-3,-2},{-3,2},{-3,-3},{-3,3}}};
// Least-squares windows: rows y-rows..y-1 at columns x-cols..x+cols, and row y at
// x-cols..x-1. Inputs are 12 own causal samples and, from each decoded plane,
// its 3x3 (prior 9) or the sample and its four neighbours (prior 5). The ridge
// scales the mean Gram diagonal. Bordered windows use solve_bordered, the others solve.
struct FitWindow{int rows,cols,prior;double ridge;bool bordered;};
constexpr int MAX_FITS=2,FIT_INPUTS=30,FIT_REACH=12;
constexpr FitWindow fit_windows[3]={
    {12,12,9,1.0/1048576,false},{12,12,5,1.0/1048576,true},{4,4,5,1.0/8192,true}};
static_assert(fit_windows[0].cols<=FIT_REACH&&fit_windows[1].cols<=FIT_REACH&&
              fit_windows[2].cols<=FIT_REACH,"fit reach");
// Feature set 3 uses window 0; feature set 4 windows 1 and 2.
using FitWindows=std::array<const FitWindow*,MAX_FITS>;
static FitWindows set_windows(uint32_t set){
    if(set==3)return {&fit_windows[0],nullptr};
    if(set==4)return {&fit_windows[1],&fit_windows[2]};
    return {nullptr,nullptr};
}
static int window_count(const FitWindows& w){return int(w[0]!=nullptr)+int(w[1]!=nullptr);}
// Fit-error positions as (dy, dx); missing positions read as zero.
const std::array<std::array<int,2>,4> fit_errors={{{0,-1},{-1,0},{-1,-1},{-1,1}}};
// Fit, spread, own fit errors, then each decoded plane's fit error at the sample.
static int fit_features(int ch){return 2+int(fit_errors.size())+(ch==1?0:ch==0?1:2);}
// Most first-layer inputs are u * 256 with small |u|. When every first-layer
// weight is a multiple of 2^shift (quantized training makes them so), inputs with
// |u| <= unit_limit take an exact int16 x int16 -> int32 path (narrow,
// transposed): unit_limit keeps each output's sum of |w >> shift| * |u| within
// int32. The rest use the int64 path on wide, the transposed weights. Rows are
// padded to a multiple of 8 outputs.
struct Plane{
    int d,base;std::vector<int64_t> guide,affine;
    std::vector<int32_t> w0,b0,w1,b1,wh,bh;
    bool narrow_ok=false,affine_ok=false;int shift=0,padded=0,head_padded=0,unit_limit=0;
    std::vector<int16_t> narrow;
    std::vector<int32_t> wide,w1t,wht;std::vector<int64_t> bias0,bias1,biash,affine_padded;
    // b * Q + Q / 2 per output, padded outputs Q / 2 (see wide_block).
    static std::vector<int64_t> biases(const std::vector<int32_t>& b,int padded){
        std::vector<int64_t> v(size_t(padded),Q/2);
        for(size_t i=0;i<b.size();i++)v[i]=int64_t(b[i])*Q+Q/2;
        return v;
    }
    static std::vector<int32_t> transpose(const std::vector<int32_t>& w,int ni,int no,int padded){
        std::vector<int32_t> t(size_t(ni)*padded,0);
        for(int i=0;i<no;i++)for(int j=0;j<ni;j++)t[size_t(j)*padded+i]=w[size_t(i)*ni+j];
        return t;
    }
    void prepare(int hidden,int heads){
        padded=(hidden+7)/8*8;head_padded=(heads+7)/8*8;shift=31;
        w1t=transpose(w1,hidden,hidden,padded);wht=transpose(wh,hidden,heads,head_padded);
        for(int32_t v:w0)if(v)shift=std::min(shift,__builtin_ctz(uint32_t(v)));
        if(shift==31)shift=0;
        narrow.assign(size_t((d+7)/8*8)*padded,0);wide.assign(size_t(d)*padded,0);narrow_ok=true;
        int64_t most=1;
        for(int i=0;i<hidden;i++){
            int64_t total=0;
            for(int j=0;j<d;j++){
                const int32_t v=w0[size_t(i)*d+j];const int32_t q=v/(int32_t(1)<<shift);
                if(q<-32767||q>32767)narrow_ok=false;
                total+=std::abs(int64_t(q));
                narrow[size_t(j)*padded+i]=int16_t(std::clamp(q,-32767,32767));wide[size_t(j)*padded+i]=v;
            }
            most=std::max(most,total);
        }
        // shift <= 23 keeps the narrow sum times 2^(8 + shift) within int64.
        unit_limit=narrow_ok&&shift<=23?int(std::min<int64_t>(32767,INT32_MAX/most)):0;
        narrow_ok=unit_limit>0;
        bias0=biases(b0,padded);bias1=biases(b1,padded);biash=biases(bh,head_padded);
        affine_padded=affine;affine_padded.resize(size_t((d+7)/8*8),0);
        affine_ok=true;
        for(int64_t a:affine)if(a<-INT64_MAX/(32767*128)||a>INT64_MAX/(32767*128))affine_ok=false;
    }
};
struct Model{
    int w,h,hidden,k,fits=0;bool context,far=false,level=false;FitWindows windows{};
    std::array<Plane,3> plane;
    explicit Model(const Bytes& bytes){
        Reader r{bytes};uint32_t magic=r.u32();
        require(magic==0x32494e4e||magic==0x33494e4e,"invalid integer configuration");
        w=r.u32();h=r.u32();hidden=r.u32();k=r.u32();auto c=r.u32();
        require(w>0&&h>0&&uint64_t(w)*h<=16777216,"invalid canvas");
        require(hidden>0&&hidden<=64&&k>0&&k<=4&&c<=1,"invalid architecture");context=c;
        // Version 3 configurations name a feature set; set 1 adds far causal taps
        // and set 2 also the plane's level and the sample's row, column and diagonal
        // parity. Sets 3 and 4 also add one and two local least-squares fits.
        if(magic==0x33494e4e){
            uint32_t set=r.u32();require(set>=1&&set<=4&&context,"invalid feature set");
            far=true;level=set>=2;windows=set_windows(set);fits=window_count(windows);
        }
        const int base_dims[3]={21,14,30},extra[3]={24,13,34};
        for(int ch:{1,0,2}){
            Plane& p=plane[ch];p.d=r.u32();p.base=r.u32();
            require(p.base==base_dims[ch]&&
                    p.d==p.base+(context?extra[ch]:0)+(far?int(far_taps.size()):0)+
                    (level?4:0)+fits*fit_features(ch),
                    "invalid feature dimensions");
            p.guide=r.i64(p.base);p.affine=r.i64(p.d);
            p.w0=r.i32(hidden*p.d);p.b0=r.i32(hidden);
            p.w1=r.i32(hidden*hidden);p.b1=r.i32(hidden);
            p.wh=r.i32(3*k*hidden);p.bh=r.i32(3*k);p.prepare(hidden,3*k);
        }
        require(r.pos==bytes.size(),"trailing configuration bytes");
    }
};
const std::array<std::array<int,2>,12> causal={{{0,-1},{-1,0},{-1,-1},{-1,1},
    {0,-2},{-2,0},{-2,-1},{-2,1},{-1,-2},{-1,2},{-2,-2},{-2,2}}};
const std::array<std::array<int,2>,9> guides={{{0,0},{0,-1},{-1,0},{-1,-1},
    {-1,1},{0,1},{1,0},{1,-1},{1,1}}};
const std::array<std::array<int,2>,5> cross={{{0,0},{0,-1},{-1,0},{0,1},{1,0}}};
struct Canvas{
    int w,h;
    std::array<Bytes,3> pixels;
    std::array<std::vector<int16_t>,3> errors;
    // Fit and spread planes per least-squares window.
    std::array<std::array<std::vector<int16_t>,3>,MAX_FITS> fits,spreads;
    Canvas(int width,int height,int count=0):w(width),h(height){
        for(auto& p:pixels)p.resize(size_t(w)*h,0);
        for(auto& e:errors)e.resize(size_t(w)*h,0);
        for(int s=0;s<count;s++){
            for(auto& f:fits[s])f.resize(size_t(w)*h,0);
            for(auto& v:spreads[s])v.resize(size_t(w)*h,0);
        }
    }
    int location(int x,int y,int dx,int dy,bool missing_zero)const{
        x+=dx;y+=dy;
        if(x<0||y<0||x>=w||y>=h){
            if(missing_zero)return -1;
            x=std::clamp(x,0,w-1);y=std::clamp(y,0,h-1);
        }
        return y*w+x;
    }
    int sample(int ch,int x,int y,int dx,int dy,bool zero=true)const{
        int i=location(x,y,dx,dy,zero);return i<0?0:pixels[ch][i];
    }
    int error(int ch,int x,int y,int dx,int dy,bool zero=true)const{
        int i=location(x,y,dx,dy,zero);return i<0?0:errors[ch][i];
    }
    int difference(int a,int b,int x,int y,int dx,int dy,bool zero)const{
        int i=location(x,y,dx,dy,zero);
        return i<0?0:int(pixels[a][i])-int(pixels[b][i]);
    }
};

// Feature sets 3 and 4 add causal-window ridge least-squares fits per sample. Window
// sums are exact integers. The solve uses IEEE-754 binary64 with only correctly
// rounded +, -, *, / and sqrt in a fixed order and no contraction, so every
// conforming build computes the same bits.
#if defined(__FAST_MATH__)
#error "fast-math changes least-squares fits"
#endif
static_assert(std::numeric_limits<double>::is_iec559,"binary64 arithmetic required");
static_assert(FLT_EVAL_METHOD==0,"double expressions must round to double");
// Short elementwise loops vectorize and fixed-size solve loops unroll; neither
// changes any operation or its order.
#if defined(__clang__)
#define PAIRED _Pragma("clang loop vectorize_width(2) interleave_count(1)")
#define UNROLLED _Pragma("clang loop unroll(full)")
#elif defined(__GNUC__)
#define PAIRED
#define UNROLLED _Pragma("GCC unroll 32")
#else
#define PAIRED
#define UNROLLED
#endif
// Own causal samples (zero outside) and decoded planes' taps (edge clamped).
template<class T>
static int fit_inputs(const Canvas& c,const FitWindow& win,int ch,int x,int y,T* f){
    int n=0;const std::array<int,2>* taps=win.prior==9?guides.data():cross.data();
    if(x>=2&&x+2<c.w&&y>=2&&y+1<c.h){
        const size_t i=size_t(y)*c.w+x;const long w=c.w;
        const uint8_t* own=&c.pixels[ch][i];
        for(auto d:causal)f[n++]=own[d[0]*w+d[1]];
        for(int prior:{1,0}){
            if(ch==1||(prior==0&&ch!=2))break;
            const uint8_t* p=&c.pixels[prior][i];
            for(int k=0;k<win.prior;k++)f[n++]=p[taps[k][0]*w+taps[k][1]];
        }
        return n;
    }
    for(auto d:causal)f[n++]=c.sample(ch,x,y,d[1],d[0]);
    if(ch!=1)for(int k=0;k<win.prior;k++)f[n++]=c.sample(1,x,y,taps[k][1],taps[k][0],false);
    if(ch==2)for(int k=0;k<win.prior;k++)f[n++]=c.sample(0,x,y,taps[k][1],taps[k][0],false);
    return n;
}
static int fit_count(const FitWindow& win,int ch){
    return 12+(ch!=1?win.prior:0)+(ch==2?win.prior:0);
}
// Window sums are the lower triangle of the sum of outer products of (inputs,
// target): the Gram rows, then input times target, then target squared. Row a
// starts at row_offset(a) and is padded to a multiple of 4 entries so rows update
// in whole vectors; padding holds products above the diagonal and is never read.
static constexpr int row_offset(int a){int o=0;for(int r=0;r<a;r++)o+=(r+4)/4*4;return o;}
static int fit_size(const FitWindow& win,int ch){return row_offset(fit_count(win,ch)+1);}
constexpr int FIT_VECTOR=(FIT_INPUTS+1+3)/4*4;
// acc += outer(p) - outer(q) for M-entry vectors zero padded to FIT_VECTOR.
// Samples are at most 255, so int16 x int16 -> int32 products are exact, and
// integer sums are exact in any order.
template<int M>
static void outer_update(int32_t* acc,const int16_t* p,const int16_t* q){
    UNROLLED
    for(int a=0;a<M;a++){
        int32_t* row=acc+row_offset(a);const int16_t pa=p[a],qa=q[a];
        UNROLLED
        for(int k=0;k<=a;k+=4){
#if defined(__ARM_NEON)
            int32x4_t v=vld1q_s32(row+k);
            v=vmlal_n_s16(v,vld1_s16(p+k),pa);v=vmlsl_n_s16(v,vld1_s16(q+k),qa);
            vst1q_s32(row+k,v);
#else
            for(int l=k;l<k+4;l++)row[l]+=int32_t(pa)*p[l]-int32_t(qa)*q[l];
#endif
        }
    }
}
// One window step: dst = src + outer(cp) - outer(cq), then
// total += dst - old + outer(lp) - outer(lq). dst may be src.
template<int M>
static void window_step(int32_t* total,int32_t* dst,const int32_t* src,const int16_t* cp,
                        const int16_t* cq,const int32_t* old,const int16_t* lp,const int16_t* lq){
    UNROLLED
    for(int a=0;a<M;a++){
        const int o=row_offset(a);
        UNROLLED
        for(int k=0;k<=a;k+=4){
#if defined(__ARM_NEON)
            int32x4_t v=vld1q_s32(src+o+k);
            v=vmlal_n_s16(v,vld1_s16(cp+k),cp[a]);v=vmlsl_n_s16(v,vld1_s16(cq+k),cq[a]);
            vst1q_s32(dst+o+k,v);
            int32x4_t t=vaddq_s32(vld1q_s32(total+o+k),vsubq_s32(v,vld1q_s32(old+o+k)));
            t=vmlal_n_s16(t,vld1_s16(lp+k),lp[a]);t=vmlsl_n_s16(t,vld1_s16(lq+k),lq[a]);
            vst1q_s32(total+o+k,t);
#else
            for(int l=k;l<k+4;l++){
                const int32_t v=src[o+l]+int32_t(cp[a])*cp[l]-int32_t(cq[a])*cq[l];dst[o+l]=v;
                total[o+l]+=v-old[o+l]+int32_t(lp[a])*lp[l]-int32_t(lq[a])*lq[l];
            }
#endif
        }
    }
}
// The packed lower triangle of the first rows rows of window sums.
static void unpack(const int32_t* sums,int rows,double* work){
    for(int a=0,k=0,o=0;a<rows;o+=(a+4)/4*4,a++)for(int b=0;b<=a;b++)work[k++]=double(sums[o+b]);
}
// Ridge solve of the window sums and the fit at inputs f. The operation order
// is normative (see above); N only fixes loop bounds at compile time.
template<int N>
static void solve(const int32_t* sums,const int32_t* f,int64_t count,double ridge_scale,double& p,
                  double& residual){
    constexpr int tri=N*(N+1)/2;
    double work[tri],lower[N*N],upper[N*N],D[N],col[N],b[N],z[N],beta[N];
    unpack(sums,N,work);
    for(int k=0;k<N;k++)b[k]=double(sums[row_offset(N)+k]);
    const double tt=double(sums[row_offset(N)+N]);
    double trace=0;for(int a=0;a<N;a++)trace+=work[a*(a+1)/2+a];
    const double ridge=std::max(1.0,trace/N)*ridge_scale;
    for(int a=0;a<N;a++)work[a*(a+1)/2+a]+=ridge;
    // Right-looking LDL^T on the packed lower triangle; lower holds L by
    // columns and upper by rows.
    UNROLLED
    for(int j=0;j<N;j++){
        const double d=work[j*(j+1)/2+j];
        if(!(d>0))return;
        D[j]=d;
        for(int i=j+1;i<N;i++){
            col[i]=work[i*(i+1)/2+j];const double l=col[i]/d;
            lower[j*N+i]=l;upper[i*N+j]=l;
        }
        UNROLLED
        for(int i=j+1;i<N;i++){
            const double l=lower[j*N+i];double* r=&work[i*(i+1)/2];
            for(int k=j+1;k<=i;k++)r[k]-=l*col[k];
        }
    }
    for(int k=0;k<N;k++)z[k]=b[k];
    UNROLLED
    for(int k=0;k<N;k++){const double v=z[k];for(int i=k+1;i<N;i++)z[i]-=lower[k*N+i]*v;}
    for(int k=0;k<N;k++)z[k]/=D[k];
    UNROLLED
    for(int k=N-1;k>=0;k--){
        const double v=z[k];beta[k]=v;for(int i=0;i<k;i++)z[i]-=upper[k*N+i]*v;
    }
    double explained=0;p=0;
    for(int i=0;i<N;i++){p+=beta[i]*f[i];explained+=beta[i]*b[i];}
    residual=(tt-explained)/double(std::max<int64_t>(1,count));
}
// The same fit from one LDL^T of the bordered matrix [[A, b], [b^T, tt]], which
// is the sums' packed layout: its last pivot is tt - b^T A^-1 b. Row f^T is
// eliminated alongside, leaving -f^T A^-1 b. Each pivot is inverted once. This
// order is normative for bordered windows.
template<int N>
static void solve_bordered(const int32_t* sums,const int32_t* f,int64_t count,double ridge_scale,
                           double& p,double& residual){
    constexpr int M=N+1,tri=M*(M+1)/2;
    double work[tri],col[M],g[M];
    unpack(sums,M,work);
    double trace=0;for(int a=0;a<N;a++)trace+=work[a*(a+1)/2+a];
    const double ridge=std::max(1.0,trace/N)*ridge_scale;
    for(int a=0;a<N;a++)work[a*(a+1)/2+a]+=ridge;
    for(int k=0;k<N;k++)g[k]=f[k];
    g[N]=0;
    UNROLLED
    for(int j=0;j<N;j++){
        const double d=work[j*(j+1)/2+j];
        if(!(d>0))return;
        const double inverse=1/d;
        for(int i=j+1;i<M;i++)col[i]=work[i*(i+1)/2+j];
        UNROLLED
        for(int i=j+1;i<M;i++){
            const double l=col[i]*inverse;double* r=&work[i*(i+1)/2];
            PAIRED
            for(int k=j+1;k<=i;k++)r[k]-=l*col[k];
        }
        const double l=g[j]*inverse;
        PAIRED
        for(int k=j+1;k<M;k++)g[k]-=l*col[k];
    }
    p=-g[N];
    residual=work[tri-1]/double(std::max<int64_t>(1,count));
}
// solve_bordered for two windows at once, lane l from sums[l], f[l], count[l] and
// ridge[l]. Each entry gets the same updates in the same j order as above, so the
// bits match; only the schedule differs: left-looking, two lanes per operation,
// and columns two at a time, column k+1 taking its updates j < k alongside
// column k's.
typedef double Lanes __attribute__((vector_size(16)));
template<int N>
static void solve_bordered2(const int32_t* const* sums,const int32_t* const* f,const int64_t* count,
                            const double* ridge_scale,double* p,double* residual){
    constexpr int M=N+1;
    // A[a][a+1] takes padding when rows convert in pairs.
    Lanes A[M][M+1],L[N][M],C[M][N],G[N];
    // Whole-vector stores: lane-by-lane stores stall the matrix loads that follow.
    UNROLLED
    for(int a=0;a<M;a++){
        const int32_t *s0=sums[0]+row_offset(a),*s1=sums[1]+row_offset(a);
#if defined(__ARM_NEON)
        UNROLLED
        for(int b=0;b<=a;b+=2){
            const float64x2_t x=vcvtq_f64_s64(vmovl_s32(vld1_s32(s0+b)));
            const float64x2_t y=vcvtq_f64_s64(vmovl_s32(vld1_s32(s1+b)));
            A[a][b]=Lanes(vzip1q_f64(x,y));A[a][b+1]=Lanes(vzip2q_f64(x,y));
        }
#else
        for(int b=0;b<=a;b++)A[a][b]=Lanes{double(s0[b]),double(s1[b])};
#endif
    }
    for(int l=0;l<2;l++){
        double trace=0;for(int a=0;a<N;a++)trace+=A[a][a][l];
        const double ridge=std::max(1.0,trace/N)*ridge_scale[l];
        for(int a=0;a<N;a++)A[a][a][l]+=ridge;
    }
    bool live[2]={true,true};
    // Pivots column k from acc (rows k..M-1) and g; false once both lanes are done.
    auto finish=[&](int k,const Lanes* acc,Lanes g)->bool{
        if(k==N){
            for(int l=0;l<2;l++)if(live[l]){
                p[l]=-g[l];residual[l]=acc[N][l]/double(std::max<int64_t>(1,count[l]));
            }
            return false;
        }
        const Lanes d=acc[k];
        for(int l=0;l<2;l++)live[l]=live[l]&&d[l]>0;
        if(!live[0]&&!live[1])return false;
        const Lanes inverse=1/d;
        for(int i=k+1;i<M;i++){C[i][k]=acc[i];L[k][i]=acc[i]*inverse;}
        G[k]=g*inverse;
        return true;
    };
    UNROLLED
    for(int k=0;k<M;k+=2){
        const bool two=k+1<M;
        Lanes acc0[M],acc1[M],g0={0.0,0.0},g1={0.0,0.0};
        for(int i=k;i<M;i++)acc0[i]=A[i][k];
        if(two)for(int i=k+1;i<M;i++)acc1[i]=A[i][k+1];
        if(k<N)g0=Lanes{double(f[0][k]),double(f[1][k])};
        if(k+1<N)g1=Lanes{double(f[0][k+1]),double(f[1][k+1])};
        UNROLLED
        for(int j=0;j<k;j++){
            const Lanes c0=C[k][j];
            acc0[k]-=L[j][k]*c0;g0-=G[j]*c0;
            if(two){
                const Lanes c1=C[k+1][j];g1-=G[j]*c1;
                UNROLLED
                for(int i=k+1;i<M;i++){const Lanes l=L[j][i];acc0[i]-=l*c0;acc1[i]-=l*c1;}
            }
        }
        if(!finish(k,acc0,g0))return;
        if(two){
            const Lanes c1=C[k+1][k];
            UNROLLED
            for(int i=k+1;i<M;i++)acc1[i]-=L[k][i]*c1;
            g1-=G[k]*c1;
            if(!finish(k+1,acc1,g1))return;
        }
    }
}
// One row of fits in column order. predict(x) reads row y only left of x and
// rows above only through column x+cols. Column sums over the window rows go
// to sums (width x fit_size); given the row above's sums they update exactly.
// total holds the window's sums: whole columns x-cols..x+cols over the rows
// above plus row y's samples x-cols..x-1.
struct LocalFit{
    using Sample=std::array<int16_t,FIT_VECTOR>;
    const Canvas& c;const FitWindow& win;int ch,y,n,m,rows;int32_t* sums;const int32_t* above;
    std::vector<int32_t> total,zero,scratch;
    std::array<int32_t,FIT_INPUTS> f{};Sample g{},h{},cp{},cq{},lp{},lq{};const Sample none{};
    LocalFit(const Canvas& canvas,const FitWindow& fit_window,int channel,int row,int32_t* row_sums,
             const int32_t* above_sums)
        :c(canvas),win(fit_window),ch(channel),y(row),sums(row_sums),
         above(row>0?above_sums:nullptr){
        n=fit_count(win,ch);m=fit_size(win,ch);
        rows=y-std::max(0,y-win.rows);
        total.assign(m,0);zero.assign(m,0);scratch.assign(m,0);
        for(int x=0;x<=std::min(c.w-1,win.cols);x++){
            const int32_t* s=column(x);for(int k=0;k<m;k++)total[k]+=s[k];
        }
    }
    // acc += outer(sample (px,py)) - outer(sample (qx,qy)); qy<0 means none.
    void update(int32_t* acc,int px,int py,int qx,int qy){
        sample(g,px,py);
        if(qy>=0)sample(h,qx,qy);else h.fill(0);
        switch(n){
            case 12:outer_update<13>(acc,g.data(),h.data());break;
            case 17:outer_update<18>(acc,g.data(),h.data());break;
            case 21:outer_update<22>(acc,g.data(),h.data());break;
            case 22:outer_update<23>(acc,g.data(),h.data());break;
            default:outer_update<31>(acc,g.data(),h.data());break;
        }
    }
    // Inputs then target at (x, y); entries past them stay zero.
    void sample(Sample& v,int x,int y)const{
        fit_inputs(c,win,ch,x,y,v.data());v[n]=c.pixels[ch][size_t(y)*c.w+x];
    }
    const int32_t* column(int x){
        int32_t* s=&sums[size_t(x)*m];
        if(above){
            std::copy(&above[size_t(x)*m],&above[size_t(x+1)*m],s);
            update(s,x,y-1,x,y-1-win.rows);
        }else{
            std::fill(s,s+m,0);for(int r=y-rows;r<y;r++)update(s,x,r,0,-1);
        }
        return s;
    }
    // Moves the window to column x: total holds its sums and f the inputs at x.
    // Returns the window's sample count. One pass adds column x+cols (built from
    // the row above's), drops column x-1-cols and moves row y's sample x-1 in and
    // x-1-cols out; absent terms read zeros.
    int64_t prepare(int x){
        if(x>0){
            const int cx=x+win.cols,dx=x-1-win.cols;
            const int32_t* old=dx>=0?&sums[size_t(dx)*m]:zero.data();
            const Sample *p=&none,*q=&none;
            int32_t* dst=scratch.data();const int32_t* src=zero.data();
            if(cx<c.w){
                dst=&sums[size_t(cx)*m];
                if(above){
                    src=&above[size_t(cx)*m];sample(cp,cx,y-1);p=&cp;
                    if(y-1-win.rows>=0){sample(cq,cx,y-1-win.rows);q=&cq;}
                }else src=column(cx);
            }
            sample(lp,x-1,y);const Sample* l=&none;
            if(dx>=0){sample(lq,dx,y);l=&lq;}
            const int16_t *a=p->data(),*b=q->data(),*e=lp.data(),*z=l->data();int32_t* t=total.data();
            switch(n){
                case 12:window_step<13>(t,dst,src,a,b,old,e,z);break;
                case 17:window_step<18>(t,dst,src,a,b,old,e,z);break;
                case 21:window_step<22>(t,dst,src,a,b,old,e,z);break;
                case 22:window_step<23>(t,dst,src,a,b,old,e,z);break;
                default:window_step<31>(t,dst,src,a,b,old,e,z);break;
            }
        }
        fit_inputs(c,win,ch,x,y,f.data());
        return int64_t(rows)*(std::min(c.w-1,x+win.cols)-std::max(0,x-win.cols)+1)+
               (x-std::max(0,x-win.cols));
    }
    // The prediction in eighths of a sample and 16 times the window fit RMS.
    void store(double p,double residual,int16_t& fit,int16_t& spread)const{
        if(!(p>-1e6&&p<1e6))p=f[0];
        if(!(residual>0&&residual<1e9))residual=0;
        fit=int16_t(std::clamp(std::floor(8*p+0.5),0.0,2040.0));
        spread=int16_t(std::min(4095.0,std::floor(16*std::sqrt(residual)+0.5)));
    }
    void predict(int x,int16_t& fit,int16_t& spread){
        const int64_t count=prepare(x);
        double p=f[0],residual=0;const double r=win.ridge;const int32_t* s=total.data();
        if(win.bordered)switch(n){
            case 12:solve_bordered<12>(s,f.data(),count,r,p,residual);break;
            case 17:solve_bordered<17>(s,f.data(),count,r,p,residual);break;
            default:solve_bordered<22>(s,f.data(),count,r,p,residual);break;
        }else switch(n){
            case 12:solve<12>(s,f.data(),count,r,p,residual);break;
            case 21:solve<21>(s,f.data(),count,r,p,residual);break;
            default:solve<30>(s,f.data(),count,r,p,residual);break;
        }
        store(p,residual,fit,spread);
    }
};
// Two bordered windows with the same inputs (feature set 4) solve together.
static bool paired(const FitWindows& w){
    return w[0]&&w[1]&&w[0]->bordered&&w[1]->bordered&&w[0]->prior==w[1]->prior;
}
// a.predict(x, fit[0], spread[0]) and b.predict(x, fit[1], spread[1]), bit for bit.
static void predict_pair(LocalFit& a,LocalFit& b,int x,int16_t& fit0,int16_t& spread0,
                         int16_t& fit1,int16_t& spread1){
    const int64_t count[2]={a.prepare(x),b.prepare(x)};
    const int32_t* sums[2]={a.total.data(),b.total.data()};
    const int32_t* f[2]={a.f.data(),b.f.data()};
    const double ridge[2]={a.win.ridge,b.win.ridge};
    double p[2]={double(a.f[0]),double(b.f[0])},residual[2]={0,0};
    switch(a.n){
        case 12:solve_bordered2<12>(sums,f,count,ridge,p,residual);break;
        case 17:solve_bordered2<17>(sums,f,count,ridge,p,residual);break;
        default:solve_bordered2<22>(sums,f,count,ridge,p,residual);break;
    }
    a.store(p[0],residual[0],fit0,spread0);b.store(p[1],residual[1],fit1,spread1);
}
// 8 log2(1 + spread/16), linear between powers of two.
static int spread_feature(int spread){
    uint32_t v=uint32_t(spread)+16;int e=31-__builtin_clz(v);
    return 8*(e-4)+int(((v-(uint32_t(1)<<e))*8)>>e);
}
struct Features{
    std::array<int32_t,128> x{};int n=0,anchor=0,prediction=0;int64_t raw=0;
    void add_q(int64_t v){require(n<128,"feature budget");x[n++]=int32_t(v);}
    void add(int v){add_q(v*(Q/32));}
};
static bool past(int dy,int dx){return dy<0||(dy==0&&dx<0);}
// Every tap the features read, (dx, dy) in [-3, 3] x [-3, 1], lies inside the
// canvas, so taps index directly; elsewhere they go through Canvas's edge rules.
static bool inside(const Canvas& c,int x,int y){return x>=3&&x+3<c.w&&y>=3&&y+1<c.h;}
template<bool INSIDE>
struct Taps{
    const Canvas& c;int x,y;size_t i;
    Taps(const Canvas& canvas,int px,int py):c(canvas),x(px),y(py),i(size_t(py)*canvas.w+px){}
    long at(int dx,int dy)const{return long(i)+long(dy)*c.w+dx;}
    int sample(int ch,int dx,int dy,bool zero=true)const{
        if constexpr(INSIDE)return c.pixels[ch][at(dx,dy)];else return c.sample(ch,x,y,dx,dy,zero);
    }
    int error(int ch,int dx,int dy,bool zero=true)const{
        if constexpr(INSIDE)return c.errors[ch][at(dx,dy)];else return c.error(ch,x,y,dx,dy,zero);
    }
    int difference(int a,int b,int dx,int dy,bool zero)const{
        if constexpr(INSIDE)return int(c.pixels[a][at(dx,dy)])-int(c.pixels[b][at(dx,dy)]);
        else return c.difference(a,b,x,y,dx,dy,zero);
    }
    int location(int dx,int dy)const{
        if constexpr(INSIDE)return int(at(dx,dy));else return c.location(x,y,dx,dy,true);
    }
};
// Reads pixels only, never errors, so every prediction (and error) of a known
// image can be computed before any context feature.
template<bool INSIDE>
static void base_features(const Model& m,const Taps<INSIDE>& t,int ch,Features& f){
    const Plane& p=m.plane[ch];
    int west=t.sample(ch,-1,0),north=t.sample(ch,0,-1),nw=t.sample(ch,-1,-1);
    int green=t.sample(1,0,0);
    f.anchor=ch==1?std::clamp(west+north-nw,std::min(west,north),std::max(west,north)):green;
    f.add(1);
    if(ch==1){
        for(auto d:causal)f.add(t.sample(ch,d[1],d[0])-f.anchor);
        f.add(std::clamp(west+north-nw,0,255)-f.anchor);
    }else{
        for(auto d:causal)f.add(t.difference(ch,1,d[1],d[0],true));
        for(size_t i=1;i<guides.size();i++){
            auto d=guides[i];f.add(t.sample(1,d[1],d[0],past(d[0],d[1]))-green);
        }
        if(ch==2)for(auto d:guides)
            f.add(t.difference(0,1,d[1],d[0],past(d[0],d[1])));
    }
    require(f.n==p.base,"base feature mismatch");
    // Base features are integer samples divided by 32; this division is exact.
    int64_t part[4]={0,0,0,0};int i=0;
    for(;i+4<=f.n;i+=4)for(int t=0;t<4;t++)part[t]+=int64_t(f.x[i+t])*p.guide[i+t];
    for(;i<f.n;i++)part[0]+=int64_t(f.x[i])*p.guide[i];
    int64_t raw=(part[0]+part[1])+(part[2]+part[3]);raw/=Q/32;
    f.raw=raw;
    f.prediction=int(std::clamp(floor_div(raw+f.anchor*GUIDE_Q+GUIDE_Q/2,GUIDE_Q),
                              int64_t(0),int64_t(255)));
}
static void base_features(const Model& m,const Canvas& c,int ch,int x,int y,Features& f){
    if(inside(c,x,y))base_features(m,Taps<true>(c,x,y),ch,f);
    else base_features(m,Taps<false>(c,x,y),ch,f);
}
template<bool INSIDE>
static Features features(const Model& m,const Taps<INSIDE>& t,int ch){
    Features f;base_features(m,t,ch,f);const Plane& p=m.plane[ch];const Canvas& c=t.c;
    if(m.context){
        int green=t.sample(1,0,0);
        for(int i=0;i<6;i++){
            auto d=causal[i];int e=t.error(ch,d[1],d[0]);f.add(e);f.add(std::abs(e));
        }
        if(ch!=1){
            for(auto d:guides)f.add(t.error(1,d[1],d[0],past(d[0],d[1])));
            if(ch==2)for(auto d:guides)f.add(t.error(0,d[1],d[0],past(d[0],d[1])));
            f.add((green%4)*8);f.add_q(green*(Q/256));
            if(ch==2)f.add((t.sample(0,0,0)%4)*8);
        }
        int64_t phase=f.raw%GUIDE_Q;if(phase<0)phase+=GUIDE_Q;
        f.add_q(rounded(phase,GUIDE_Q/Q));
    }
    if(m.far)for(auto d:far_taps){
        if(ch==1)f.add(t.sample(ch,d[1],d[0])-f.anchor);
        else f.add(t.difference(ch,1,d[1],d[0],true));
    }
    if(m.level){
        f.add(ch==1?f.anchor:t.sample(ch,-1,0));
        f.add((t.y%2)*32);f.add((t.x%2)*32);f.add(((t.x+t.y)%2)*32);
    }
    for(int s=0;s<m.fits;s++){
        const size_t i=t.i;const auto& fits=c.fits[s];
        f.add_q(int64_t(fits[ch][i]-8*f.anchor)*(Q/256));
        f.add(spread_feature(c.spreads[s][ch][i]));
        for(auto d:fit_errors){
            int j=t.location(d[1],d[0]);
            f.add_q(j<0?0:int64_t(8*c.pixels[ch][j]-fits[ch][j])*(Q/256));
        }
        for(int prior:{1,0}){
            if(ch==1||(prior==0&&ch!=2))break;
            f.add_q(int64_t(8*c.pixels[prior][i]-fits[prior][i])*(Q/256));
        }
    }
    require(f.n==p.d,"full feature mismatch");return f;
}
static Features features(const Model& m,const Canvas& c,int ch,int x,int y){
    if(inside(c,x,y))return features(m,Taps<true>(c,x,y),ch);
    return features(m,Taps<false>(c,x,y),ch);
}
static void dense(const int32_t* in,int ni,const std::vector<int32_t>& w,
                  const std::vector<int32_t>& b,int32_t* out,int no,bool relu){
    for(int i=0;i<no;i++){
        int64_t s=int64_t(b[i])*Q;
        for(int j=0;j<ni;j++)s+=int64_t(in[j])*w[i*ni+j];
        int64_t value=std::clamp(rounded(s,Q),-LIMIT,LIMIT);
        out[i]=int32_t(relu?std::max(int64_t(0),value):value);
    }
}
// The layers of dense(), bit for bit: integer sums are exact in any order and
// zero inputs add nothing, so these skip zeros and take first-layer inputs
// u * 256 with |u| <= limit through int16 products (Plane::prepare): value[j]
// is u for those inputs and zero for the others, which large lists. value is
// zero from n to the next multiple of 8; x must be readable that far.
struct Split{alignas(16) int16_t value[128];int large[128];int nl=0;};
static bool narrow_input(int32_t v,int limit){return (v&255)==0&&v/256>=-limit&&v/256<=limit;}
static void split_inputs(const int32_t* x,int n,int limit,Split& s){
    int nl=0;
#if defined(__ARM_NEON)
    const int32x4_t lim=vdupq_n_s32(limit),low=vdupq_n_s32(255),count=vdupq_n_s32(n),step=vdupq_n_s32(4);
    const int32_t first[4]={0,1,2,3};int32x4_t index=vld1q_s32(first);
    for(int j=0;j<n;j+=8){
        int16x4_t half[2];
        UNROLLED for(int h=0;h<2;h++){
            const int32x4_t v=vld1q_s32(x+j+4*h),u=vshrq_n_s32(v,8);
            const uint32x4_t inside=vcltq_s32(index,count);
            const uint32x4_t ok=vandq_u32(vceqzq_s32(vandq_s32(v,low)),vcleq_s32(vabsq_s32(u),lim));
            half[h]=vmovn_s32(vandq_s32(u,vreinterpretq_s32_u32(vandq_u32(ok,inside))));
            if(vmaxvq_u32(vbicq_u32(inside,ok)))
                for(int l=j+4*h;l<std::min(n,j+4*h+4);l++)if(!narrow_input(x[l],limit))s.large[nl++]=l;
            index=vaddq_s32(index,step);
        }
        vst1q_s16(s.value+j,vcombine_s16(half[0],half[1]));
    }
#else
    for(int j=0;j<n;j++){
        const bool fits=narrow_input(x[j],limit);
        s.value[j]=int16_t(fits?x[j]/256:0);
        if(!fits)s.large[nl++]=j;
    }
    for(int j=n;j%8;j++)s.value[j]=0;
#endif
    s.nl=nl;
}
// Each layer adds bias[i] = b[i] * Q + Q / 2 (Plane::prepare), so its output
// clamp(rounded(sum, Q), -LIMIT, LIMIT) is clamp(floor(sum' / Q), ...).
#if defined(__ARM_NEON)
// acc holds outputs o..o+H-1 in pairs. Adds the listed inputs to all of them at
// once, so the sums' dependency chains overlap, then stores them clamped.
template<int P,int H>
static inline void wide_block(int64x2_t* acc,const int32_t* wt,int o,const int* idx,int n,
                              const int32_t* x,int32_t* out,bool relu){
    for(int l=0;l<n;l++){
        const int32_t* row=wt+size_t(idx[l])*P+o;const int32_t v=x[idx[l]];
        UNROLLED for(int c=0;c<H/4;c++){
            const int32x4_t w=vld1q_s32(row+4*c);
            acc[2*c]=vmlal_n_s32(acc[2*c],vget_low_s32(w),v);
            acc[2*c+1]=vmlal_high_n_s32(acc[2*c+1],w,v);
        }
    }
    const int32x4_t low=vdupq_n_s32(relu?0:int32_t(-LIMIT)),high=vdupq_n_s32(int32_t(LIMIT));
    UNROLLED for(int c=0;c<H/4;c++){
        const int32x4_t v=vcombine_s32(vqmovn_s64(vshrq_n_s64(acc[2*c],16)),
                                       vqmovn_s64(vshrq_n_s64(acc[2*c+1],16)));
        vst1q_s32(out+o+4*c,vminq_s32(vmaxq_s32(v,low),high));
    }
}
// Output blocks whose accumulators fit in registers.
template<int P> constexpr int BLOCK=P<=48?P:P/2;
#else
static void finish_scalar(int64_t* sum,const int32_t* wt,int P,const int* idx,int n,const int32_t* x,
                          int32_t* out,bool relu){
    for(int l=0;l<n;l++){
        const int32_t* row=wt+size_t(idx[l])*P;const int64_t v=x[idx[l]];
        for(int i=0;i<P;i++)sum[i]+=v*row[i];
    }
    for(int i=0;i<P;i++){
        const int64_t value=std::clamp(floor_div(sum[i],Q),-LIMIT,LIMIT);
        out[i]=int32_t(relu?std::max(int64_t(0),value):value);
    }
}
#endif
static_assert(Q==65536,"layer shifts");
template<int P>
static void first_layer(const Plane& p,const int32_t* x,const Split& s,int32_t* out){
#if defined(__ARM_NEON)
    // Two accumulator sets halve the dependency chains while registers allow.
    constexpr int S=P<=48?2:1;
    int32x4_t acc[S][P/4];
    UNROLLED for(int k=0;k<P/4;k++)UNROLLED for(int t=0;t<S;t++)acc[t][k]=vdupq_n_s32(0);
    const int16_t* row=p.narrow.data();
    for(int j=0;j<p.d;j+=8,row+=8*P){
        const int16x8_t v=vld1q_s16(s.value+j);
#define NARROW_STEP(L) {const int16x8_t w=vld1q_s16(row+(L)*P+8*k);constexpr int t=(L)*S/8; \
            acc[t][2*k]=vmlal_laneq_s16(acc[t][2*k],vget_low_s16(w),v,L); \
            acc[t][2*k+1]=vmlal_high_laneq_s16(acc[t][2*k+1],w,v,L);}
        UNROLLED for(int k=0;k<P/8;k++){
            NARROW_STEP(0)NARROW_STEP(1)NARROW_STEP(2)NARROW_STEP(3)
            NARROW_STEP(4)NARROW_STEP(5)NARROW_STEP(6)NARROW_STEP(7)
        }
#undef NARROW_STEP
    }
    if constexpr(S==2){UNROLLED for(int k=0;k<P/4;k++)acc[0][k]=vaddq_s32(acc[0][k],acc[1][k]);}
    const int64x2_t shift=vdupq_n_s64(8+p.shift);
    constexpr int H=BLOCK<P>;
    UNROLLED for(int o=0;o<P;o+=H){
        int64x2_t a[H/2];
        UNROLLED for(int c=0;c<H/4;c++){
            const int32x4_t v=acc[0][o/4+c];
            a[2*c]=vaddq_s64(vshlq_s64(vmovl_s32(vget_low_s32(v)),shift),vld1q_s64(&p.bias0[o+4*c]));
            a[2*c+1]=vaddq_s64(vshlq_s64(vmovl_high_s32(v),shift),vld1q_s64(&p.bias0[o+4*c+2]));
        }
        wide_block<P,H>(a,p.wide.data(),o,s.large,s.nl,x,out,true);
    }
#else
    int32_t narrow[P]{};int64_t sum[P];
    for(int j=0;j<p.d;j++){
        const int16_t* row=&p.narrow[size_t(j)*P];const int32_t v=s.value[j];
        for(int i=0;i<P;i++)narrow[i]+=v*row[i];
    }
    for(int i=0;i<P;i++)sum[i]=int64_t(narrow[i])*(int64_t(256)<<p.shift)+p.bias0[i];
    finish_scalar(sum,p.wide.data(),P,s.large,s.nl,x,out,true);
#endif
}
template<int P>
static void sparse_layer(const int32_t* in,int ni,const std::vector<int32_t>& wt,
                         const std::vector<int64_t>& bias,int32_t* out,bool relu){
    int idx[64];int n=0;
    for(int j=0;j<ni;j++){idx[n]=j;n+=in[j]!=0;}
#if defined(__ARM_NEON)
    constexpr int H=BLOCK<P>;
    UNROLLED for(int o=0;o<P;o+=H){
        int64x2_t a[H/2];
        UNROLLED for(int c=0;c<H/2;c++)a[c]=vld1q_s64(&bias[o+2*c]);
        wide_block<P,H>(a,wt.data(),o,idx,n,in,out,relu);
    }
#else
    int64_t sum[P];for(int i=0;i<P;i++)sum[i]=bias[i];
    finish_scalar(sum,wt.data(),P,idx,n,in,out,relu);
#endif
}
// Writes padded outputs; padding reads zero weights and yields zero.
static void sparse_layer(const int32_t* in,int ni,const std::vector<int32_t>& wt,int padded,
                         const std::vector<int64_t>& bias,int32_t* out,bool relu){
    switch(padded){
        case 8:sparse_layer<8>(in,ni,wt,bias,out,relu);break;
        case 16:sparse_layer<16>(in,ni,wt,bias,out,relu);break;
        case 24:sparse_layer<24>(in,ni,wt,bias,out,relu);break;
        case 32:sparse_layer<32>(in,ni,wt,bias,out,relu);break;
        case 40:sparse_layer<40>(in,ni,wt,bias,out,relu);break;
        case 48:sparse_layer<48>(in,ni,wt,bias,out,relu);break;
        case 56:sparse_layer<56>(in,ni,wt,bias,out,relu);break;
        case 64:sparse_layer<64>(in,ni,wt,bias,out,relu);break;
        default:throw std::logic_error("layer width");
    }
}
static void first_layer(const Plane& p,const int32_t* x,const Split& s,int32_t* out,int hidden){
    switch(p.narrow_ok?p.padded:0){
        case 8:first_layer<8>(p,x,s,out);break;
        case 16:first_layer<16>(p,x,s,out);break;
        case 24:first_layer<24>(p,x,s,out);break;
        case 32:first_layer<32>(p,x,s,out);break;
        case 40:first_layer<40>(p,x,s,out);break;
        case 48:first_layer<48>(p,x,s,out);break;
        case 56:first_layer<56>(p,x,s,out);break;
        case 64:first_layer<64>(p,x,s,out);break;
        default:dense(x,p.d,p.w0,p.b0,out,hidden,true);
    }
}
struct Tables{
    std::vector<uint32_t> negative,scales,positive;
    Tables():negative(16*Q+1),scales(11*Q+1),positive(16*Q+1){
        // Normative integer recurrence, not a platform exponential function.
        // Constants: rounded exp(+/-1/65536)*2^62 and exp(-4)*2^48.
        uint64_t n=uint64_t(1)<<48,p=5155394029647ULL;
        for(auto& value:negative){
            value=uint32_t((n+(uint64_t(1)<<17))>>18);
            n=uint64_t((__uint128_t(n)*4611615650220078421ULL+
                        (__uint128_t(1)<<61))>>62);
        }
        for(auto& value:scales){
            value=uint32_t((p+(uint64_t(1)<<31))>>32);
            p=uint64_t((__uint128_t(p)*4611756387708439211ULL+
                        (__uint128_t(1)<<61))>>62);
        }
        // Rounded 2^60 / (2^30 + negative), the logistic of z >= 0 (Distribution::sigmoid).
        for(size_t i=0;i<positive.size();i++){
            const uint64_t den=(uint64_t(1)<<30)+negative[i];
            positive[i]=uint32_t(((uint64_t(1)<<60)+den/2)/den);
        }
    }
};
static const Tables& tables(){static const Tables t;return t;}
constexpr uint64_t PROB=uint64_t(1)<<30;
struct Distribution{
    int k,anchor;uint32_t low,mass;uint64_t normal=0;
    std::array<uint32_t,4> weights{},scales{};std::array<int64_t,4> means{};
    static uint32_t sigmoid(int64_t z){
        if(z>=16*Q)return uint32_t(PROB);if(z<=-16*Q)return 0;
        const uint32_t positive=tables().positive[size_t(z<0?-z:z)];
        return z<0?uint32_t(PROB)-positive:positive;
    }
    uint32_t raw(int v)const{
        uint64_t sum=0;
        for(int i=0;i<k;i++){
            int64_t diff=int64_t(v)*Q-Q/2-int64_t(anchor)*Q-means[i];
            sum+=uint64_t(weights[i])*sigmoid(rounded(diff*Q,scales[i]));
        }
        return uint32_t((sum+normal/2)/normal);
    }
    uint32_t cdf(int v)const{
        if(v<=0)return 0;if(v>=256)return TOTAL;
        if(!mass)return uint32_t(v)*(TOTAL/256);
        return uint32_t(v)+uint32_t((uint64_t(TOTAL-256)*(raw(v)-low)+mass/2)/mass);
    }
    // The cdf is strictly increasing, so exactly one v has cdf(v)<=target<cdf(v+1);
    // searching outward from the heaviest component only changes the cost.
    int locate(uint32_t target,uint32_t& lo,uint32_t& hi)const{
        int best=0;for(int i=1;i<k;i++)if(weights[i]>weights[best])best=i;
        int left=int(std::clamp(int64_t(anchor)+rounded(means[best],Q),int64_t(0),int64_t(255)));
        int right=left;uint32_t start=cdf(left);
        if(start<=target){
            lo=start;
            for(int step=1;;step*=2){
                right=std::min(256,left+step);hi=cdf(right);if(hi>target)break;
                left=right;lo=hi;
            }
        }else{
            hi=start;
            for(int step=1;;step*=2){
                left=std::max(0,right-step);lo=cdf(left);if(lo<=target)break;
                right=left;hi=lo;
            }
        }
        while(left+1<right){
            int mid=(left+right)/2;uint32_t value=cdf(mid);
            if(value<=target){left=mid;lo=value;}else{right=mid;hi=value;}
        }
        return left;
    }
};
static Distribution infer(const Model& m,int ch,const Features& f){
    const Plane& p=m.plane[ch];std::array<int32_t,64> h0{},h1{};
    std::array<int32_t,16> raw{};
    Split s;split_inputs(f.x.data(),p.d,p.unit_limit,s);
    first_layer(p,f.x.data(),s,h0.data(),m.hidden);
    sparse_layer(h0.data(),m.hidden,p.w1t,p.padded,p.bias1,h1.data(),true);
    sparse_layer(h1.data(),m.hidden,p.wht,p.head_padded,p.biash,raw.data(),false);
    __int128 affine_raw=0;
    if(p.affine_ok){
        // |u| <= 32767, at most 128 terms, |affine| bounded in Plane::prepare.
        // Four partial sums shorten the dependency chain; integer sums are exact.
        int64_t small[4]={0,0,0,0};const int64_t* a=p.affine_padded.data();
        for(int j=0;j<p.d;j+=4)for(int t=0;t<4;t++)small[t]+=int64_t(s.value[j+t])*a[j+t];
        affine_raw=__int128((small[0]+small[1])+(small[2]+small[3]))*256;
    }else{
        for(int j=0;j<p.d;j++)affine_raw+=__int128(s.value[j])*256*p.affine[size_t(j)];
    }
    for(int l=0;l<s.nl;l++)affine_raw+=__int128(f.x[s.large[l]])*p.affine[size_t(s.large[l])];
    affine_raw+=GUIDE_Q/2;
    __int128 affine=affine_raw/GUIDE_Q-(affine_raw%GUIDE_Q<0);
    affine=std::max(-(__int128(1)<<30),std::min(__int128(1)<<30,affine));
    int32_t maximum=*std::max_element(raw.begin(),raw.begin()+m.k);
    Distribution d;d.k=m.k;d.anchor=f.anchor;
    for(int i=0;i<m.k;i++){
        int64_t distance=int64_t(maximum)-raw[i];
        d.weights[i]=distance>16*Q?0:tables().negative[size_t(distance)];
        d.normal+=d.weights[i];d.means[i]=raw[m.k+i]+int64_t(affine);
        int64_t logarithm=std::clamp(int64_t(raw[2*m.k+i]),-4*Q,7*Q);
        d.scales[i]=tables().scales[size_t(logarithm+4*Q)];
    }
    d.low=d.raw(0);d.mass=d.raw(256)-d.low;return d;
}

// Runs work(0..count-1) on up to `threads` threads, claiming indices in
// increasing order. The first exception wins and sets *abort for spinning peers.
template<class Work>
static void parallel(int count,int threads,Work&& work,std::atomic<bool>* abort=nullptr){
    threads=std::max(1,std::min(threads,count));
    std::atomic<int> next{0};std::exception_ptr error;std::mutex lock;
    auto run=[&]{
        try{for(int i;(i=next.fetch_add(1))<count;)work(i);}
        catch(...){
            {std::lock_guard<std::mutex> hold(lock);if(!error)error=std::current_exception();}
            next.store(count);if(abort)abort->store(true);
        }
    };
    std::vector<std::thread> pool;
    for(int t=1;t<threads;t++){
        try{pool.emplace_back(run);}catch(const std::system_error&){break;}
    }
    run();for(auto& t:pool)t.join();
    if(error)std::rethrow_exception(error);
}
static int automatic_threads(bool performance_only){
    int count=int(std::thread::hardware_concurrency());
#if defined(__APPLE__)
    // Wavefront rows wait on the row above, so one efficiency core slows every
    // row below it; decoding uses performance cores only.
    int cores=0;size_t size=sizeof(cores);
    if(performance_only&&sysctlbyname("hw.perflevel0.physicalcpu",&cores,&size,nullptr,0)==0&&
       cores>0)count=cores;
#else
    (void)performance_only;
#endif
    return std::clamp(count,1,64);
}

constexpr int DEFAULT_STREAMS=16,MAX_STREAMS=64;
static void put_varint(Bytes& out,uint64_t value){
    do{uint8_t byte=value&0x7f;value>>=7;out.push_back(byte|(value?0x80:0));}while(value);
}
static uint64_t read_varint(const Bytes& in,size_t& pos){
    uint64_t value=0;
    for(int shift=0;;shift+=7){
        require(pos<in.size()&&shift<=28,"invalid stream lengths");
        uint8_t byte=in[pos++];value|=uint64_t(byte&0x7f)<<shift;
        if(!(byte&0x80))return value;
    }
}

// Every fit of a known image. Each block of rows starts its column sums from
// scratch; exact integer sums make that identical to the decoder's updates.
constexpr int FIT_BLOCK=16;
static void fit_image(Canvas& c,const FitWindows& windows,int threads){
    if(paired(windows)){
        for(int ch:{1,0,2}){
            const size_t size=size_t(c.w)*fit_size(*windows[0],ch);
            parallel((c.h+FIT_BLOCK-1)/FIT_BLOCK,threads,[&](int block){
                std::vector<int32_t> a0(size),b0(size),a1(size),b1(size);
                for(int y=block*FIT_BLOCK;y<std::min(c.h,(block+1)*FIT_BLOCK);y++){
                    const bool first=y==block*FIT_BLOCK;
                    LocalFit fit0(c,*windows[0],ch,y,b0.data(),first?nullptr:a0.data());
                    LocalFit fit1(c,*windows[1],ch,y,b1.data(),first?nullptr:a1.data());
                    for(int x=0;x<c.w;x++){
                        size_t i=size_t(y)*c.w+x;
                        predict_pair(fit0,fit1,x,c.fits[0][ch][i],c.spreads[0][ch][i],
                                     c.fits[1][ch][i],c.spreads[1][ch][i]);
                    }
                    std::swap(a0,b0);std::swap(a1,b1);
                }
            });
        }
        return;
    }
    for(int s=0;s<window_count(windows);s++)for(int ch:{1,0,2}){
        const FitWindow& win=*windows[s];const size_t size=size_t(c.w)*fit_size(win,ch);
        parallel((c.h+FIT_BLOCK-1)/FIT_BLOCK,threads,[&](int block){
            std::vector<int32_t> a(size),b(size);
            for(int y=block*FIT_BLOCK;y<std::min(c.h,(block+1)*FIT_BLOCK);y++){
                LocalFit fit(c,win,ch,y,b.data(),y==block*FIT_BLOCK?nullptr:a.data());
                for(int x=0;x<c.w;x++){
                    size_t i=size_t(y)*c.w+x;fit.predict(x,c.fits[s][ch][i],c.spreads[s][ch][i]);
                }
                std::swap(a,b);
            }
        });
    }
}
// Every sample of a known image has a computable distribution, so the encoder
// evaluates them in parallel and only arithmetic coding stays in sample order.
// Row y belongs to stream y % streams; one stream reproduces the v3 bitstream.
// fitted says the canvas already holds this image's fits.
static Bytes encode_image(const Model& m,Canvas& c,int streams,int threads,bool split,
                          bool fitted){
    for(int ch:{1,0,2})parallel(m.h,threads,[&](int y){
        for(int x=0;x<m.w;x++){
            Features f;base_features(m,c,ch,x,y,f);size_t i=size_t(y)*m.w+x;
            c.errors[ch][i]=int16_t(c.pixels[ch][i]-f.prediction);
        }
    });
    if(m.fits&&!fitted)fit_image(c,m.windows,threads);
    std::vector<ArithmeticEncoder> coders(static_cast<size_t>(streams));
    const int chunk=std::max(1,(1<<19)/m.w);
    std::vector<std::array<uint32_t,2>> pairs(size_t(std::min(chunk,m.h))*m.w);
    for(int ch:{1,0,2})for(int y0=0;y0<m.h;y0+=chunk){
        const int rows=std::min(chunk,m.h-y0);
        parallel(rows,threads,[&](int r){
            const int y=y0+r;
            for(int x=0;x<m.w;x++){
                Distribution p=infer(m,ch,features(m,c,ch,x,y));
                int value=c.pixels[ch][size_t(y)*m.w+x];
                pairs[size_t(r)*m.w+x]={p.cdf(value),p.cdf(value+1)};
            }
        });
        parallel(streams,threads,[&](int s){
            for(int y=y0+(s-y0%streams+streams)%streams;y<y0+rows;y+=streams)
                for(int x=0;x<m.w;x++){
                    auto pair=pairs[size_t(y-y0)*m.w+x];coders[s].put(pair[0],pair[1]);
                }
        });
    }
    if(!split)return coders[0].finish();
    // Substreams omit the four zero padding bytes; the decoder restores them.
    Bytes out{uint8_t(streams)};std::vector<Bytes> parts;
    for(auto& coder:coders){
        parts.push_back(coder.finish());parts.back().resize(parts.back().size()-4);
        put_varint(out,parts.back().size());
    }
    for(auto& part:parts)out.insert(out.end(),part.begin(),part.end());
    return out;
}
// For each least-squares window s, sums[s] receives this row's column sums and
// above[s] holds the previous row's (see LocalFit).
using FitSums=std::array<int32_t*,MAX_FITS>;
template<class Wait,class Done>
static void decode_row(const Model& m,Canvas& c,int ch,int y,ArithmeticDecoder& decoder,
                       Wait&& wait,Done&& done,const FitSums& sums,const FitSums& above){
    std::array<std::optional<LocalFit>,MAX_FITS> fit;
    const bool pair=paired(m.windows);
    for(int x=0;x<m.w;x++){
        wait(x);
        size_t i=size_t(y)*m.w+x;
        for(int s=0;s<m.fits;s++)if(!fit[s])fit[s].emplace(c,*m.windows[s],ch,y,sums[s],above[s]);
        if(pair)predict_pair(*fit[0],*fit[1],x,c.fits[0][ch][i],c.spreads[0][ch][i],
                             c.fits[1][ch][i],c.spreads[1][ch][i]);
        else for(int s=0;s<m.fits;s++)fit[s]->predict(x,c.fits[s][ch][i],c.spreads[s][ch][i]);
        Features f=features(m,c,ch,x,y);
        Distribution p=infer(m,ch,f);
        uint32_t low,high;int value=p.locate(decoder.target(),low,high);
        decoder.take(low,high);
        c.pixels[ch][i]=uint8_t(value);c.errors[ch][i]=int16_t(value-f.prediction);
        done(x);
    }
}
// Column-sum buffers for every window, slots per window, each sized for plane B.
struct FitRing{
    std::array<std::vector<int32_t>,MAX_FITS> buffers;std::array<size_t,MAX_FITS> size{};
    FitRing(const Model& m,int slots){
        for(int s=0;s<m.fits;s++){
            size[s]=size_t(m.w)*fit_size(*m.windows[s],2);buffers[s].resize(size[s]*slots);
        }
    }
    FitSums slot(int k){
        FitSums out{};for(int s=0;s<MAX_FITS;s++)out[s]=buffers[s].data()+size[s]*k;return out;
    }
};
static void decode_single(const Model& m,Canvas& c,const Bytes& source){
    ArithmeticDecoder decoder(source);FitRing ring(m,2);
    for(int ch:{1,0,2})for(int y=0;y<m.h;y++)
        decode_row(m,c,ch,y,decoder,[](int){},[](int){},ring.slot(y%2),ring.slot((y+1)%2));
}
// Wavefront: row y needs rows above it decoded through column x+3 (the widest
// causal tap), or x+FIT_REACH+1 with least-squares fits, and its stream's previous row
// finished. Rows are claimed in
// order, so the lowest unfinished row can always proceed.
static void decode_split(const Model& m,Canvas& c,const Bytes& source,int threads){
    require(!source.empty(),"truncated arithmetic stream");
    size_t pos=1;const int streams=source[0];
    require(streams>=1&&streams<=MAX_STREAMS&&streams<=m.h,"invalid stream count");
    std::vector<uint64_t> sizes(static_cast<size_t>(streams));uint64_t total=0;
    for(auto& size:sizes){size=read_varint(source,pos);total+=size;}
    require(total==source.size()-pos,"invalid stream lengths");
    std::vector<Bytes> parts(static_cast<size_t>(streams));
    for(int s=0;s<streams;s++){
        parts[s].assign(source.begin()+long(pos),source.begin()+long(pos+sizes[s]));
        parts[s].insert(parts[s].end(),4,0);pos+=sizes[s];
    }
    std::vector<std::unique_ptr<ArithmeticDecoder>> decoders;
    for(auto& part:parts)decoders.push_back(std::make_unique<ArithmeticDecoder>(part));
    std::unique_ptr<std::atomic<int>[]> progress(new std::atomic<int>[size_t(m.h)]);
    std::atomic<bool> failed{false};
    auto await=[&](int row,int need){
        int seen;
        while((seen=progress[row].load(std::memory_order_acquire))<need){
            if(failed.load(std::memory_order_relaxed))throw std::runtime_error("decode aborted");
            std::this_thread::yield();
        }
        return seen;
    };
    const int reach=m.fits?FIT_REACH+1:4;
    // Rows finish in order, so at most `workers` rows are unfinished and a ring of
    // workers+1 column-sum buffers never overwrites one that is still read.
    const int workers=std::min(threads,streams),slots=workers+1;
    FitRing ring(m,slots);
    for(int ch:{1,0,2}){
        for(int y=0;y<m.h;y++)progress[y].store(0,std::memory_order_relaxed);
        parallel(m.h,workers,[&](int y){
            if(y>=streams)await(y-streams,m.w);
            int above=y>0?0:m.w;
            decode_row(m,c,ch,y,*decoders[y%streams],
                [&](int x){int need=std::min(m.w,x+reach);if(above<need)above=await(y-1,need);},
                [&](int x){progress[y].store(x+1,std::memory_order_release);},
                ring.slot(y%slots),ring.slot((y+slots-1)%slots));
        },&failed);
    }
}
static int count_argument(const char* text,int low,int high){
    char* end=nullptr;long value=std::strtol(text,&end,10);
    require(end!=text&&*end==0&&value>=low&&value<=high,"invalid thread or stream count");
    return int(value);
}
// Exact least-squares fits for training. For each window of the feature set (3
// by default): int16 fits, then spreads, planes G, R, B.
static int local_fit_main(int argc,char** argv){
    require(argc>=6&&argc<=8,
            "usage: nnproto-integer local-fit width height input output [threads [set]]");
    const int w=count_argument(argv[2],1,16777216),h=count_argument(argv[3],1,16777216);
    require(uint64_t(w)*h<=16777216,"invalid canvas");
    int threads=argc>6?count_argument(argv[6],0,MAX_STREAMS):0;
    if(!threads)threads=automatic_threads(false);
    const FitWindows windows=set_windows(argc>7?uint32_t(count_argument(argv[7],3,4)):3);
    const int count=window_count(windows);
    Bytes source=load(argv[4],size_t(3)*w*h);
    require(source.size()==size_t(w)*h*3,"input pixel size mismatch");
    Canvas c(w,h,count);
    for(size_t i=0;i<size_t(w)*h;i++)for(int ch=0;ch<3;ch++)c.pixels[ch][i]=source[3*i+ch];
    fit_image(c,windows,threads);
    Bytes output;output.reserve(size_t(w)*h*12*count);
    for(int s=0;s<count;s++)for(auto* planes:{&c.fits[s],&c.spreads[s]})
        for(int ch:{1,0,2})for(int16_t value:(*planes)[ch]){
            output.push_back(uint8_t(value&0xff));output.push_back(uint8_t((value>>8)&0xff));
        }
    save(argv[5],output);return 0;
}
// Reads local-fit output for the canvas's pixels. Wrong fits code a file that
// does not decode, so callers must decode what they encode.
static void load_fits(const Model& m,Canvas& c,const char* path){
    require(m.fits>0,"the feature set has no least-squares fits");
    const size_t n=size_t(m.w)*m.h;Bytes source=load(path,n*12*m.fits);
    require(source.size()==n*12*m.fits,"local fit size mismatch");
    size_t pos=0;
    for(int s=0;s<m.fits;s++)for(auto* planes:{&c.fits[s],&c.spreads[s]})
        for(int ch:{1,0,2})for(int16_t& value:(*planes)[ch]){
            value=int16_t(uint16_t(source[pos])|uint16_t(source[pos+1]<<8));pos+=2;
        }
}
int main(int argc,char** argv){
    try{
        if(argc>1&&std::string(argv[1])=="local-fit")return local_fit_main(argc,argv);
        require(argc>=5&&argc<=8,
                "usage: nnproto-integer encode|decode|encode-v4|decode-v4 "
                "config input output [threads [streams [fits]]]");
        const std::string mode=argv[1];
        const bool split=mode=="encode-v4"||mode=="decode-v4";
        const bool encoding=mode=="encode"||mode=="encode-v4";
        require(encoding||mode=="decode"||split,"invalid mode");
        int threads=argc>5?count_argument(argv[5],0,MAX_STREAMS):0;
        if(!threads)threads=automatic_threads(!encoding);
        require(argc<7||mode=="encode-v4","only encode-v4 takes a stream count");
        int streams=argc>6?count_argument(argv[6],1,MAX_STREAMS):DEFAULT_STREAMS;
        Model m(load(argv[2],1024*1024));Canvas c(m.w,m.h,m.fits);
        Bytes source=load(argv[3],128*1024*1024);
        Bytes output;
        if(encoding){
            require(source.size()==size_t(m.w)*m.h*3,"input pixel size mismatch");
            for(size_t i=0;i<size_t(m.w)*m.h;i++)
                for(int ch=0;ch<3;ch++)c.pixels[ch][i]=source[3*i+ch];
            if(argc>7)load_fits(m,c,argv[7]);
            output=encode_image(m,c,split?std::min(streams,m.h):1,threads,split,argc>7);
        }else{
            if(split)decode_split(m,c,source,threads);else decode_single(m,c,source);
            output.resize(size_t(m.w)*m.h*3);
            for(size_t i=0;i<size_t(m.w)*m.h;i++)
                for(int ch=0;ch<3;ch++)output[3*i+ch]=c.pixels[ch][i];
        }
        save(argv[4],output);std::cout<<output.size()<<" bytes\n";
    }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
}
