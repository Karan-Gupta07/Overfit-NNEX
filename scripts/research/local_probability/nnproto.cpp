#include "arithmetic.hpp"
struct Reader{
    const Bytes& b;size_t pos=0;
    uint32_t u32(){
        require(pos+4<=b.size(),"truncated configuration");
        uint32_t v=uint32_t(b[pos])|(uint32_t(b[pos+1])<<8)|
                   (uint32_t(b[pos+2])<<16)|(uint32_t(b[pos+3])<<24);
        pos+=4;return v;
    }
    std::vector<double> values(size_t n){
        require(n<=16384&&pos+8*n<=b.size(),"invalid parameter count");
        std::vector<double> result(n);
        for(size_t i=0;i<n;i++){
            uint64_t raw=0;for(int k=0;k<8;k++)raw|=uint64_t(b[pos++])<<(8*k);
            std::memcpy(&result[i],&raw,8);
            require(std::isfinite(result[i])&&std::abs(result[i])<10000,"invalid parameter");
        }
        return result;
    }
};
struct Plane{
    int d,base;
    std::vector<double> guide,affine,w0,b0,w1,b1,wh,bh;
};
struct Model{
    int w,h,hidden,k;bool context;
    std::array<Plane,3> plane;
    explicit Model(const Bytes& bytes){
        Reader r{bytes};
        require(r.u32()==0x31504e4e,"invalid reference configuration");
        w=r.u32();h=r.u32();hidden=r.u32();k=r.u32();auto c=r.u32();
        require(w>0&&h>0&&uint64_t(w)*h<=16777216,"invalid canvas");
        require(hidden>0&&hidden<=64&&k>0&&k<=4&&c<=1,"invalid architecture");
        context=c;
        const int base_dims[3]={21,14,30},extra[3]={24,13,34};
        for(int channel: {1,0,2}){
            Plane& p=plane[channel];p.d=r.u32();p.base=r.u32();
            require(p.base==base_dims[channel]&&p.d==p.base+(context?extra[channel]:0),
                    "invalid feature dimensions");
            p.guide=r.values(p.base);p.affine=r.values(p.d);
            p.w0=r.values(hidden*p.d);p.b0=r.values(hidden);
            p.w1=r.values(hidden*hidden);p.b1=r.values(hidden);
            p.wh=r.values(3*k*hidden);p.bh=r.values(3*k);
        }
        require(r.pos==bytes.size(),"trailing configuration bytes");
    }
};
const std::array<std::array<int,2>,12> causal={{{0,-1},{-1,0},{-1,-1},{-1,1},
    {0,-2},{-2,0},{-2,-1},{-2,1},{-1,-2},{-1,2},{-2,-2},{-2,2}}};
const std::array<std::array<int,2>,9> guides={{{0,0},{0,-1},{-1,0},{-1,-1},
    {-1,1},{0,1},{1,0},{1,-1},{1,1}}};
struct Canvas{
    int w,h;
    std::array<Bytes,3> pixels;
    std::array<std::vector<int16_t>,3> errors;
    Canvas(int width,int height):w(width),h(height){
        for(auto& p:pixels)p.resize(size_t(w)*h,0);
        for(auto& e:errors)e.resize(size_t(w)*h,0);
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
struct Features{
    std::array<double,128> x{};int n=0,anchor=0,prediction=0;
    void add(double v){require(n<128,"feature budget");x[n++]=v;}
};
static bool past(int dy,int dx){return dy<0||(dy==0&&dx<0);}
static Features features(const Model& model,const Canvas& c,int ch,int x,int y){
    Features f;const Plane& p=model.plane[ch];
    int west=c.sample(ch,x,y,-1,0),north=c.sample(ch,x,y,0,-1),nw=c.sample(ch,x,y,-1,-1);
    int green=c.sample(1,x,y,0,0);
    f.anchor=ch==1?std::clamp(west+north-nw,std::min(west,north),std::max(west,north)):green;
    f.add(1);
    if(ch==1){
        for(auto d:causal)f.add(c.sample(ch,x,y,d[1],d[0])-f.anchor);
        f.add(std::clamp(west+north-nw,0,255)-f.anchor);
    }else{
        for(auto d:causal)f.add(c.difference(ch,1,x,y,d[1],d[0],true));
        for(size_t i=1;i<guides.size();i++){
            auto d=guides[i];f.add(c.sample(1,x,y,d[1],d[0],past(d[0],d[1]))-green);
        }
        if(ch==2)for(auto d:guides)
            f.add(c.difference(0,1,x,y,d[1],d[0],past(d[0],d[1])));
    }
    require(f.n==p.base,"base feature mismatch");
    double raw=0;for(int i=0;i<f.n;i++)raw+=f.x[i]*p.guide[i];
    f.prediction=int(std::clamp(std::floor(raw+f.anchor+0.5),0.,255.));
    if(model.context){
        for(int i=0;i<6;i++){
            auto d=causal[i];int e=c.error(ch,x,y,d[1],d[0]);f.add(e);f.add(std::abs(e));
        }
        if(ch!=1){
            for(auto d:guides)f.add(c.error(1,x,y,d[1],d[0],past(d[0],d[1])));
            if(ch==2)for(auto d:guides)f.add(c.error(0,x,y,d[1],d[0],past(d[0],d[1])));
            f.add((green%4)*8);f.add(green/8.);
            if(ch==2)f.add((c.sample(0,x,y,0,0)%4)*8);
        }
        f.add((raw-std::floor(raw))*32);
    }
    require(f.n==p.d,"full feature mismatch");
    for(int i=0;i<f.n;i++)f.x[i]/=32.;
    return f;
}
static void dense(const double* in,int ni,const std::vector<double>& weights,
                  const std::vector<double>& bias,double* out,int no,bool relu){
    for(int i=0;i<no;i++){
        double s=bias[i];
        for(int j=0;j<ni;j++)s+=in[j]*weights[i*ni+j];
        out[i]=relu?std::max(0.,s):s;
    }
}
struct Distribution{
    int k;double anchor,low,mass,normal;
    std::array<double,4> weights{},means{},scales{};
    static double sigmoid(double v){
        if(v>=0)return 1./(1.+std::exp(-v));
        double e=std::exp(v);return e/(1.+e);
    }
    double raw(int v)const{
        double sum=0;
        for(int i=0;i<k;i++)sum+=weights[i]*sigmoid((v-.5-anchor-means[i])/scales[i]);
        return sum/normal;
    }
    uint32_t cdf(int v)const{
        if(v<=0)return 0;if(v>=256)return TOTAL;
        double q=mass>1e-12?(raw(v)-low)/mass:v/256.;
        q=std::clamp(q,0.,1.);
        return uint32_t(v)+uint32_t(std::floor((TOTAL-256)*q+.5));
    }
};
static Distribution infer(const Model& m,int ch,const Features& f){
    const Plane& p=m.plane[ch];
    std::array<double,64> h0{},h1{};std::array<double,12> raw{};
    dense(f.x.data(),p.d,p.w0,p.b0,h0.data(),m.hidden,true);
    dense(h0.data(),m.hidden,p.w1,p.b1,h1.data(),m.hidden,true);
    dense(h1.data(),m.hidden,p.wh,p.bh,raw.data(),3*m.k,false);
    double affine=0;for(int i=0;i<p.d;i++)affine+=f.x[i]*p.affine[i];
    double maximum=*std::max_element(raw.begin(),raw.begin()+m.k);
    Distribution d;d.k=m.k;d.anchor=f.anchor;d.normal=0;
    for(int i=0;i<m.k;i++){
        d.weights[i]=std::exp(raw[i]-maximum);d.normal+=d.weights[i];
        d.means[i]=raw[m.k+i]+affine;
        d.scales[i]=std::exp(std::clamp(raw[2*m.k+i],-4.,7.));
    }
    d.low=d.raw(0);d.mass=d.raw(256)-d.low;
    require(std::isfinite(d.low)&&std::isfinite(d.mass),"invalid probability model");
    return d;
}
int main(int argc,char** argv){
    try{
        require(argc==5,"usage: nnproto encode|decode config input output");
        Model m(load(argv[2],1024*1024));Canvas c(m.w,m.h);
        bool encoding=std::string(argv[1])=="encode";
        require(encoding||std::string(argv[1])=="decode","invalid mode");
        Bytes source=load(argv[3],128*1024*1024);
        if(encoding){
            require(source.size()==size_t(m.w)*m.h*3,"input pixel size mismatch");
            for(size_t i=0;i<size_t(m.w)*m.h;i++)
                for(int ch=0;ch<3;ch++)c.pixels[ch][i]=source[3*i+ch];
        }
        ArithmeticEncoder encoder;
        Bytes dummy(8,0);ArithmeticDecoder decoder(encoding?dummy:source);
        for(int ch:{1,0,2}){
            for(int y=0;y<m.h;y++)for(int x=0;x<m.w;x++){
                size_t i=size_t(y)*m.w+x;
                Features f=features(m,c,ch,x,y);
                Distribution p=infer(m,ch,f);
                int value;
                if(encoding){
                    value=c.pixels[ch][i];encoder.put(p.cdf(value),p.cdf(value+1));
                }else{
                    uint32_t target=decoder.target();
                    int left=0,right=256;
                    while(left+1<right){int mid=(left+right)/2;
                        if(p.cdf(mid)<=target)left=mid;else right=mid;}
                    value=left;decoder.take(p.cdf(value),p.cdf(value+1));
                    c.pixels[ch][i]=uint8_t(value);
                }
                c.errors[ch][i]=int16_t(value-f.prediction);
            }
        }
        Bytes output;
        if(encoding)output=encoder.finish();
        else{
            output.resize(size_t(m.w)*m.h*3);
            for(size_t i=0;i<size_t(m.w)*m.h;i++)
                for(int ch=0;ch<3;ch++)output[3*i+ch]=c.pixels[ch][i];
        }
        save(argv[4],output);std::cout<<output.size()<<" bytes\n";
    }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
}
