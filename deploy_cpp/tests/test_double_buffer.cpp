#include "DoubleBuffer.hpp"
#include <array>
#include <atomic>
#include <cstdint>
#include <iostream>
#include <thread>
struct Payload {std::uint64_t sequence{};std::array<std::uint64_t,64> values{};};
int main(){
    DoubleBuffer<Payload> buffer;
    Payload first;
    if(buffer.tryRead(first)) return 1;
    std::atomic<bool> done{false};
    std::atomic<int> errors{0};
    std::uint64_t count=0;
    std::thread writer([&]{
        for(std::uint64_t i=1;i<=200000;++i){
            Payload p;p.sequence=i;p.values.fill(i);
            while(!buffer.tryWrite(p)) std::this_thread::yield();
        }
        done.store(true,std::memory_order_release);
    });
    std::thread reader([&]{
        std::uint64_t last=0;
        auto verify=[&](const Payload& p){
            if(p.sequence<=last) ++errors;
            for(auto v:p.values) if(v!=p.sequence) ++errors;
            last=p.sequence;++count;
        };
        while(!done.load(std::memory_order_acquire)){
            Payload p;
            if(buffer.tryRead(p)) verify(p);
        }
        Payload p;
        if(buffer.tryRead(p)) verify(p);
    });
    writer.join();reader.join();
    std::cout<<"snapshots="<<count<<" errors="<<errors<<"\n";
    return errors.load()==0 && count>0?0:1;
}
