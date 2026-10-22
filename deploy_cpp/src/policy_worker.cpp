// Persistent TensorRT inference only. UniLab owns the full physics simulation.
// Binary protocol (little endian): RMA1 + uint32[49,50,12] on startup;
// each request is float32 observation[49], history[50*49]; reply action[12].
#include "TensorRTPolicy.hpp"
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>

int main(int argc,char** argv){
    static_assert(sizeof(float)==4 && std::numeric_limits<float>::is_iec559,
                  "The worker protocol requires IEEE754 float32");
    try {
        if(argc!=3 || std::string(argv[1])!="--engine")
            throw std::runtime_error("Usage: policy_worker --engine file.engine");
        const std::uint32_t endian=1;
        if(*reinterpret_cast<const unsigned char*>(&endian)!=1)
            throw std::runtime_error("The worker protocol requires little endian");
        TensorRTPolicy policy(argv[2]);
        std::array<float,49> obs{};
        std::array<float,2450> history{};
        std::array<float,12> action{};
        policy.infer(obs,history,action); // Finish CUDA initialization before handshake.
        for(float value:action) if(!std::isfinite(value))
            throw std::runtime_error("Non-finite warm-up output");
        const std::array<std::uint32_t,3> dimensions{49,50,12};
        std::cout.write("RMA1",4);
        std::cout.write(reinterpret_cast<const char*>(dimensions.data()),sizeof(dimensions));
        std::cout.flush();
        while(true){
            std::cin.read(reinterpret_cast<char*>(obs.data()),sizeof(obs));
            if(std::cin.gcount()==0 && std::cin.eof()) break;
            if(!std::cin) throw std::runtime_error("Incomplete observation frame");
            std::cin.read(reinterpret_cast<char*>(history.data()),sizeof(history));
            if(!std::cin) throw std::runtime_error("Incomplete history frame");
            for(float value:obs) if(!std::isfinite(value))
                throw std::runtime_error("Non-finite observation input");
            for(float value:history) if(!std::isfinite(value))
                throw std::runtime_error("Non-finite history input");
            policy.infer(obs,history,action);
            for(float value:action) if(!std::isfinite(value))
                throw std::runtime_error("Non-finite action output");
            std::cout.write(reinterpret_cast<const char*>(action.data()),sizeof(action));
            std::cout.flush();
            if(!std::cout) throw std::runtime_error("Cannot write action frame");
        }
        return 0;
    } catch(const std::exception& error){
        std::cerr<<"TensorRT worker: "<<error.what()<<"\n";
        return 1;
    }
}
