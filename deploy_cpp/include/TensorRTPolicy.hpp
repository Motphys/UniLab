#pragma once
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <array>
#include <cstdio>
#include <initializer_list>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

class TensorRTPolicy {
    struct Logger : nvinfer1::ILogger {
        void log(Severity severity,const char* msg) noexcept override {
            if(severity<=Severity::kERROR) std::fprintf(stderr,"TensorRT: %s\n",msg);
        }
    } logger_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    cudaStream_t stream_{};
    void *obs_{},*history_{},*action_{};
    static void check(cudaError_t result){
        if(result!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
    }
    void cleanup() noexcept {
        if(stream_) cudaStreamSynchronize(stream_);
        if(obs_) cudaFree(obs_);
        if(history_) cudaFree(history_);
        if(action_) cudaFree(action_);
        if(stream_) cudaStreamDestroy(stream_);
    }
    void require(const char* name,nvinfer1::TensorIOMode mode,std::initializer_list<int> shape){
        if(engine_->getTensorIOMode(name)!=mode || engine_->getTensorDataType(name)!=nvinfer1::DataType::kFLOAT ||
           engine_->getTensorLocation(name)!=nvinfer1::TensorLocation::kDEVICE)
            throw std::runtime_error(std::string("Engine IO type/mode mismatch: ")+name);
        const auto dims=engine_->getTensorShape(name);
        if(dims.nbDims!=static_cast<int>(shape.size())) throw std::runtime_error("Engine rank mismatch");
        int i=0;
        for(int d:shape) if(dims.d[i++]!=d) throw std::runtime_error("Engine shape mismatch");
    }
public:
    explicit TensorRTPolicy(const std::string& path){
        try {
            std::ifstream file(path,std::ios::binary);
            if(!file) throw std::runtime_error("Cannot open TensorRT engine");
            std::vector<char> bytes((std::istreambuf_iterator<char>(file)),{});
            runtime_.reset(nvinfer1::createInferRuntime(logger_));
            if(!runtime_) throw std::runtime_error("Cannot create TensorRT runtime");
            engine_.reset(runtime_->deserializeCudaEngine(bytes.data(),bytes.size()));
            if(!engine_) throw std::runtime_error("Cannot deserialize engine");
            if(engine_->getNbIOTensors()!=3) throw std::runtime_error("Expected exactly 3 IO tensors");
            require("observation",nvinfer1::TensorIOMode::kINPUT,{1,49});
            require("history",nvinfer1::TensorIOMode::kINPUT,{1,50,49});
            require("action",nvinfer1::TensorIOMode::kOUTPUT,{1,12});
            context_.reset(engine_->createExecutionContext());
            if(!context_) throw std::runtime_error("Cannot create execution context");
            check(cudaStreamCreate(&stream_));
            check(cudaMalloc(&obs_,49*sizeof(float)));
            check(cudaMalloc(&history_,2450*sizeof(float)));
            check(cudaMalloc(&action_,12*sizeof(float)));
            if(!context_->setTensorAddress("observation",obs_) ||
               !context_->setTensorAddress("history",history_) ||
               !context_->setTensorAddress("action",action_)) throw std::runtime_error("Cannot bind IO");
        } catch(...) { cleanup(); throw; }
    }
    ~TensorRTPolicy(){ cleanup(); }
    void infer(const std::array<float,49>& obs,const std::array<float,2450>& history,
               std::array<float,12>& action){
        check(cudaMemcpyAsync(obs_,obs.data(),sizeof(obs),cudaMemcpyHostToDevice,stream_));
        check(cudaMemcpyAsync(history_,history.data(),sizeof(history),cudaMemcpyHostToDevice,stream_));
        if(!context_->enqueueV3(stream_)) throw std::runtime_error("enqueueV3 failed");
        check(cudaMemcpyAsync(action.data(),action_,sizeof(action),cudaMemcpyDeviceToHost,stream_));
        check(cudaStreamSynchronize(stream_));
    }
    TensorRTPolicy(const TensorRTPolicy&)=delete;
    TensorRTPolicy& operator=(const TensorRTPolicy&)=delete;
};
