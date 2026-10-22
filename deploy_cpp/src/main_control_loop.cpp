// Timing/IO framework. MOCK ROBOT ONLY: no Unitree SDK commands are sent.
// C++17, TensorRT 10 optional, static FP32 IO with optional internal FP16 engine.
#include "DoubleBuffer.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#ifdef WITH_TENSORRT
#include "TensorRTPolicy.hpp"
#endif

using Clock=std::chrono::steady_clock;
std::int64_t nowNs(){return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();}
struct Sensor {
    // q,dq in joint order FL FR RL RR; q0 is supplied in actuator order FR FL RR RL.
    std::array<float,12> q{},dq{};
    std::array<float,3> gyro{},gravity{0,0,-1},command{};
    std::int64_t stamp{};
};
struct Command {
    std::array<float,12> target{};
    std::int64_t stamp{};
    std::uint64_t sequence{};
};
struct ControlConfig {
    std::array<float,12> q0{},kp{},kd{},limit{};
    ControlConfig(){kp.fill(35);kd.fill(.5);limit.fill(10);}
    void load(const std::string& path){
        std::ifstream f(path);
        if(!f) throw std::runtime_error("Cannot read control config");
        for(int i=0;i<12;++i){
            if(!(f>>q0[i]>>kp[i]>>kd[i]>>limit[i]) || !std::isfinite(q0[i]) ||
               !std::isfinite(kp[i]) || !std::isfinite(kd[i]) || !std::isfinite(limit[i]) ||
               kp[i]<0 || kd[i]<0 || limit[i]<=0) throw std::runtime_error("Invalid control config row");
        }
        std::string extra;
        if(f>>extra) throw std::runtime_error("Control config must contain exactly 12 rows of q0 kp kd torque_limit");
    }
};
// Joint -> actuator index, based on copied YAML; verify against materialized scene before real use.
constexpr std::array<int,12> jointToAct{3,4,5,0,1,2,9,10,11,6,7,8};

int main(int argc,char** argv){
    try {
        int seconds=5;
        std::string enginePath, configPath;
        for(int i=1;i<argc;++i){
            const std::string a=argv[i];
            if(a=="--engine" && i+1<argc) enginePath=argv[++i];
            else if(a=="--seconds" && i+1<argc) seconds=std::stoi(argv[++i]);
            else if(a=="--control-config" && i+1<argc) configPath=argv[++i];
            else throw std::runtime_error("Usage: control_loop [--seconds 5] [--engine file.engine --control-config file.txt]");
        }
        if(seconds<1 || seconds>3600) throw std::runtime_error("seconds must be 1..3600");
        ControlConfig cfg;
        if(!configPath.empty()) cfg.load(configPath);
        if(!enginePath.empty() && configPath.empty()) throw std::runtime_error("Engine requires matching home/PD config");
#ifdef WITH_TENSORRT
        std::unique_ptr<TensorRTPolicy> policy;
        if(!enginePath.empty()){
            policy=std::make_unique<TensorRTPolicy>(enginePath);
            // Trigger lazy CUDA/kernel initialization before timed control starts.
            std::array<float,49> warmObs{};
            std::array<float,2450> warmHistory{};
            std::array<float,12> warmAction{};
            policy->infer(warmObs,warmHistory,warmAction);
            for(float a:warmAction) if(!std::isfinite(a)) throw std::runtime_error("Non-finite warm-up action");
            std::cout<<"TensorRT engine loaded; IO checked; warm-up complete\n";
        }
#else
        if(!enginePath.empty()) throw std::runtime_error("Rebuild with WITH_TENSORRT=ON");
#endif
        std::cout<<"MOCK ROBOT ONLY; "<<(enginePath.empty()?"zero-action mock policy":"TensorRT policy")
                 <<"; 50 Hz policy / 1000 Hz PD requested\n";
        DoubleBuffer<Sensor> sensors;
        DoubleBuffer<Command> commands;
        std::atomic<bool> stop{false};
        std::atomic<std::uint64_t> fastTicks{0},slowTicks{0},stale{0},fastMiss{0},slowMiss{0};
        // Thread-owned statistics; main reads them only after both joins.
        std::uint64_t commandsPublished=0,commandsConsumed=0;
        float maxActionAbs=0.f,maxTargetOffset=0.f,maxTorqueAbs=0.f;
#ifdef WITH_TENSORRT
        std::uint64_t inferenceCalls=0;
        double inferenceTotalMs=0.,inferenceMaxMs=0.;
#endif
        std::string failure; // Written only by slow thread; read after join.
        const auto end=Clock::now()+std::chrono::seconds(seconds);
        std::thread fast([&]{
            Sensor s;
            for(int j=0;j<12;++j) s.q[j]=cfg.q0[jointToAct[j]];
            Command active;
            auto deadline=Clock::now();
            while(!stop.load(std::memory_order_acquire) && Clock::now()<end){
                s.stamp=nowNs();
                sensors.tryWrite(s);
                Command candidate;
                if(commands.tryRead(candidate) && candidate.sequence>active.sequence){
                    active=candidate;
                    ++commandsConsumed;
                }
                const bool fresh=active.sequence>0 && s.stamp>=active.stamp && s.stamp-active.stamp<100000000;
                if(!fresh) ++stale;
                for(int j=0;j<12;++j){
                    const int a=jointToAct[j];
                    float torque=fresh?cfg.kp[a]*(active.target[a]-s.q[j])-cfg.kd[a]*s.dq[j]:0.f;
                    torque=std::clamp(torque,-cfg.limit[a],cfg.limit[a]);
                    maxTorqueAbs=std::max(maxTorqueAbs,std::abs(torque));
                    // Mock plant, no gravity/contact. Replace with timestamped SDK sensor/command IO.
                    s.dq[j]+=(torque-.2f*s.dq[j])/.2f*.001f;
                    s.q[j]+=s.dq[j]*.001f;
                }
                ++fastTicks;
                deadline+=std::chrono::milliseconds(1);
                if(Clock::now()>deadline){++fastMiss;deadline=Clock::now();}
                std::this_thread::sleep_until(deadline);
            }
            stop.store(true,std::memory_order_release);
        });
        std::thread slow([&]{
            try {
                std::array<float,2450> history{};
                std::array<float,12> previous{};
                auto deadline=Clock::now();
                std::uint64_t sequence=0;
                float phase=0.f;
                while(!stop.load(std::memory_order_acquire)){
                    Sensor s;
                    if(sensors.tryRead(s) && nowNs()-s.stamp<20000000){
                        std::array<float,49> obs{};
                        for(int i=0;i<3;++i){obs[i]=s.gyro[i];obs[3+i]=s.gravity[i];obs[42+i]=s.command[i];}
                        for(int j=0;j<12;++j){obs[6+j]=s.q[j]-cfg.q0[jointToAct[j]];obs[18+j]=s.dq[j];obs[30+j]=previous[j];}
                        // Reset phase on process start. Real runtime must match environment phase resets.
                        constexpr float offsets[4]={0,.5f,.5f,0};
                        for(int i=0;i<4;++i) obs[45+i]=std::fmod(phase+offsets[i],1.f);
                        // Shift left explicitly; no overlapping std::copy ranges.
                        for(int i=0;i<2450-49;++i) history[i]=history[i+49];
                        std::copy(obs.begin(),obs.end(),history.end()-49);
                        std::array<float,12> action{};
#ifdef WITH_TENSORRT
                        if(policy){
                            const auto started=Clock::now();
                            policy->infer(obs,history,action);
                            const double elapsed=std::chrono::duration<double,std::milli>(Clock::now()-started).count();
                            ++inferenceCalls;
                            inferenceTotalMs+=elapsed;
                            inferenceMaxMs=std::max(inferenceMaxMs,elapsed);
                        }
#endif
                        Command cmd;
                        cmd.stamp=s.stamp; // Sensor age includes inference latency.
                        cmd.sequence=++sequence;
                        for(int i=0;i<12;++i){
                            if(!std::isfinite(action[i])) throw std::runtime_error("Non-finite model action");
                            action[i]=std::clamp(action[i],-1.f,1.f);
                            cmd.target[i]=cfg.q0[i]+.25f*action[i];
                            maxActionAbs=std::max(maxActionAbs,std::abs(action[i]));
                            maxTargetOffset=std::max(maxTargetOffset,std::abs(cmd.target[i]-cfg.q0[i]));
                        }
                        if(commands.tryWrite(cmd)){
                            previous=action;
                            ++commandsPublished;
                        }
                        phase=std::fmod(phase+.04f,1.f); // Same float32 0.02s * 2Hz update as training.
                        ++slowTicks;
                    }
                    deadline+=std::chrono::milliseconds(20);
                    if(Clock::now()>deadline){++slowMiss;deadline=Clock::now();}
                    std::this_thread::sleep_until(deadline);
                }
            } catch(const std::exception& e){failure=e.what();stop.store(true,std::memory_order_release);}
        });
        fast.join();slow.join();
        std::cout<<"PD ticks="<<fastTicks<<", policy ticks="<<slowTicks<<", stale ticks="<<stale
                 <<", PD deadline misses="<<fastMiss<<", policy deadline misses="<<slowMiss<<"\n";
        std::cout<<"Commands published="<<commandsPublished<<", consumed="<<commandsConsumed
                 <<", max |action|="<<maxActionAbs<<", max |target-q0|="<<maxTargetOffset
                 <<" rad, max |torque|="<<maxTorqueAbs<<" N m\n";
#ifdef WITH_TENSORRT
        if(inferenceCalls>0) std::cout<<"TensorRT calls="<<inferenceCalls
            <<", mean inference="<<inferenceTotalMs/inferenceCalls<<" ms, max inference="<<inferenceMaxMs<<" ms\n";
#endif
        if(!failure.empty()) throw std::runtime_error(failure);
        return 0;
    } catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
}
