#pragma once
#include <array>
#include <atomic>
#include <type_traits>

// Single producer / single consumer, latest-value mailbox, exactly two payload slots.
// Publication alone is insufficient: slot ownership prevents payload data races.
// Nonblocking try operations: caller retains last valid value on read failure.
template<class T> class DoubleBuffer {
    static_assert(std::is_trivially_copyable_v<T>, "Payload must be trivially copyable");
    static_assert(std::atomic<int>::is_always_lock_free, "Requires lock-free atomic<int>");
    enum { Free=0, Writing=1, Ready=2, Reading=3 };
    struct alignas(64) Slot {
        std::atomic<int> state{Free};
        T value{};
    };
    std::array<Slot,2> slots_{};
    std::atomic<int> published_{-1};
public:
    bool tryWrite(const T& value) noexcept {
        const int previous=published_.load(std::memory_order_acquire);
        const int first=previous<0?0:1-previous;
        for(int attempt=0;attempt<2;++attempt){
            const int index=attempt==0?first:1-first;
            auto& slot=slots_[index];
            int state=slot.state.load(std::memory_order_acquire);
            if(state!=Free && state!=Ready) continue;
            if(!slot.state.compare_exchange_strong(state,Writing,
                    std::memory_order_acq_rel,std::memory_order_relaxed)) continue;
            slot.value=value;
            // Announce while Writing, then release the payload. A reader of an
            // old index cannot consume a recycled slot before its publication;
            // Ready acquire also orders the publication before its next read.
            published_.store(index,std::memory_order_release);
            slot.state.store(Ready,std::memory_order_release);
            return true;
        }
        return false;
    }
    bool tryRead(T& value) noexcept {
        const int index=published_.load(std::memory_order_acquire);
        if(index<0) return false;
        auto& slot=slots_[index];
        int state=Ready;
        if(!slot.state.compare_exchange_strong(state,Reading,
                std::memory_order_acq_rel,std::memory_order_relaxed)) return false;
        value=slot.value;
        slot.state.store(Free,std::memory_order_release);
        return true;
    }
    DoubleBuffer()=default;
    DoubleBuffer(const DoubleBuffer&)=delete;
    DoubleBuffer& operator=(const DoubleBuffer&)=delete;
};
