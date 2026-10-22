#pragma once

#include <array>

namespace rma {
// Policy/observations: FL FR RL RR. Scene actuator storage: FR FL RR RL.
// Per leg: hip, thigh, calf. Verify the scene contract before hardware use.
inline constexpr std::array<int, 12> policyToActuator{3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8};

inline std::array<float, 12> actuatorTargets(
    const std::array<float, 12>& home,
    const std::array<float, 12>& policyAction,
    float scale = 0.25f) {
    std::array<float, 12> target{};
    for (int joint = 0; joint < 12; ++joint) {
        const int actuator = policyToActuator[joint];
        target[actuator] = home[actuator] + scale * policyAction[joint];
    }
    return target;
}
}  // namespace rma
