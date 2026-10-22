#include "ControlMapping.hpp"
#include <array>
#include <cmath>
#include <iostream>

int main() {
    // Independent expected destinations; nonuniform home values expose swaps.
    constexpr std::array<int, 12> expected{3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8};
    std::array<float, 12> home{};
    for (int index = 0; index < 12; ++index) home[index] = 0.1f * index;
    for (int joint = 0; joint < 12; ++joint) {
        std::array<float, 12> action{};
        action[joint] = 1.0f;
        const auto target = rma::actuatorTargets(home, action);
        for (int actuator = 0; actuator < 12; ++actuator) {
            const float wanted = home[actuator] + (actuator == expected[joint] ? 0.25f : 0.0f);
            if (std::abs(target[actuator] - wanted) > 1e-6f) {
                std::cerr << "Wrong target for policy joint " << joint
                          << ", actuator " << actuator << '\n';
                return 1;
            }
        }
    }
    std::cout << "All 12 one-hot policy actions reach the expected actuator\n";
    return 0;
}
