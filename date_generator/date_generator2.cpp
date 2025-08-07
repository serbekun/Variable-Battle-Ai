#include <iostream>
#include <fstream>
#include <random>
#include <thread>
#include <vector>
#include <atomic>
#include <mutex>
#include <filesystem>
#include <cmath>
#include <cstring>

namespace fs = std::filesystem;

constexpr int64_t TARGET_RECORDS = 1'000'000'0;
const std::string DATA_SET_SAVE_PATH = "../date_packs/";
const std::string DATA_SET_NAME = "data_dg2_cpp.ndjson";
const std::string DATA_END_SAVE_PATH = DATA_SET_SAVE_PATH + DATA_SET_NAME;
constexpr int MAX_ROUNDS = 100;
constexpr int BATCH_SIZE = 1024;

std::mutex file_mutex;
std::atomic<int64_t> records_generated(0);
std::ofstream out_file;

enum class AIStyle {
    Balanced,
    Aggressive,
    Defensive
};

AIStyle g_ai_style = AIStyle::Balanced;

struct ThreadRNG {
    std::mt19937 engine;
    std::uniform_int_distribution<int> hp_dist{90, 159};
    std::uniform_int_distribution<int> attack_dist{25, 84};
    std::uniform_int_distribution<int> heal_dist{15, 44};
    std::uniform_real_distribution<float> prob_dist{0.0f, 1.0f};
    std::discrete_distribution<int> action_dist;

    ThreadRNG() {
        std::random_device rd;
        engine.seed(rd());
    }
};

int choose_action(int hp, int enemy_hp, int last_action, ThreadRNG& trng, AIStyle style) {
    std::vector<int> weights;

    if (style == AIStyle::Aggressive) {
        weights = {50, 10, 5, 25, 10};
    } else if (style == AIStyle::Defensive) {
        weights = {10, 30, 30, 10, 20};
    } else {
        if (hp < 30) {
            if (enemy_hp < 40) weights = {40, 10, 10, 30, 10};
            else weights = {10, 25, 50, 5, 10};
        } else if (hp > enemy_hp + 30) {
            weights = {50, 15, 5, 25, 5};
        } else {
            weights = {35, 20, 20, 15, 10};
        }
    }

    if (last_action == 1) {
        weights[1] = std::max(5, weights[1] - 15);
    }

    trng.action_dist.param(
        std::discrete_distribution<int>::param_type(weights.begin(), weights.end())
    );
    return trng.action_dist(trng.engine);
}

void simulate_battle(std::vector<std::string>& buffer, ThreadRNG& trng) {

    int human_hp = trng.hp_dist(trng.engine);
    int model_hp = trng.hp_dist(trng.engine);
    int human_attack = trng.attack_dist(trng.engine);
    int model_attack = trng.attack_dist(trng.engine);
    int human_heal = trng.heal_dist(trng.engine);
    int model_heal = trng.heal_dist(trng.engine);

    int round_count = 1;
    int last_human_action = -1;
    int last_model_action = -1;
    std::string battle_result = "draw";

    std::vector<std::string> round_data;
    round_data.reserve(MAX_ROUNDS);

    while (human_hp > 0 && model_hp > 0 && round_count <= MAX_ROUNDS) {

        int human_action = choose_action(
            human_hp, model_hp, last_human_action, trng, g_ai_style
        );
        int model_action = choose_action(
            model_hp, human_hp, last_model_action, trng, g_ai_style
        );

        bool human_block = (human_action == 1);
        bool model_block = (model_action == 1);
        int dmg_to_model = 0;
        int dmg_to_human = 0;
        int heal_to_human = 0;
        int heal_to_model = 0;

        if (human_action == 0) {
            dmg_to_model = human_attack;
        } else if (human_action == 2) {
            heal_to_human = human_heal;
        } else if (human_action == 3) {
            if (trng.prob_dist(trng.engine) > 0.3f) {
                dmg_to_model = static_cast<int>(human_attack * 1.8f);
            }
        } else if (human_action == 4) {
            if (trng.prob_dist(trng.engine) > 0.4f) {
                heal_to_human = static_cast<int>(human_heal * 1.5f);
            }
        }

        if (model_action == 0) {
            dmg_to_human = model_attack;
        } else if (model_action == 2) {
            heal_to_model = model_heal;
        } else if (model_action == 3) {
            if (trng.prob_dist(trng.engine) > 0.3f) {
                dmg_to_human = static_cast<int>(model_attack * 1.8f);
            }
        } else if (model_action == 4) {
            if (trng.prob_dist(trng.engine) > 0.4f) {
                heal_to_model = static_cast<int>(model_heal * 1.5f);
            }
        }

        if (model_block) dmg_to_model = std::max(1, dmg_to_model / 2);
        if (human_block) dmg_to_human = std::max(1, dmg_to_human / 2);

        int new_human_hp = std::max(0, human_hp + heal_to_human - dmg_to_human);
        int new_model_hp = std::max(0, model_hp + heal_to_model - dmg_to_model);

        std::string json_str = "{";
        json_str += "\"round\":" + std::to_string(round_count) + ",";
        json_str += "\"human\":{";
        json_str += "\"hp\":" + std::to_string(human_hp) + ",";
        json_str += "\"new_hp\":" + std::to_string(new_human_hp) + ",";
        json_str += "\"attack\":" + std::to_string(human_attack) + ",";
        json_str += "\"heal\":" + std::to_string(human_heal) + ",";
        json_str += "\"block\":" + std::string(human_block ? "true" : "false") + ",";
        json_str += "\"action\":" + std::to_string(human_action) + ",";
        json_str += "\"damage_dealt\":" + std::to_string(dmg_to_model) + ",";
        json_str += "\"healing_done\":" + std::to_string(heal_to_human);
        json_str += "},";
        json_str += "\"model\":{";
        json_str += "\"hp\":" + std::to_string(model_hp) + ",";
        json_str += "\"new_hp\":" + std::to_string(new_model_hp) + ",";
        json_str += "\"attack\":" + std::to_string(model_attack) + ",";
        json_str += "\"heal\":" + std::to_string(model_heal) + ",";
        json_str += "\"block\":" + std::string(model_block ? "true" : "false") + ",";
        json_str += "\"action\":" + std::to_string(model_action) + ",";
        json_str += "\"damage_dealt\":" + std::to_string(dmg_to_human) + ",";
        json_str += "\"healing_done\":" + std::to_string(heal_to_model);
        json_str += "}}";

        round_data.push_back(std::move(json_str));

        human_hp = new_human_hp;
        model_hp = new_model_hp;
        last_human_action = human_action;
        last_model_action = model_action;
        round_count++;
    }

    if (human_hp <= 0 && model_hp > 0) battle_result = "model_win";
    else if (model_hp <= 0 && human_hp > 0) battle_result = "human_win";

    for (auto& json_str : round_data) {
        if (records_generated >= TARGET_RECORDS) break;

        size_t pos = json_str.find('{');
        json_str.insert(pos + 1, "\"battle_result\":\"" + battle_result + "\",");

        buffer.push_back(std::move(json_str));
        records_generated++;
    }
}

void worker() {
    ThreadRNG trng;
    std::vector<std::string> local_buffer;
    local_buffer.reserve(BATCH_SIZE);

    while (records_generated < TARGET_RECORDS) {
        simulate_battle(local_buffer, trng);

        if (local_buffer.size() >= BATCH_SIZE) {
            std::lock_guard<std::mutex> lock(file_mutex);
            for (auto& record : local_buffer) {
                out_file << record << '\n';
            }
            local_buffer.clear();
        }
    }

    if (!local_buffer.empty()) {
        std::lock_guard<std::mutex> lock(file_mutex);
        for (auto& record : local_buffer) {
            out_file << record << '\n';
        }
    }
}

int main(int argc, char* argv[]) {
    for (int i = 1; i < argc; ++i) {
        if (std::strncmp(argv[i], "--style=", 8) == 0) {
            std::string style_arg = argv[i] + 8;
            if (style_arg == "aggressive") {
                g_ai_style = AIStyle::Aggressive;
            } else if (style_arg == "defensive") {
                g_ai_style = AIStyle::Defensive;
            } else if (style_arg == "balanced") {
                g_ai_style = AIStyle::Balanced;
            } else {
                std::cerr << "Unknown style: " << style_arg << std::endl;
                return 1;
            }
        }
    }

    std::cout << "AI style set to: ";
    switch (g_ai_style) {
        case AIStyle::Aggressive: std::cout << "Aggressive\n"; break;
        case AIStyle::Defensive: std::cout << "Defensive\n"; break;
        default: std::cout << "Balanced\n"; break;
    }

    fs::create_directories(DATA_SET_SAVE_PATH);

    out_file.open(DATA_END_SAVE_PATH);
    if (!out_file) {
        std::cerr << "Error opening file: " << DATA_END_SAVE_PATH << std::endl;
        return 1;
    }

    unsigned num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 1;
    std::vector<std::thread> threads;

    for (unsigned i = 0; i < num_threads; ++i) {
        threads.emplace_back(worker);
    }

    std::cout << "Generating data:\n";
    while (records_generated < TARGET_RECORDS) {
        int progress = static_cast<int>(
            (static_cast<double>(records_generated) / TARGET_RECORDS * 50));
        std::cout << "\r[" << std::string(progress, '=') 
                  << std::string(50 - progress, ' ') << "] "
                  << records_generated << "/" << TARGET_RECORDS;
        std::cout.flush();
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    std::cout << "\r[" << std::string(50, '=') << "] " 
              << TARGET_RECORDS << "/" << TARGET_RECORDS << std::endl;

    for (auto& t : threads) {
        t.join();
    }

    out_file.close();
    std::cout << "Generated data saved to " << DATA_END_SAVE_PATH << std::endl;
    return 0;
}