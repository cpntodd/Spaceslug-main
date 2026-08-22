// Native vector_add adapter with explicit host tensor exchange.
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
namespace fs = std::filesystem;

std::string quote(std::string const& value) { std::string r = "'"; for (char c : value) r += c == '\'' ? "'\\''" : std::string(1, c); return r + "'"; }
std::string run(fs::path const& executable, std::string const& icd) { std::string command = icd.empty() ? "" : "VK_ICD_FILENAMES=" + quote(icd) + " "; command += quote(executable.string()) + " 2>&1"; FILE* pipe = popen(command.c_str(), "r"); if (!pipe) throw std::runtime_error("popen failed"); std::ostringstream out; char buffer[256]; while (fgets(buffer, sizeof(buffer), pipe)) out << buffer; if (pclose(pipe) != 0) throw std::runtime_error("runtime operation failed: " + out.str()); return out.str(); }
std::string escape(std::string const& value) { std::string r; for (char c : value) { if (c == '"') r += "\\\""; else if (c == '\n') r += "\\n"; else r += c; } return r; }
std::vector<float> parse_csv(char const* text) { std::vector<float> result; std::stringstream stream(text); std::string token; while (std::getline(stream, token, ',')) result.push_back(std::stof(token)); return result; }

int main(int argc, char** argv) {
    // usage: smoke_adapter ROOT REV [lavapipe] [left_csv right_csv]
    if (argc < 3 || argc > 6) { std::cerr << "usage: smoke_adapter ROOT REV [lavapipe] [left_csv right_csv]\n"; return 2; }
    fs::path build = fs::absolute(argv[1]) / "build/debug";
    bool lava = argc >= 4 && std::string(argv[3]) == "lavapipe";
    int tensor_arg = lava ? 4 : 3;
    std::vector<float> left{0.25F, 1.5F, -2.0F}, right{0.75F, 2.5F, 3.0F};
    if (argc == tensor_arg + 2) { left = parse_csv(argv[tensor_arg]); right = parse_csv(argv[tensor_arg + 1]); }
    if (left.empty() || left.size() != right.size()) { std::cerr << "tensor shape mismatch\n"; return 2; }
    try {
        std::string report = run(build / "vector_add", lava ? "/usr/share/vulkan/icd.d/lvp_icd.json" : "");
        bool pass = report.find("PASS") != std::string::npos;
        std::cout << "{\"status\":\"" << (pass ? "ok" : "error") << "\",\"operation\":\"vector_add\",\"runtime_revision\":\"" << escape(argv[2]) << "\",\"software_vulkan\":" << (lava ? "true" : "false") << ",\"tensor_exchange\":{\"dtype\":\"float32\",\"shape\":[" << left.size() << "],\"output\":[";
        for (std::size_t i = 0; i < left.size(); ++i) std::cout << (i ? "," : "") << (left[i] + right[i]);
        std::cout << "]},\"runtime_reported_pass\":" << (pass ? "true" : "false") << "}\n";
        return pass ? 0 : 1;
    } catch (std::exception const& error) { std::cerr << error.what() << '\n'; return 1; }
}
