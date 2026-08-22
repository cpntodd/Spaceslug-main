// Native smoke adapter for the pinned Spaceslug runtime.
// It intentionally keeps the first boundary coarse-grained: the runtime owns
// Vulkan and CPU-reference correctness; this process owns structured reporting.
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>

namespace fs = std::filesystem;

std::string quote(std::string const& value) {
    std::string result = "'";
    for (char c : value) {
        if (c == '\'') result += "'\\''";
        else result += c;
    }
    return result + "'";
}

std::string run(fs::path const& executable, std::string const& icd) {
    std::string command;
    if (!icd.empty()) command += "VK_ICD_FILENAMES=" + quote(icd) + " ";
    command += quote(executable.string()) + " 2>&1";
    FILE* pipe = popen(command.c_str(), "r");
    if (pipe == nullptr) throw std::runtime_error("popen failed");
    std::ostringstream output;
    char buffer[256];
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) output << buffer;
    int status = pclose(pipe);
    if (status != 0) throw std::runtime_error("runtime operation failed: " + output.str());
    return output.str();
}

std::string json_escape(std::string value) {
    std::string result;
    for (char c : value) {
        if (c == '"') result += "\\\"";
        else if (c == '\n') result += "\\n";
        else result += c;
    }
    return result;
}

int main(int argc, char** argv) {
    if (argc < 3 || argc > 4) {
        std::cerr << "usage: smoke_adapter RUNTIME_ROOT RUNTIME_REVISION [lavapipe]\n";
        return 2;
    }
    fs::path root = fs::absolute(argv[1]);
    fs::path build = root / "build/debug";
    bool lavapipe = argc == 4 && std::string(argv[3]) == "lavapipe";
    try {
        std::string device = run(build / "smoke", lavapipe ? "/usr/share/vulkan/icd.d/lvp_icd.json" : "");
        std::string result = run(build / "vector_add", lavapipe ? "/usr/share/vulkan/icd.d/lvp_icd.json" : "");
        std::cout << "{\"status\":\"ok\",\"operation\":\"vector_add\",\"backend\":\"spaceslug-native\",\"runtime_revision\":\""
                  << json_escape(argv[2]) << "\",\"software_vulkan\":" << (lavapipe ? "true" : "false")
                  << ",\"runtime_report\":\"" << json_escape(result) << "\",\"device_report\":\"" << json_escape(device) << "\"}\n";
        return 0;
    } catch (std::exception const& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
