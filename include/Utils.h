#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
#include <regex>
#include <yaml-cpp/yaml.h>
#include <TFile.h>
#include <TH2.h>
#include <TGraph2D.h>
#include <TCanvas.h>
#include <TColor.h>
#include <TStyle.h>

std::pair<int, int> get_coordinates_8x8(int ref_number) {
    ref_number -= 1;
    if (0 <= ref_number && ref_number < 64) {
        int y_quotient = ref_number / 8;
        int y_remainder = ref_number % 8;
        int y = 8 - y_quotient;
        int x;
        if (y % 2 == 1) {
            x = (8 - y_remainder) % 8;
            if (x == 0) x = 8;
        } else {
            x = y_remainder + 1;
        }
        return std::make_pair(x, y);
    } else {
        return std::make_pair(-1, -1);
    }
}

std::pair<int, int> get_coordinates_4x4(int ref_number) {
    ref_number -= 1;
    if (0 <= ref_number && ref_number < 16) {
        int y_quotient = ref_number / 4;
        int y_remainder = ref_number % 4;
        int y = 4 - y_quotient;
        int x = y_remainder + 1;
        return std::make_pair(x, y);
    } else {
        return std::make_pair(-1, -1);
    }
}

std::pair<double, double> convert_coordinates_8x8(int x_index, int y_index, const YAML::Node& yaml_data) {
    // ... (implementation similar to Python)

    return std::make_pair(x, y);
}

std::pair<double, double> convert_coordinates_4x4(double x_po, double y_po, double x, double y, const YAML::Node& yaml_data, std::pair<double, double> original = std::make_pair(0.0, 0.0)) {
    // ... (implementation similar to Python)

    return std::make_pair(x_real, y_real);
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_directory_or_file>" << std::endl;
        return 1;
    }

    std::string path = argv[1];
    // ... (rest of the main function, similar to Python)

    return 0;
}

