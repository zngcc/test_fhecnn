#pragma	once
#include "seal/seal.h"
#include "SEALcomp.h"
#include "MinicompFunc.h"
#include "func.h"
#include "PolyUpdate.h"
#include "program.h"
#include "Bootstrapper.h"
#include "cnn_seal.h"
#include <omp.h>
#include <NTL/RR.h>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <ctime>
#include <filesystem>

// import parameters
void import_parameters_cifar10(vector<double> &linear_weight, vector<double> &linear_bias, vector<vector<double>> &conv_weight, vector<vector<double>> &bn_bias, vector<vector<double>> &bn_running_mean, vector<vector<double>> &bn_running_var, vector<vector<double>> &bn_weight, size_t layer_num, size_t end_num, const std::string &model_path = "../../pretrained_parameters/mpcnn_models/");
void import_parameters_cifar100(vector<double> &linear_weight, vector<double> &linear_bias, vector<vector<double>> &conv_weight, vector<vector<double>> &bn_bias, vector<vector<double>> &bn_running_mean, vector<vector<double>> &bn_running_var, vector<vector<double>> &bn_weight, vector<vector<double>> &shortcut_weight, vector<vector<double>> &shortcut_bn_bias, vector<vector<double>> &shortcut_bn_mean, vector<vector<double>> &shortcut_bn_var, vector<vector<double>> &shortcut_bn_weight, size_t layer_num, size_t end_num, const std::string &model_path = "../../pretrained_parameters/mpcnn_models/");

// cifar10, cifar100 integrated
void ResNet_cifar10_seal_sparse(size_t layer_num, size_t start_image_id, size_t end_image_id, std::string precison_par = "deg13", std::string model_path = "../../pretrained_parameters/mpcnn_models/", int threads = 1);
// void ResNet_cifar100_seal_sparse(size_t layer_num, size_t start_image_id, size_t end_image_id);