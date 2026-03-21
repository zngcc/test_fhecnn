#pragma once

#include <NTL/RR.h>
//#include "Remez.h"
#include "MinicompRemez.h"
//#include "func.h"
#include "MinicompFunc.h"
#include "PolyUpdate.h"
#include "program.h"
using namespace minicomp;

void testRemez();
void testReLU(size_t deg);
RR GetError(int d, RR t, bool is_first_function, int type, RR scale);
//void GetCoeff(int d, RR t, RR *coeff, bool is_first_function, int type, RR scale);
RR GetErrorCoeff(int d, RR t, vector<RR> &coeff, bool is_first_function, int type, RR scale);
//RR GetErrorCoeff_extpoint(int d, RR t, vector<RR> &coeff, vector<RR> &ext_xpos, bool is_first_function, int type, RR scale);
struct DegConfig {
    string name;
    vector<int> degs;
};

long coeff_number(long deg, Tree& tree);
void generate_minimax_coefficients(size_t deg, const string& output_file);
long count_total_coefficients(const vector<int>& degs, evaltype ev_type);
void print_tree_info(const vector<int>& degs, evaltype ev_type);
void generate_single_deg(const DegConfig& config);
void generate_all_degs();
void generate_d9_coefficients();
void generate_d13_coefficients();