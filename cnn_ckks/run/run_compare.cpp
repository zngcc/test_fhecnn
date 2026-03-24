// basic
#include <iostream>
#include <NTL/RR.h>
#include <cmath>
#include <map>
#include <fstream>
#include "PolyUpdate.h"
#include "program.h"
#include "seal/seal.h"
#include "SEALcomp.h"
#include "RemezApp.h"

using namespace seal;
using namespace std;
using namespace NTL;

// Configuration structure (same as original)
struct ReLUConfig {
    long alpha;
    int comp_no;
    vector<int> deg;
    double scaled_val;
    int remaining_level;
};

// Pre‑defined configurations
map<string, ReLUConfig> relu_configs = {
    {"deg13", {13, 3, {15, 15, 27}, 1.7, 16}},
    {"deg9",  {9,  2, {15, 15},     1.7, 11}},
    // add more if needed
};

// ----------------------------------------------------------------------
// 1. Generate coefficients (polynomials in power basis) using Remez
// ----------------------------------------------------------------------
void generate_coefficients(const string& config_name) {
    auto it = relu_configs.find(config_name);
    if (it == relu_configs.end()) {
        cerr << "Unknown configuration: " << config_name << endl;
        return;
    }
    const ReLUConfig& cfg = it->second;
    cout << "Generating coefficients for " << config_name << endl;
    cout << "degs: ";
    for (int d : cfg.deg) cout << d << " ";
    cout << endl;

    // Remez parameters (same as in generate_single_deg)
    long prec = 9;
    long iteration = 400;
    long RRprec = 5000;
    int num = 10000;
    long type = 1;
    RR scale = RR(1);

    size_t inter_num = 1;
    vector<RR> inter_start, inter_end;
    inter_start.emplace_back(RR(-1));
    inter_end.emplace_back(RR(1));

    RR::SetPrecision(RRprec);
    RR sc = RR(2.0) / RR(num);

    string output_file = "../approx_para/" + config_name + "_new.txt";
    ofstream out(output_file);
    if (!out) {
        cerr << "Cannot open output file: " << output_file << endl;
        return;
    }

    for (size_t i = 0; i < cfg.deg.size(); ++i) {
        long deg = cfg.deg[i];
        cout << "Processing component " << i << " (degree " << deg << ")..." << endl;

        // Run Remez for ReLU approximation on [-1,1]
        Remez rm(ReLU, inter_num, inter_start, inter_end, sc, prec, deg, iteration, type, scale, true);
        rm.test();
        RR maxerr = rm.getmaxerr();
        cout << "  maxerr: " << maxerr << endl;

        vector<RR> chebcoeff, coeff;
        rm.getcoeff(chebcoeff);

        Polynomial poly(deg, chebcoeff, "cheb");
        poly.cheb_to_power();            // convert to power basis
        poly.get_coeff(coeff);           // coeff[0] = constant term, coeff[deg] = leading coefficient

        cout << "  Writing " << coeff.size() << " coefficients..." << endl;
        for (auto num : coeff) {
            out << num << endl;
        }
    }

    out.close();
    cout << "Coefficients saved to " << output_file << endl;
}

// ----------------------------------------------------------------------
// 2. Test approximation accuracy of the composite polynomial
// ----------------------------------------------------------------------
void test_approximation(const string& config_name) {
    auto it = relu_configs.find(config_name);
    if (it == relu_configs.end()) {
        cerr << "Unknown configuration: " << config_name << endl;
        return;
    }
    const ReLUConfig& cfg = it->second;
    cout << "Testing approximation for " << config_name << endl;
    cout << "degs: ";
    for (int d : cfg.deg) cout << d << " ";
    cout << endl;
    cout << "alpha = " << cfg.alpha
         << ", expected error < 2^-" << cfg.alpha << " = " << pow(2.0, -cfg.alpha) << endl;

    // Read coefficients from file
    string coeff_file = "../approx_para/" + config_name + "_new.txt";
    ifstream in(coeff_file);
    if (!in) {
        cerr << "Cannot open coefficient file: " << coeff_file << endl;
        return;
    }

    vector<vector<RR>> coeffs;   // coeffs[i] = coefficients of i‑th component (power basis)
    for (size_t i = 0; i < cfg.deg.size(); ++i) {
        int deg = cfg.deg[i];
        vector<RR> poly_coeff;
        RR c;
        for (int j = 0; j <= deg; ++j) {
            if (!(in >> c)) {
                cerr << "Error reading coefficient " << j << " for component " << i << endl;
                return;
            }
            poly_coeff.push_back(c);
        }
        coeffs.push_back(poly_coeff);
    }

    // High precision for error measurement
    long prec = 300;
    RR::SetPrecision(prec);

    int num_points = 20000;           // dense sampling
    RR max_err = RR(0);
    RR x;

    cout << "\nSample evaluation (first 10 points):" << endl;
    cout << setw(15) << "x" << setw(20) << "poly(x)" << setw(20) << "ReLU(x)" << setw(20) << "error" << endl;
    cout << string(75, '-') << endl;

    int printed = 0;
    const int print_limit = 10;

    for (int k = 0; k <= num_points; ++k) {
        x = RR(-1) + RR(2) * RR(k) / RR(num_points);

        // Composite evaluation: y = P_{n-1}( ... P_1(P_0(x)) ... )
        RR y = x;
        for (int i = coeffs.size() - 1; i >= 0; --i) {
            // Horner evaluation of the i‑th polynomial at y
            RR val = RR(0);
            const vector<RR>& poly = coeffs[i];
            for (int j = poly.size() - 1; j >= 0; --j) {
                val = val * y + poly[j];
            }
            y = val;
        }

        RR true_val = ReLU(x);
        RR err = abs(y - true_val);
        if (err > max_err) max_err = err;

        // Print first few points
        if (printed < print_limit) {
            cout << setw(15) << x.toString() 
                 << setw(20) << y.toString() 
                 << setw(20) << true_val.toString() 
                 << setw(20) << err.toString() << endl;
            printed++;
        }
    }

    cout << "\nMaximum approximation error: " << max_err << endl;
    double theoretical_bound = pow(2.0, -cfg.alpha);
    cout << "Theoretical bound: " << theoretical_bound << endl;
    if (max_err < RR(theoretical_bound)) {
        cout << "PASS: error within bound." << endl;
    } else {
        cout << "FAIL: error exceeds bound." << endl;
    }
}

// ----------------------------------------------------------------------
// 3. Original full SEAL evaluation (extracted from the original main)
// ----------------------------------------------------------------------
void run_full_test(const string& config_name) {
    auto it = relu_configs.find(config_name);
    if (it == relu_configs.end()) {
        cerr << "Unknown configuration: " << config_name << endl;
        return;
    }

    long alpha = it->second.alpha;
    int comp_no = it->second.comp_no;
    vector<int> deg = it->second.deg;
    double scaled_val = it->second.scaled_val;
    int remaining_level = it->second.remaining_level;

    cout << "Using configuration: " << config_name << endl;
    cout << "alpha = " << alpha << ", comp_no = " << comp_no << ", deg = ";
    for (int d : deg) cout << d << " ";
    cout << ", scaled_val = " << scaled_val << endl;

    int boot_level = 0;
    int total_level = remaining_level + boot_level;

    long scalingfactor = 45;          // log2 of scaling factor
    double eta = pow(2.0, -15);
    double max_factor = 16;
    evaltype eval_type = evaltype::oddbaby;
    RR::SetOutputPrecision(25);

    vector<Tree> tree;
    for (int i = 0; i < comp_no; i++) {
        Tree tr;
        if (eval_type == evaltype::oddbaby)
            upgrade_oddbaby(deg[i], tr);
        else if (eval_type == evaltype::baby)
            upgrade_baby(deg[i], tr);
        else
            throw invalid_argument("evaluation type is not correct");
        tree.emplace_back(tr);
        tr.print();
    }

    long log_modulus = scalingfactor;
    EncryptionParameters parms(scheme_type::ckks);
    size_t poly_modulus_degree = 65536;
    long n = poly_modulus_degree / 2;
    parms.set_poly_modulus_degree(poly_modulus_degree);

    vector<int> modulus_list;
    modulus_list.emplace_back(60);
    for (int i = 0; i < total_level; i++) modulus_list.emplace_back(log_modulus);
    modulus_list.emplace_back(60);
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, modulus_list));

    SEALContext context(parms);
    chrono::high_resolution_clock::time_point time_start, time_end;
    chrono::microseconds time_diff;

    KeyGenerator keygen(context);
    PublicKey public_key;
    keygen.create_public_key(public_key);
    auto secret_key = keygen.secret_key();
    RelinKeys relin_keys;
    keygen.create_relin_keys(relin_keys);
    GaloisKeys gal_keys;
    keygen.create_galois_keys(gal_keys);

    Encryptor encryptor(context, public_key);
    CKKSEncoder encoder(context);
    Evaluator evaluator(context, encoder);
    Decryptor decryptor(context, secret_key);

    vector<double> m_x(n);
    for (int i = 0; i < n; i++)
        m_x[i] = -1.0 + 2.0 * static_cast<double>(i) / static_cast<double>(n - 1);

    Plaintext plain_x;
    double scale = pow(2.0, log_modulus);
    encoder.encode(m_x, scale, plain_x);
    Ciphertext cipher_x;
    encryptor.encrypt(plain_x, cipher_x);

    cout << endl << "Original plaintext (first 5 values):" << endl;
    for (int i = 0; i < 5; i++) cout << m_x[i] << " ";
    cout << endl;

    cout << endl << "MinimaxReLU evaluation..." << endl;
    time_start = chrono::high_resolution_clock::now();
    minimax_ReLU_seal(comp_no, deg, alpha, tree, scaled_val, scalingfactor,
                      encryptor, evaluator, decryptor, encoder,
                      public_key, secret_key, relin_keys,
                      cipher_x, cipher_x);
    time_end = chrono::high_resolution_clock::now();
    time_diff = chrono::duration_cast<chrono::milliseconds>(time_end - time_start);

    cout << "Remaining level after evaluation: "
         << context.get_context_data(cipher_x.parms_id())->chain_index() << endl;
    cout << "ReLU function time: " << time_diff.count() / 1000 << " ms" << endl;

    Plaintext plain_res;
    Plaintext plain_res;
    decryptor.decrypt(cipher_x, plain_res);
    vector<double> res;
    encoder.decode(plain_res, res);

    // Print first 10 entries with comparison
    cout << "\nDecrypted ReLU result vs true ReLU (first 10 values):" << endl;
    cout << setw(12) << "x" << setw(20) << "Decrypted" << setw(20) << "True ReLU" << setw(20) << "Error" << endl;
    cout << string(72, '-') << endl;

    int print_limit = min(10, n);
    for (int i = 0; i < print_limit; i++) {
        double true_val = max(0.0, m_x[i]);
        double err = fabs(res[i] - true_val);
        cout << setw(12) << m_x[i] 
             << setw(20) << res[i] 
             << setw(20) << true_val 
             << setw(20) << err << endl;
    }

    // Compute max error over all points
    double max_err = 0.0;
    for (int i = 0; i < n; i++) {
        double true_val = max(0.0, m_x[i]);
        double err = fabs(res[i] - true_val);
        if (err > max_err) max_err = err;
    }
    cout << "\nMaximum error: " << max_err << " (should be < 2^{-" << alpha << "} = " << pow(2.0, -alpha) << ")" << endl;
}

// ----------------------------------------------------------------------
// 4. Main entry point: choose action based on command line arguments
// ----------------------------------------------------------------------
int main(int argc, char **argv) {
    if (argc >= 3) {
        string cmd = argv[1];
        string config = argv[2];

        if (cmd == "generate") {
            generate_coefficients(config);
            return 0;
        } else if (cmd == "test") {
            test_approximation(config);
            return 0;
        } else {
            // If first argument is not "generate" or "test", treat it as config name
            run_full_test(cmd);
            return 0;
        }
    } else if (argc == 2) {
        // Only config name provided (e.g. ./test deg13)
        run_full_test(argv[1]);
        return 0;
    } else {
        // No arguments -> default full test (deg13)
        run_full_test("deg13");
        return 0;
    }
}