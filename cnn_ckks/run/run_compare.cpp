// basic
#include <iostream>
#include <NTL/RR.h>
#include <cmath>
#include <map>
#include "PolyUpdate.h"
#include "program.h"
#include "seal/seal.h"
#include "SEALcomp.h"

using namespace seal;
using namespace std;
using namespace NTL;

struct ReLUConfig {
	long alpha;
	int comp_no;
	vector<int> deg;
	double scaled_val;
	int remaining_level; 
};

int main(int argc, char **argv) {

    string config = "deg13";
    if (argc >= 2) {
        config = argv[1];
    }

	map<string, ReLUConfig> relu_configs = {
		{"deg13", {13, 3, {15, 15, 27}, 1.7, 16}},
		{"deg9",  {9,  2, {15, 15},     1.7, 11}},
		// more options
		// 
	};

    auto it = relu_configs.find(config);
    if (it == relu_configs.end()) {
        cerr << "Unknown configuration: " << config << endl;
        cerr << "Available: ";
        for (auto &c : relu_configs) cerr << c.first << " ";
        cerr << endl;
        return 1;
    }

    long alpha = it->second.alpha;
    int comp_no = it->second.comp_no;
    vector<int> deg = it->second.deg;
    double scaled_val = it->second.scaled_val;

    cout << "Using configuration: " << config << endl;
    cout << "alpha = " << alpha << ", comp_no = " << comp_no << ", deg = ";
    for (int d : deg) cout << d << " ";
    cout << ", scaled_val = " << scaled_val << endl;

    int remaining_level = it->second.remaining_level;

    int boot_level = 0; 
    int total_level = remaining_level + boot_level;

    long scalingfactor = 45;   // log2 of scaling factor
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
    decryptor.decrypt(cipher_x, plain_res);
    vector<double> res;
    encoder.decode(plain_res, res);
    cout << "Decrypted ReLU result (first 5 values):" << endl;
    for (int i = 0; i < 5; i++) cout << res[i] << " ";
    cout << endl;

    // error
    double max_err = 0.0;
    for (int i = 0; i < n; i++) {
        double true_val = max(0.0, m_x[i]);
        double err = fabs(res[i] - true_val);
        if (err > max_err) max_err = err;
    }
    cout << "Maximum error: " << max_err << " (should be < 2^{-" << alpha << "} = " << pow(2.0, -alpha) << ")" << endl;

    return 0;
}


// // basic
// #include <iostream>
// #include <NTL/RR.h>
// #include <cmath>
// #include "PolyUpdate.h"
// #include "program.h"
// #include "seal/seal.h"
// #include "SEALcomp.h"

// using namespace seal;
// using namespace std;
// using namespace NTL;


// int main() {

// 	// SEAL User setting
// 	long level = 14;		// number of levels L. L = sum ceil(log2(di)) for comparison operation and L = sum ceil(log2(di))+1 for max & ReLU function. di is degree of component polynomials.
// 	long alpha = 13;			// precision parameter alpha
// 	long comp_no = 3;		// number of compositions
// 	long scalingfactor = 45;		// log2 of scaling factor
// 	vector<int> deg = {15,15,27};		// degrees of component polynomials
// 	double eta = pow(2.0,-15);		// margin
// 	double scaled_val = 1.7;		// scaled_val: the last scaled value
// 	double max_factor = 16;		// max_factor = 1 for comparison operation. max_factor > 1 for max or ReLU function
// 	vector<Tree> tree;		// structure of polynomial evaluation
// 	evaltype eval_type = evaltype::oddbaby;
// 	RR::SetOutputPrecision(25);

// 	// generate tree
// 	for(int i=0; i<comp_no; i++) 
// 	{
// 		Tree tr;
// 		if(eval_type == evaltype::oddbaby) upgrade_oddbaby(deg[i], tr);
// 		else if(eval_type == evaltype::baby) upgrade_baby(deg[i], tr);
// 		else std::invalid_argument("evaluation type is not correct");
// 		tree.emplace_back(tr);
// 		tr.print();
// 	}

// 	// SEAL setting
// 	long log_modulus = scalingfactor;
// 	EncryptionParameters parms(scheme_type::ckks);	
// 	size_t poly_modulus_degree = 65536;
// 	long n = poly_modulus_degree / 2;
// 	parms.set_poly_modulus_degree(poly_modulus_degree);
// 	vector<int> modulus_list;
// 	modulus_list.emplace_back(60); 	for(int i=0; i<level; i++) modulus_list.emplace_back(log_modulus);		modulus_list.emplace_back(60);
// 	parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, modulus_list));
// 	SEALContext context(parms);
// 	chrono::high_resolution_clock::time_point time_start, time_end;
// 	chrono::microseconds time_diff;
	
// 	// key generate
// 	KeyGenerator keygen(context);
// 	PublicKey public_key;
// 	keygen.create_public_key(public_key);
// 	auto secret_key = keygen.secret_key();
// 	RelinKeys relin_keys;
// 	keygen.create_relin_keys(relin_keys);
// 	GaloisKeys gal_keys;		// this is for real ReLU
// 	keygen.create_galois_keys(gal_keys);

// 	// encryptor, evaluator, decryptor, encoder, scale_evaluator
// 	Encryptor encryptor(context, public_key);
// 	CKKSEncoder encoder(context);
// 	Evaluator evaluator(context, encoder);
// 	Decryptor decryptor(context, secret_key);
// 	// ScaleInvEvaluator scale_evaluator(context, encoder, relin_keys);

// 	// generate two input vectors to compare
// 	vector<double> m_x(n);
// 	vector<double> output;
// 	for(int i=0; i<n; i++) m_x[i] = -1.0 + 2.0 * static_cast<double>(i) / static_cast<double>(n-1);

// 	// encode
// 	Plaintext plain_x;
// 	double scale = pow(2.0, log_modulus);
// 	encoder.encode(m_x, scale, plain_x);

// 	// encrypt
// 	Ciphertext cipher_x;
// 	encryptor.encrypt(plain_x, cipher_x);

// 	// MinimaxReLU
// 	cout << endl << "Decrypt" << endl;
// 	// print_cipher(decryptor, encoder, public_key, secret_key, relin_keys, cipher_x);
// 	decrypt_and_print_part(cipher_x, decryptor, encoder, n, 0, 5);
// 	time_start = chrono::high_resolution_clock::now();
// 		cout << endl << "MinimaxReLU" << endl;
// 	minimax_ReLU_seal(comp_no, deg, alpha, tree, scaled_val, scalingfactor, encryptor, evaluator, decryptor, encoder, public_key, secret_key, relin_keys, cipher_x, cipher_x);
// 			cout << endl << "MinimaxReLU end" << endl;
// 	time_end = chrono::high_resolution_clock::now();
// 	time_diff = chrono::duration_cast<chrono::milliseconds>(time_end - time_start);

// 	// result
// 	cout << "remaining level : " << context.get_context_data(cipher_x.parms_id())->chain_index() << endl;
// 	// print_cipher(decryptor, encoder, public_key, secret_key, relin_keys, cipher_x);
// 	decrypt_and_print_part(cipher_x, decryptor, encoder, n, 0, 5);
// 	ShowFailure_ReLU(decryptor, encoder, cipher_x, m_x, alpha, n);
// 	cout << "ReLU function time : " << time_diff.count() / 1000 << " ms" << endl;

// 	return 0;
// }
