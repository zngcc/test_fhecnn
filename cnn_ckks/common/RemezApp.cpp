#include "RemezApp.h"

// d is odd!!
void testReLU(size_t deg)
{
	RR maxerr;
//	long prec = 300;
	long prec = 50;
	long iteration = 200;
	long RRprec;
	int num;
	RR sc;
//	long d = 300;
	long type = 1;
	RR scale = RR(1);

	size_t inter_num = 1;
	vector<RR> inter_start, inter_end;
	vector<RR> coeff, chebcoeff;
	inter_start.emplace_back(RR(-1));
	inter_end.emplace_back(RR(1));

	num = 1000;
//	num = 2000;
	RRprec = 300;
//	RRprec = 2000;
	RR::SetPrecision(RRprec);
	sc = RR(2.0)/RR(num);

	Remez rm(ReLU, inter_num, inter_start, inter_end, sc, prec, deg, iteration, type, scale, true);
	rm.test();
	maxerr = rm.getmaxerr();
	cout << "maxerr: " << maxerr << endl;

	rm.getcoeff(chebcoeff);
	Polynomial poly(deg, chebcoeff, "cheb");
	poly.cheb_to_power();
	poly.get_coeff(coeff);
//	cout << "coefficients: " << endl;
	ofstream out1("../result/ReLUpower.txt"), out2("../result/ReLUcheb.txt");
	for(auto num : coeff) out1 << num << endl;
	for(auto num : chebcoeff) out2 << num << endl;
	cout << endl;

	// test for coefficients
	RR max = RR(0);
	for(RR x = RR(-1); x<RR(1); x+=RR(0.0001)) if(abs(poly.evaluate(x) - ReLU(x)) > max) max = abs(poly.evaluate(x) - ReLU(x));
	cout << "power basis approx maxerr: " << max << endl;
	max = RR(0);
	for(RR x = RR(-1); x<RR(1); x+=RR(0.0001)) if(abs(poly.evaluate_cheb(x) - ReLU(x)) > max) max = abs(poly.evaluate_cheb(x) - ReLU(x));
	cout << "cheb basis approx maxerr: " << max << endl;
	

}
RR GetError(int d, RR t, bool is_first_function, int type, RR scale)
{
	Remez *rm;
	RR maxerr;
	long prec = 40;
	long iteration = 20;
	long RRprec;
	int num;
	RR sc;

	size_t inter_num = 2;
	vector<RR> inter_start, inter_end;
	if(is_first_function == true)
	{
		inter_start.emplace_back(RR(-1));
		inter_start.emplace_back(RR(1)-t);
		inter_end.emplace_back(RR(-1)+t);
		inter_end.emplace_back(RR(1));
	}
	else
	{
		inter_start.emplace_back(RR(-1)-t);
		inter_start.emplace_back(RR(1)-t);
		inter_end.emplace_back(RR(-1)+t);
		inter_end.emplace_back(RR(1)+t);
	}

	//  num & RRprec setting!!!
	num = 500;
	RRprec = 1500;		// for deg 64. RRprec >= 1500 approximately. But, if tau > 1/2, RRprec = 300 is ok.

	RR::SetPrecision(RRprec);
	sc = t/static_cast<RR>(num);
	//	rm = new Remez(t, is_first_function, sc, prec, d+1, iteration, type, scale);
	rm = new Remez(sgn, inter_num, inter_start, inter_end, sc, prec, d+1, iteration, type, scale, true);
	rm->test();
	maxerr = rm -> getmaxerr();
	
	delete rm;
	return maxerr;
}
/*
void GetCoeff(int d, RR t, RR coeff[], bool is_first_function, int type, RR scale)
{
	Remez *rm;
	RR maxerr;
	long prec = 40;
	long iteration = 20;
	long RRprec;
	int num;
	RR sc;

	//  num & RRprec setting!!!
	num = 2000;
	RRprec = 800;

	RR::SetPrecision(RRprec);
	sc = t/static_cast<RR>(num);
	rm = new Remez(t, is_first_function, sc, prec, d+1, iteration, type, scale);
	rm->test();
	rm->getcoeff(coeff);

	delete rm;
}
*/
RR GetErrorCoeff(int d, RR t, vector<RR> &coeff, bool is_first_function, int type, RR scale)
{
	Remez *rm;
	RR maxerr;
	long prec = 40;
	long iteration = 25;
	long RRprec;
	int num;;
	RR sc;

	size_t inter_num = 2;
	vector<RR> inter_start, inter_end;
	if(is_first_function == true)
	{
		inter_start.emplace_back(RR(-1));
		inter_start.emplace_back(RR(1)-t);
		inter_end.emplace_back(RR(-1)+t);
		inter_end.emplace_back(RR(1));
	}
	else
	{
		inter_start.emplace_back(RR(-1)-t);
		inter_start.emplace_back(RR(1)-t);
		inter_end.emplace_back(RR(-1)+t);
		inter_end.emplace_back(RR(1)+t);
	}

	//  num & RRprec setting!!!
	num = 1000;
	RRprec = 300;

	RR::SetPrecision(RRprec);
	sc = t/static_cast<RR>(num);
//	rm = new Remez(t, is_first_function, sc, prec, d+1, iteration, type, scale);
	rm = new Remez(sgn, inter_num, inter_start, inter_end, sc, prec, d+1, iteration, type, scale, true);
	rm->test(); 
	rm->getcoeff(coeff);
	maxerr = rm->getmaxerr();

	delete rm;
	return maxerr;
}
/*
RR GetErrorCoeff_extpoint(int d, RR t, vector<RR> &coeff, vector<RR> &ext_xpos, bool is_first_function, int type, RR scale)
{
	Remez *rm;
	RR maxerr;
	long prec = 40;
	long iteration = 20;
	long RRprec;
	int num;;
	RR sc;

	//  num & RRprec setting!!!
	num = 1000;
	RRprec = 300;

	RR::SetPrecision(RRprec);
	sc = t/static_cast<RR>(num);
	rm = new Remez(t, is_first_function, sc, prec, d+1, iteration, type, scale);
	rm->test();
	rm->getcoeff(coeff);
	rm->getext_xpos(ext_xpos);
	maxerr = rm->getmaxerr();

	delete rm;
	return maxerr;
}
*/

////////////////////////////////

long coeff_number(long deg, Tree& tree) 
{
    long num = 0;
    long* decomp_deg = new long[pow2(tree.depth+1)];
    decomp_deg[1] = deg;
    for(int i=1; i<=tree.depth; i++)
    {
        for(int j=pow2(i); j<pow2(i+1); j++)
        {
            if(j%2 == 0) decomp_deg[j] = tree.tree[j/2] - 1;
            else if(j%2 == 1) decomp_deg[j] = decomp_deg[j/2] - tree.tree[j/2];
        }
    }

    for(int i=0; i<pow2(tree.depth+1); i++)
    {
        if(tree.tree[i] == 0) 
        {
            num += (decomp_deg[i]+1);
        }
    }
    delete decomp_deg;
    return num;
}

void generate_minimax_coefficients(size_t deg, const string& output_file) {
    RR maxerr;
    long prec = 50;
    long iteration = 200;
    long RRprec = 300;
    int num = 1000;
    long type = 1;
    RR scale = RR(1);
    
    size_t inter_num = 1;
    vector<RR> inter_start, inter_end;
    inter_start.emplace_back(RR(-1));
    inter_end.emplace_back(RR(1));
    
    RR::SetPrecision(RRprec);
    RR sc = RR(2.0) / RR(num);
    
    cout << "Generating minimax polynomial for ReLU, degree " << deg << "..." << endl;
    
    Remez rm(ReLU, inter_num, inter_start, inter_end, sc, prec, deg, iteration, type, scale, true);
    rm.test();
    maxerr = rm.getmaxerr();
    cout << "maxerr: " << maxerr << endl;
    
    vector<RR> chebcoeff, coeff;
    rm.getcoeff(chebcoeff);
    
    Polynomial poly(deg, chebcoeff, "cheb");
    poly.cheb_to_power();
    poly.get_coeff(coeff);
    
    ofstream out(output_file);
    for(auto num : coeff) out << num << endl;
    out.close();
    
    cout << "Coefficients saved to " << output_file << endl;
    cout << "Number of coefficients: " << coeff.size() << endl;
}

long count_total_coefficients(const vector<int>& degs, evaltype ev_type) {
    long total = 0;
    for(size_t i = 0; i < degs.size(); i++) {
        Tree tree;
        if(ev_type == evaltype::oddbaby) {
            upgrade_oddbaby(degs[i], tree);
        }
        total += coeff_number(degs[i], tree);
    }
    return total;
}

void print_tree_info(const vector<int>& degs, evaltype ev_type) {
    cout << "\n========== Polynomial Composition Info ==========" << endl;
    long total_coeff = 0;
    for(size_t i = 0; i < degs.size(); i++) {
        Tree tree;
        if(ev_type == evaltype::oddbaby) {
            upgrade_oddbaby(degs[i], tree);
        }
        long n = coeff_number(degs[i], tree);
        total_coeff += n;
        cout << "Component " << i << ": degree=" << degs[i] << ", tree structure:" << endl;
        tree.print();
        cout << "  -> Coefficients needed: " << n << endl;
        cout << endl;
    }
    cout << "Total coefficients: " << total_coeff << endl;
    cout << "===============================================" << endl;
}
void generate_single_deg(const DegConfig& config) {
    evaltype ev_type = evaltype::oddbaby;
    
    cout << "\n========== Generating " << config.name << " Coefficients ==========" << endl;
    cout << "Configuration: degs = {";
    for(size_t i = 0; i < config.degs.size(); i++) {
        cout << config.degs[i];
        if(i < config.degs.size() - 1) cout << ", ";
    }
    cout << "}" << endl;
    
    print_tree_info(config.degs, ev_type);
    
    long prec = 50;
    long iteration = 200;
    long RRprec = 300;
    int num = 1000;
    long type = 1;
    RR scale = RR(1);
    
    size_t inter_num = 1;
    vector<RR> inter_start, inter_end;
    inter_start.emplace_back(RR(-1));
    inter_end.emplace_back(RR(1));
    
    RR::SetPrecision(RRprec);
    RR sc = RR(2.0) / RR(num);
    
    string output_file = "../approx_para/" + config.name + "_new.txt";
    ofstream out(output_file);
    
    for(size_t i = 0; i < config.degs.size(); i++) {
        cout << "\nProcessing component " << i << " (degree " << config.degs[i] << ")..." << endl;
        
        Tree tree;
        upgrade_oddbaby(config.degs[i], tree);
        
        Remez rm(ReLU, inter_num, inter_start, inter_end, sc, prec, config.degs[i], iteration, type, scale, true);
        rm.test();
        RR maxerr = rm.getmaxerr();
        cout << "  maxerr: " << maxerr << endl;
        
        vector<RR> chebcoeff, coeff;
        rm.getcoeff(chebcoeff);
        
        Polynomial poly(config.degs[i], chebcoeff, "cheb");
        poly.cheb_to_power();
        poly.get_coeff(coeff);
        
        cout << "  Writing " << coeff.size() << " coefficients..." << endl;
        for(auto num : coeff) {
            out << num << endl;
        }
    }
    
    out.close();
    cout << "\n==================================================" << endl;
    cout << "Output saved to: " << output_file << endl;
}

void generate_all_degs() {
    vector<DegConfig> configs = {
        {"d8",  {7, 15}},
        {"d9",  {15, 15}},
        {"d10", {7, 7, 13}},
        {"d11", {7, 7, 27}},
        {"d12", {7, 15, 27}},
        {"d13", {15, 15, 27}}
    };
    
    cout << "############################################" << endl;
    cout << "#  Generating all minimax coefficients     #" << endl;
    cout << "#  for deg8, deg9, deg10, deg11, deg12, deg13 #" << endl;
    cout << "############################################" << endl;
    
    for(size_t i = 0; i < configs.size(); i++) {
        generate_single_deg(configs[i]);
    }
    
    cout << "\n*** All generations completed! ***" << endl;
    cout << "Output files:" << endl;
    for(size_t i = 0; i < configs.size(); i++) {
        cout << "  ../approx_para/" << configs[i].name << "_new.txt" << endl;
    }
}