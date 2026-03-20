#include <iostream>
#include "cnn_seal.h"
#include "infer_seal.h"
#include <algorithm>


#include <unistd.h>
#include <climits> // For PAGESIZE
using namespace std;


void printSysconfMemoryInfo() {
    long pagesize = sysconf(_SC_PAGESIZE);
    long page_count = sysconf(_SC_PHYS_PAGES);
    long avail_page_count = sysconf(_SC_AVPHYS_PAGES);

    if (pagesize == -1 || page_count == -1 || avail_page_count == -1) {
        std::cerr << "无法获取内存信息\n";
        return;
    }

    std::cout << "页面大小: " << pagesize << " bytes\n";
    std::cout << "总物理内存: " << (page_count * pagesize) / (1024 * 1024) << " MB\n";
    std::cout << "可用物理内存: " << (avail_page_count * pagesize) / (1024 * 1024) << " MB\n";
}
int main(int argc, char **argv) {
	printSysconfMemoryInfo();
	if (argc < 5) {
        cerr << "Usage: " << argv[0] << " layer dataset start end [precison_par] [model_path]" << endl;
        return 1;
    }
	int layer = atoi(argv[1]);
	int dataset = atoi(argv[2]);
	int start = atoi(argv[3]);
	int end = atoi(argv[4]);
	string precison = (argc >= 6) ? argv[5] : "deg13";
    string model_path = (argc >= 7) ? argv[6] : "../../pretrained_parameters/mpcnn_models/";
	int threads = (argc >= 8) ? atoi(argv[7]) : 8; 

	if (start < 0 || start >= 10000) throw std::invalid_argument("start number is not correct");
	if (end < 0 || end >= 10000) throw std::invalid_argument("end number is not correct");
	if (start > end) throw std::invalid_argument("start number is larger than end number");

	cout << "model: ResNet-" << layer << endl;
	cout << "dataset: CIFAR-" << dataset << endl;
	cout << "start image: " << start << endl;
	cout << "end image: " << end << endl;
	cout << "precison_par: " << precison << endl;
    cout << "model path: " << model_path << endl;
	cout << "threads: " << threads << endl;

	if (dataset == 10) ResNet_cifar10_seal_sparse(layer, start, end, precison, model_path, threads);
	// else if (dataset == 100) ResNet_cifar100_seal_sparse(layer, start, end);
	// else throw std::invalid_argument("dataset number is not correct");

	return 0;
}
