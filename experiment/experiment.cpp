#include <bits/stdc++.h>
#include <sys/stat.h>
#include <chrono>
#include "data_loader.h"
#include "../hnswlib/hnswlib.h"
#include "calc_group_truth.h"
#include "recall_test.h"
#include "ArgParser.h"
#include "config.h"
#include "statis_tasks.h"
#include "dir_vector.h"
#include "k_means.h"

using namespace std;
using DATALOADER::DataLoader;

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        cout << "Usage: No max_elements?" << endl;
        return 1;
    }
    string data_dir = "/share/ann_benchmarks/gist/";
    int max_elements = atoi(argv[1]);
    int M = 16;
    int ef_construction = 200;
    std::mt19937 rng;
    rng.seed(47);
    int dim = 960;
    int K = 20;
    DataLoader *data_loader = new DataLoader("f", max_elements, data_dir + "train.fvecs", "gist");
    DataLoader *query_data_loader = new DataLoader("f", 1000, data_dir + "test.fvecs", "gist");
    hnswlib::L2Space space(dim);
    GroundTruth::calc_gt("../../gist_hnsw/gnd", data_loader, query_data_loader, space, K, 0);
    GroundTruth::GT_Loader *gt_loader = new GroundTruth::GT_Loader("../../gist_hnsw/gnd", data_loader, query_data_loader, K);
    hnswlib::HierarchicalNSW<float> *alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);
    string hnsw_path = "../../gist_hnsw/index/hnsw" + to_string(max_elements) + ".bin";
    ifstream index(hnsw_path.c_str());
    if (index.good())
    {
        cout << "Load index from " << hnsw_path << "\n";
        alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, hnsw_path);
    }
    else
    {
        for (int i = 0; i < data_loader->get_elements(); i++)
        {
            alg_hnsw->addPoint(data_loader->point_data(i), i);
            if (i > 0 && i % 10000 == 0)
                cout << "Insert " << i << " elements\n";
        }
        alg_hnsw->saveIndex(hnsw_path);
    }
    ofstream qps("../../gist_hnsw/qps/hnsw" + to_string(max_elements) + ".txt");
    ofstream rec("../../gist_hnsw/recall/hnsw" + to_string(max_elements) + ".txt");
    ofstream res("../../gist_hnsw/result/hnsw" + to_string(max_elements) + ".txt");
    cout << "Start search\n";
    for (int ef = 60; ef <= 600; ef += 10)
    {
        alg_hnsw->setEf(ef);
        float duration = 0;
        float recall = 0;
        for (int i = 0; i < query_data_loader->get_elements(); i++)
        {
            auto start = std::chrono::high_resolution_clock::now();
            std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnn(query_data_loader->point_data(i), K);
            auto end = std::chrono::high_resolution_clock::now();
            duration += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            recall += gt_loader->calc_recall(result, i);
        }
        cout << "ef = " << ef << " qps = " << query_data_loader->get_elements() * 1000000 / duration << " recall = " << recall/1000 << "\n";
        qps << query_data_loader->get_elements() * 1000000 / duration << "\n";
        rec << recall / query_data_loader->get_elements() << "\n";
    }
    qps.close();
    rec.close();
    string command = "python3 ../experiment/graph.py " + to_string(max_elements);
    system(command.c_str());
    return 0;
}