#include <bits/stdc++.h>
using namespace std;

// 供货商类
class supplier {
    public:
        string name;
        char type;
        array<double, 240> requests, supply, performance_rate;

        // 供货量均值、突发订单次数、履约率均值、履约率方差、表现分数、有订单的天数
        double supply_mean, big_supply, performance_rate_mean, performance_rate_variance, score, active_days;
};

// 供货商列表
array<supplier, 402> supplierList;

void readfile() {
    ifstream fin("cleanned-data/requests.mdb", ios::in);
    for (auto i = 0; i < 402; ++i) {
        fin >> supplierList[i].name;
        fin >> supplierList[i].type;
        for (auto j = 0; j < 240; ++j) 
            fin >> supplierList[i].requests[j];
    }
    fin.close();
    fin.open("cleanned-data/supply.mdb", ios::in);
    for (auto i = 0; i < 402; ++i) {
        fin >> supplierList[i].name;
        fin >> supplierList[i].type;
        for (auto j = 0; j < 240; ++j) {
            fin >> supplierList[i].supply[j];
            supplierList[i].performance_rate[j] = (supplierList[i].requests[j]) ? 
            supplierList[i].supply[j] / supplierList[i].requests[j] : 0;
        }
    }
    fin.close();
}

// 计算供货商的各项表现参数
void getClassProperties() {
    for (auto &i: supplierList) {
        i.supply_mean = accumulate(i.supply.begin(), i.supply.end(), 0.0)/240;
        
        i.active_days = count_if(i.requests.begin(), i.requests.end(), [](auto &j){
            return j > 0;
        });

        i.performance_rate_mean = accumulate(
            i.performance_rate.begin(), 
            i.performance_rate.end(), 
            0.0
        ) / i.active_days;

        i.performance_rate_variance = accumulate(
            i.performance_rate.begin(), 
            i.performance_rate.end(), 
            0.0, 
            [&i](auto init, auto j) {
                return !j ? init : init + (i.performance_rate_mean - j) * (i.performance_rate_mean - j);
            }
        ) / i.active_days;

        i.big_supply = count_if(i.supply.begin(), i.supply.end(), [&i](auto& j) {
            return j > i.supply_mean;
        });

        i.score = pow(i.supply_mean, 2) * exp(i.performance_rate_mean) * (1 - i.performance_rate_variance) * i.active_days / 240;
    }
}

int main() {
    readfile();
    getClassProperties();
    sort(supplierList.begin(), supplierList.end(), [](auto &i, auto &j){
        return i.score > j.score;
    });
    for (auto &i: supplierList) {
        cout << "|" << i.name << "|" << i.active_days << "|";
        printf("%c|%lf|%lf|%lf|%lf|\n", 
            i.type,
            i.supply_mean, 
            i.performance_rate_mean, i.performance_rate_variance,
            i.score
        );
    }

    // for (auto &i: supplierList[228].performance_rate)
    //     cout << i << ", ";
    return 0;
}