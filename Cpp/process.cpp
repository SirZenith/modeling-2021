#include <bits/stdc++.h>
using namespace std;

// 供货商类
class supplier {
    public:
        string name;
        char type;
        array<double, 240> requests, supply, performance_rate;

        // 供货量均值、供货量周期、履约率均值、履约率方差、表现分数
        double supply_mean, supply_cycle, performance_rate_mean, performance_rate_variance, score;
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
        i.supply_mean = accumulate(i.supply.begin(), i.supply.end(), 0)/240;

        i.performance_rate_mean = accumulate(
            i.performance_rate.begin(), 
            i.performance_rate.end(), 
            0
        ) / 240;

        i.performance_rate_variance = accumulate(
            i.performance_rate.begin(), 
            i.performance_rate.end(), 
            0, 
            [&i](auto init, auto j) {
                return init + fabs(i.performance_rate_mean - j);
            }
        );

        i.supply_cycle = 0;
        double supply_cycle_tmp, supply_cycle_min = 0;
        for (auto j = 1; j < 120; ++j) {
            for (auto k = 0; k < 240 - j; ++k)
                supply_cycle_tmp += fabs(i.supply[k] - i.supply[k + j]);
            supply_cycle_tmp /= (240 - j);
            if (!supply_cycle_min || supply_cycle_min > supply_cycle_tmp) {
                i.supply_cycle = j;
                supply_cycle_min = supply_cycle_tmp;
            } 
        }
    }
}

int main() {
    readfile();
    getClassProperties();
    for (auto &i: supplierList) {
        cout << i.name << " ";
        printf("%c %lf %lf %lf %lf\n", 
            i.type,
            i.supply_mean, i.supply_cycle,
            i.performance_rate_mean, i.performance_rate_variance
        );
    }
    return 0;
}