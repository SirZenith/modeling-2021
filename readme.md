# 大纲

## TODO

- [ ] 周期性分析之中对于数据的计算可能存在问题，因为对于数据缺乏明显周期性的数据计算出的周期值是相对较小的数值。观察当前方法在更多数据集上的检测结果。

## 小笔记

### init
此处为一些细碎观察点记录。

1. 根据订货量和是供应量差距判断潜在供货能力。
2. 根据向某个供应商发订单的频率识别长期稳定合作伙伴。

### 9-9-20:51 -Czile
1. 是否存在延时供货现象
2. 是否有接受大量订单的能力
3. 每个星期的实际供货量是否达到订单量
4. 根据历史订货量、订货周期，判定供货能力
5. 考虑运输损耗率和产品类型

## 题目解决思路
### 1. 判断企业的供货能力。  
供货能力分为供货量、履约率、供货稳定性。  
**供货量**：该供货公司是否能接大订单。体现在供货量的**均值**。  
**履约率**：这家供货公司的**实际供货量/订购量**  
**供货稳定性**：如果一家公司供货量很高，但是只合作过一次，我们认为这家公司也是多少有点问题的，不然为什么不长期合作呢？该能力体现在供货量的**周期**  
因此本题中需要的数据是：**供货量**的**均值**和**周期**，以及**履约率**的**均值**和**方差**，设定**分数函数**，函数以上述四个参数为自变量，并对分数函数进行排序。从而选出最佳的50家供应商。

对于供货量而言，如果在 5 年的区间上进行**均值**的计算，可以体现长期以来某个供应商为制造厂提供资源的数量，主要可以体现供应商的产能；而如果需要分析厂商的及时供货能力，在计算均值时应当限制在有订单的周之间。（主要在履约率计算方式上体现）

周期也许并不是用来判断长期稳定合作伙伴的好方法，可以考虑使用供应商有订单的时间在 240 周中的占比替代周期性指标。以供应商 S229 和 S140 对比为例，周期性数据几乎看不出内容。S140 提供了非常大量的资源，同时履约率也很高，但是周期值高达 117；S229 周期短，但是供货质量上可能一定程度上低于 S140。


**更新**：供货商的质量与下述参数相关（按重要程度）：
1. 供货量均值
2. 履约率的均值和方差（0-1）
3. 活跃周数（缺名词解释）
4. 大订单（名词解释）的履约率均值和方差

分数函数：
$$ = 供货量均值^2 * 履约率均值 * （2-履约率方差） * 活跃周数 $$