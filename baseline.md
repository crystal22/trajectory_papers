Welcome to the occuflow wiki!
# 轨迹预测数据集
行人轨迹预测的数据集主要有三个:
[ETH](http://www.vision.ee.ethz.ch/en/datasets/)
[UCY](https://graphics.cs.ucy.ac.cy/research/downloads/crowd-data)
[SSD](http://cvgl.stanford.edu/projects/uav_data/)
其中SSD 是最近出的比较大的库。

# 轨迹预测度量方法：ADE, NDE, FDE.
ADE: Average displacement error - The mean square error(MSE) over all estimated points of a trajectory and the true points.

<a href="https://www.codecogs.com/eqnedit.php?latex=ADE&space;=&space;\frac{\sum_{i=1}^N&space;\sum_{t=1}^T&space;(x^p_{i,t}-x^o_{i,t})^2}{N&space;\times&space;T}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?ADE&space;=&space;\frac{\sum_{i=1}^N&space;\sum_{t=1}^T&space;(x^p_{i,t}-x^o_{i,t})^2}{N&space;\times&space;T}" title="ADE = \frac{\sum_{i=1}^N \sum_{t=1}^T (x^p_{i,t}-x^o_{i,t})^2}{N \times T}" /></a>

FDE: Final displacement error - The distance between the predicted final destination and the true final destination at end of the prediction period T.

<a href="https://www.codecogs.com/eqnedit.php?latex=FDE&space;=&space;\frac{\sum_{i=1}^{N}\sqrt{(x^p_{i,t}-x^o_{i,t})^2}&space;}{N}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?FDE&space;=&space;\frac{\sum_{i=1}^{N}\sqrt{(x^p_{i,t}-x^o_{i,t})^2}&space;}{N}" title="FDE = \frac{\sum_{i=1}^{N}\sqrt{(x^p_{i,t}-x^o_{i,t})^2} }{N}" /></a>

NDE: Average non-linear displacement error - The NDE is the MSE at the non-linear regions of a trajectory, where we set a heuristic threshold on the norm of the second derivative to identify non-linear regions.

<a href="https://www.codecogs.com/eqnedit.php?latex=NDE&space;=&space;\frac{\sum_{i=1}^N&space;\sum_{t=1}^T&space;I_{non}(x_{i,t}^p)(x^p_{i,t}-x^o_{i,t})^2&space;}{\sum_{i=1}^N&space;\sum_{t=1}^T&space;I_{non}(x_{i,t}^p)&space;}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?NDE&space;=&space;\frac{\sum_{i=1}^N&space;\sum_{t=1}^T&space;I_{non}(x_{i,t}^p)(x^p_{i,t}-x^o_{i,t})^2&space;}{\sum_{i=1}^N&space;\sum_{t=1}^T&space;I_{non}(x_{i,t}^p)&space;}" title="NDE = \frac{\sum_{i=1}^N \sum_{t=1}^T I_{non}(x_{i,t}^p)(x^p_{i,t}-x^o_{i,t})^2 }{\sum_{i=1}^N \sum_{t=1}^T I_{non}(x_{i,t}^p) }" /></a>

不同论文中对非线性区域的定义似乎不太一致，需要进一步check:

[SH Attention](https://arxiv.org/pdf/1702.05552.pdf):定义2阶倒数不为零则是非线性区域

<a href="https://www.codecogs.com/eqnedit.php?latex=I_{non}(x_{i,t}^p)=\left\{\begin{matrix}&space;1&space;&&space;\frac{d^2y_{i,t}}{dx^2_{i,t}}\neq&space;0&space;\\&space;0&space;&&space;o.w&space;\end{matrix}\right." target="_blank"><img src="https://latex.codecogs.com/gif.latex?I_{non}(x_{i,t}^p)=\left\{\begin{matrix}&space;1&space;&&space;\frac{d^2y_{i,t}}{dx^2_{i,t}}\neq&space;0&space;\\&space;0&space;&&space;o.w&space;\end{matrix}\right." title="I_{non}(x_{i,t}^p)=\left\{\begin{matrix} 1 & \frac{d^2y_{i,t}}{dx^2_{i,t}}\neq 0 \\ 0 & o.w \end{matrix}\right." /></a>

[Sence LSTM](https://arxiv.org/ftp/arxiv/papers/1808/1808.04018.pdf):利用起点，中点，和终点定义了一个非线性程度，并认为非线性程度大于0.2 的为非线性区域

<a href="https://www.codecogs.com/eqnedit.php?latex=I_{non}(x_{i,t}^p)=\left\{\begin{matrix}&space;1&space;&&space;|y_{i,0}/2&plus;y_{i,T}/2-y_{i,m}|>0.2&space;\\&space;0&space;&&space;o.w&space;\end{matrix}\right." target="_blank"><img src="https://latex.codecogs.com/gif.latex?I_{non}(x_{i,t}^p)=\left\{\begin{matrix}&space;1&space;&&space;|y_{i,0}/2&plus;y_{i,T}/2-y_{i,m}|>0.2&space;\\&space;0&space;&&space;o.w&space;\end{matrix}\right." title="I_{non}(x_{i,t}^p)=\left\{\begin{matrix} 1 & |y_{i,0}/2+y_{i,T}/2-y_{i,m}|>0.2 \\ 0 & o.w \end{matrix}\right." /></a>


# 轨迹预测已有baselines
简短的总结一下已有的一些baselines：

第一种是基于LSTM的，包括：
[Social LSTM](http://openaccess.thecvf.com/content_cvpr_2016/papers/Alahi_Social_LSTM_Human_CVPR_2016_paper.pdf)
[Bi-LSTM](https://www.researchgate.net/profile/Du_Huynh/publication/322001876_Bi-Prediction_Pedestrian_Trajectory_Prediction_Based_on_Bidirectional_LSTM_Classification/links/5c03cef4a6fdcc1b8d5029bb/Bi-Prediction-Pedestrian-Trajectory-Prediction-Based-on-Bidirectional-LSTM-Classification.pdf)
[Sence LSTM](https://arxiv.org/ftp/arxiv/papers/1808/1808.04018.pdf)
[Social-Sence LSTM](https://www.researchgate.net/profile/Du_Huynh/publication/2269555_Self-Calibrating_a_Stereo_Head_An_Error_Analysis_in_the_Neighbourhood_of_Degenerate_Configurations/links/5c03ccb0a6fdcc1b8d502965/Self-Calibrating-a-Stereo-Head-An-Error-Analysis-in-the-Neighbourhood-of-Degenerate-Configurations.pdf)
[St-LSTM](https://arxiv.org/pdf/1807.08381.pdf)
[CIDNN](http://openaccess.thecvf.com/content_cvpr_2018/papers/Xu_Encoding_Crowd_Interaction_CVPR_2018_paper.pdf)
[SR-LSTM](https://arxiv.org/pdf/1903.02793.pdf)

第二种是基于Attention的，包括：
[Social Attention](https://arxiv.org/pdf/1710.04689.pdf)
[SH Attention](https://arxiv.org/pdf/1702.05552.pdf)

第三种是基于GAN的，包括：
[Social GAN](http://openaccess.thecvf.com/content_cvpr_2018/papers/Gupta_Social_GAN_Socially_CVPR_2018_paper.pdf)
[Sophie](https://arxiv.org/pdf/1806.01482.pdf)

**SR-LSTM** 应该是当前的State-of-art，是今年cvpr19的paper.


ETH-hotel

| Metrics | Social LSTM | Bi-LSTM |  Sence LSTM-a | SS LSTM-c | St-LSTM | CIDNN | SR-LSTM(Variant id=7) | Social Attention | SH Attention | Social GAN(20V-20) | Sophie(T_A+I_A) |
|------------------|-------------|---------|--------------|--------------|------------|-------|---------|------------------|--------------|------------|--------|
| ADE | 0.11 | null | 0.06 | 0.066 | null | 0.11 | 0.37 | 0.29 | null | 0.48 | 0.76 |  
| NDE | 0.07 | null | 0.07 | null | null | null | null | null | null | null | null |  
| FDE | 0.23 | null | 0.06 | 0.101 | null | null | 0.74 | 2.64 | null | 0.95 | 1.67 |  

ETH-univ

| Metrics | Social LSTM | Bi-LSTM |  Sence LSTM-n | SS LSTM-l | St-LSTM | CIDNN | SR-LSTM(Variant id=7) | Social Attention | SH Attention | Social GAN(20V-20) | Sophie(T_A+I_A) |
|------------------|-------------|---------|--------------|-----------|----------|-------|---------|------------------|--------------|------------|--------|
| ADE | 0.50 | null | 0.10 | 0.095 | null | 0.09 | 0.63 | 0.39 | null | 0.61 | 0.70 | 
| NDE | 0.25 | null | 0.13 | null | null | null | null | null | null | null | null |  
| FDE | 1.07 | null | 0.18 | 0.235 | null | null | 1.25 | 3.74 | null | 1.22 | 1.43 | 

* UCY-univ

| Metrics | Social LSTM | Bi-LSTM | Sence LSTM | SS LSTM | St-LSTM | CIDNN | SR-LSTM | Social Attention | SH Attention | Social GAN | Sophie |
|------------------|-------------|---------|--------------|-----------|---------|-------|---------|------------------|--------------|------------|--------|
| ADE | 0.27 | - | 0.09 | 0.081 | - | 0.12 | 0.51 | 0.33 | - | 0.36 | 0.49 |
| NDE | 0.16 | - | 0.10 | - | - | - | - | - | - | - | - |
| FDE | 0.77 | - | 0.02 | 0.131 | - | - | 1.10 | 3.92 | - | 1.26 | 1.19 |

* UCY-Zara1

| Metrics | Social LSTM | Bi-LSTM | Sence LSTM | SS LSTM | St-LSTM | CIDNN | SR-LSTM | Social Attention | SH Attention | Social GAN | Sophie |
|-------------------|-------------|---------|--------------|-----------|---------|-------|---------|------------------|--------------|------------|--------|
| ADE | 0.22 | - | 0.07 | 0.050 | - | 0.15 | 0.41 | 0.20 | - | 0.21 | 0.30 |
| NDE | 0.13 | - | 0.09 | - | - | - | - | - | - | - | - |
| FDE | 0.48 | - | 0.07 | 0.081 | - | - | 0.90 | 0.52 | - | 0.42 | 0.63 |

* UCY-Zara2

| Metrics | Social LSTM | Bi-LSTM | Sence LSTM | SS LSTM | St-LSTM | CIDNN | SR-LSTM | Social Attention | SH Attention | Social GAN | Sophie |
|-------------------|-------------|---------|--------------|-----------|---------|-------|---------|------------------|--------------|------------|--------|
| ADE | 0.25 | - | 0.05 | 0.054 | - | 0.10 | 0.32 | 0.30 | - | 0.27 | 0.38 |
| NDE | 0.16 | - | 0.06 | - | - | - | - | - | - | - | - |
| FDE | 0.50 | - | 0.02 | 0.091 | - | - | 0.70 | 2.13 | - | 0.54 | 0.78 |

-----
*Baseline源码

| Baseline | Lib. | Dataset | Metrics | Comment |
|-------------------|-------------|---------|--------------|-----------------|
| [Social LSTM](https://github.com/SZamboni/Social_lstm_pedestrian_prediction) | Tensorflow 1.5 | ETH,UCY | FDE,ADE,NDE | - |
| [Social GAN](https://github.com/agrimgupta92/sgan) | torch==0.4.0 torchvision==0.2.1 | ETH,UCY | FDE,ADE | - |


-----

* 非官方的Baseline代码

| Baseline | Lib. | Dataset | Metrics | Comment |
|-------------------|-------------|---------|--------------|-----------------|
| [linear regression,KNN regression, Vanilla LSTM, GRU](https://github.com/aroongta/Pedestrian_Trajectory_Prediction) | torch==0.4.0 torchvision==0.2.1 | ETH,UCY,Stanford | FDE,ADE,MSE | - |
| [Social LSTM,OLSTM,vanilla LSTM](https://github.com/quancore/social-lstm) | pytorch | UCY, Stanford,BIWI,MOT | FDE,ADE | - |
| [SSLSTM](https://github.com/xuehaouwa/SS-LSTM) | tensorflow | ETH | - | 无任何说明 | 
| [LSTM](https://github.com/trungmanhhuynh/3d_human_trajectory_prediction) | pytorch | ETH,UCY,Stanford | - | - |
| [Hybrid](https://github.com/liuyinglxl/TrajectoryPrediction) | tensorflow | - | - | - |
| [CNN](https://github.com/biy001/social-cnn-pytorch) | pytorch | Stanford,UCY,BIWI | - | - |
| [RNN](https://github.com/karthik4444/nn-trajectory-prediction) | Theano | Stanford,UCY,BIWI | - | - |

