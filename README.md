# Interpretable machine learning for HVAC predictive control: A case-study based implementation

## Overview

This is the core code repository for the submitted paper [Title] (Manuscript ID STBE-0220-2022) at **Science and Technology for the Built Environment**. Note that the used dataset is not shared due to the fact that it is a data property of General Technology Ltd, Athens, Greece.

#### Abstract 

Energy efficiency and thermal comfort levels are key attributes to be considered in the design and implementation of a Heating, Ventilation and Air Conditioning (HVAC) system. With the increased availability of Internet of Things (IoT) devices, it is now possible to continuously monitor multiple variables that influence a user’s thermal comfort and the system’s energy efficiency, thus acting pre-emptively to optimize these factors. To this end, this paper reports on a case study with a two-fold aim; first, to analyze the performance of a conventional HVAC system through data analytics; secondly, to explore the use of interpretable machine learning techniques for predictive control with the vision of realizing a trusted and autonomous HVAC system. A new Interpretable Machine Learning (IML) algorithm called Permutation Feature-based Frequency Response Analysis (PF-FRA) is also proposed to quantify the contribution of each predictor in the frequency domain. Results demonstrate that the proposed model can generate accurate forecasts of short-term and long-term Room Temperature (RT) levels by taking into account historical RT information, as well as additional environmental and time-series features. Our proposed model achieves 1.73\% and 4.01\% of Mean Absolute Percentage Error (MAPE) for 1-hour and 8-hour ahead RT prediction, respectively. Tools such as surrogate models and Shapley graphs are employed to explain the model's global and local behaviors with a view to making the machine-made control decision trusted and reliable.

#### Citing

Please use one of the following to cite the code of this repository.
`<
@article{mao2021interpreting,
  title={Interpreting machine learning models for room temperature prediction in non-domestic buildings},
  author={Mao, Jianqiao and Ryan, Grammenos},
  journal={arXiv preprint arXiv:2111.13760},
  year={2021}
}
>`

## Dataset

The data considered in this work was acquired from both indoor and outdoor sensors and system indicators of an HVAC system installed in an 11-story commercial office building in Athens, Greece between December 2017 and September 2020. The building did not employ any other sensors except the ones provided in the dataset. It is worth highlighting that this is not a new build and was constructed in the 1990s with some refurbishments in the HVAC system over the past decades. It is worth noting that for the results presented in this work, the data observations from February 29th, 2020 onwards are excluded from the analysis to avoid the inconsistency in user patterns that inevitably occurred during the breakout of the COVID-19 pandemic. With this in mind, the overall dataset was split into the following subsets: a training set covering the period from December 8th, 2017 to June 30th, 2019; a validation set covering the period from July 1st, 2019 to October 10th, 2019; and a test set covering the period from October 11th, 2019 to February 29th, 2020.

<div align=center><img src=https://github.com/JianqiaoMao/Interpretable-machine-learning-for-HVAC-predictive-control/blob/main/figures/table_dataset_description.png width=800 /></div>

## Framework of data acquisition, processing and modeling phases

The figure below depicts the framework adopted in this work from the data acquisition phase through to the interpretation of the machine learning model employed. The process starts with the collection of the data from different sensors including HVAC status indicators, as well as environmental data from indoor and outdoor sensors. After pre-processing the data and addressing challenges such as different sampling strategies and time misalignment, comprehensive Exploratory Data Analysis (EDA) is conducted to investigate the data distribution of the predictor and target variables, followed by stationarity and hidden pattern analysis. By recursively optimizing our feature engineering strategy based on knowledge discovery from the EDA process, we extract important information that is subsequently used in the predictive modeling phase.

<div align=center><img src=https://github.com/JianqiaoMao/Interpretable-machine-learning-for-HVAC-predictive-control/blob/main/figures/process.png width=800 /></div>

## Performance of Room Temperature Prediction

The figure below compares the finalized model’s predictions with the true room temperature series over the whole period with a detailed plot on the validation and test sets. It is observed that the model successfully forecasts the trends as well as most fluctuations despite failures for some extreme values.


<div align=center><img src=https://github.com/JianqiaoMao/Interpretable-machine-learning-for-HVAC-predictive-control/blob/main/figures/performance.png
 width=800 /></div>
 
To validate our proposed method, we also compare our predictive model’s performance with related studies [1][2] which have comparable settings and evaluation metrics. The table below shows that the proposed modeling method has a more satisfactory performance than the others for the short-term RT prediction. In terms of long-term RT prediction, our method also shows capability with certain performance decay, while the investigated work does not discuss it.

<div align=center><img src=https://github.com/JianqiaoMao/Interpretable-machine-learning-for-HVAC-predictive-control/blob/main/figures/comparison.png
 width=600 /></div>
 
 ## Interpretation (Demo. of PF-FRA)

in this paper, we propose a new global interpretation technique called **Permutation Feature-based Frequency Response Analysis (PF-FRA)** to investigate the features' contribution in the frequency domain. By viewing the features' effects through spectrum analysis, the time-series model can be explained in the frequency domain. This spectrum enables us to identify the features that contribute to the high-frequency components, which in turn lead to fluctuations, as well as the features that contribute to the DC component, which determines the overall trend. By applying the proposed PF-FRA to study how the critical historical RT feature (MVART) boosts the model so significantly, we compare the XGBM regressor’s frequency responses (magnitude only) on the training and validation set with one of the MVART and IOTS features valid in the figure below:

<div align=center><img src=https://github.com/JianqiaoMao/Interpretable-machine-learning-for-HVAC-predictive-control/blob/main/figures/PF-FRA.png width=800 /></div>

## File Description

1) The .py file **final_model_and_interpretation.py** is the core code, including the data loading, pre-processing, feature engineering, modeling and interpretation.

2) The .py file **EDA.py** is the code for Exploratory Data Analysis.

3) The .py file **PFFRA.py** is packaged module that implement the proposed PF-FRA algorithm, which should be imported to run the core code.

4) The .py file **LSTM_RT.py** is the code to use LSTM as the predictive model.

5) The .py file **XGBM_RT_MVA_window_size.py** is the code to select the best window size of moving average filter.

6) The .py file **XGBM_RT_how_long_it_can_pred.py** is the code to compare the performance in predicting different timestamp-ahead room temperatures.

7) The file folder **models** contains the fine-tuned regression models, including the XGBM and LSTM.

8) The file folder **figures** contains the figures and tables shown in the readme file for demonstration.

## Key dependencies

 * numpy 1.19.5
 * pandas 1.1.5
 * scikit-learn 0.24.2
 * tensorflow 2.6.2
 * xgboost 1.5.2
 * seaborn 0.11.2
 * matplotlib 3.3.4
 * holidays 0.13
 * scipy 1.5.4
 * joblib 1.1.0
 * lime 0.2.0.1
 * statsmodels 0.12.2
