# Interpretable machine learning for HVAC predictive control: A case-study based implementation

## Overview

Energy efficiency and thermal comfort levels are key attributes to be considered in the design and implementation of a Heating, Ventilation and Air Conditioning (HVAC) system. With the increased availability of Internet of Things (IoT) devices, it is now possible to continuously monitor multiple variables that influence a user’s thermal comfort and the system’s energy efficiency, thus acting pre-emptively to optimize these factors. \textcolor{red}{To this end, this paper reports on a case study with a two-fold aim; first, to analyze the performance of a conventional HVAC system through data analytics; secondly, to explore the use of interpretable machine learning techniques for predictive control with the vision of realizing a trusted and autonomous HVAC system.} A new Interpretable Machine Learning (IML) algorithm called Permutation Feature-based Frequency Response Analysis (PF-FRA) is also proposed to quantify the contribution of each predictor in the frequency domain. Results demonstrate that the proposed model can generate accurate forecasts of short-term and long-term Room Temperature (RT) levels by taking into account historical RT information, as well as additional environmental and time-series features. Our proposed model achieves 1.73\% and 4.01\% of Mean Absolute Percentage Error (MAPE) for 1-hour and 8-hour ahead RT prediction, respectively. Tools such as surrogate models and Shapley graphs are employed to explain the model's global and local behaviors with a view to making the machine-made control decision trusted and reliable.

## Dataset

<div align=center><img src=https://github.com/JianqiaoMao/Interpretable-machine-learning-for-HVAC-predictive-control/blob/main/figures/table_dataset_description.png width=750 /></div>
