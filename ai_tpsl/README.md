<!--
 * @Author: hibana2077 hibana2077@gmail.com
 * @Date: 2023-06-27 22:15:36
 * @LastEditors: hibana2077 hibana2077@gmail.com
 * @LastEditTime: 2023-07-04 11:21:09
 * @FilePath: \fintech_studies\ai_tpsl\README.md
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
-->
# AI TPSL

![python](https://img.shields.io/badge/python-3.10-blue?style=plastic-square&logo=python)
![tensorflow](https://img.shields.io/badge/tensorflow-2.12.0-orange?style=plastic-square&logo=tensorflow)
![keras](https://img.shields.io/badge/keras-2.12.0-red?style=plastic-square&logo=keras)
![numpy](https://img.shields.io/badge/numpy-1.19.5-blue?style=plastic-square&logo=numpy)
![pandas](https://img.shields.io/badge/pandas-2.0.2-blue?style=plastic-square&logo=pandas)
![matplotlib](https://img.shields.io/badge/matplotlib-3.7.1-blue?style=plastic-square&logo=matplotlib)
![Openai-GYM](https://img.shields.io/badge/Openai--GYM-0.26.2-blue?style=plastic-square&logo=Openai-GYM)

## 1. Introduction

## 2. Usage

Install dependencies

```bash
pip install -r requirements.txt
```

Go to src directory

```bash
cd src
```

Get data from exchange

```bash
python get_csv.py
```

Data preprocessing (Add features)

```bash
python data_preprocessing.py -i <input_file>
```

Make label

```bash
python make_ans.py -i <input_file> -l <leverage> -q <quantity> -t <stop_loss_threshold>
```

Train model

```bash
python train.py -i <input_file> -e <epochs> -b <batch_size> -l <learning_rate> -m <model_name> -w <window_size> -ep <early_stopping_patience> -ev <early_stopping_verbose>
```

When you finish training, it will save the model with the name `model_name` and the loss curve with the name `loss.png`.
And you can apply the model to the trading bot.