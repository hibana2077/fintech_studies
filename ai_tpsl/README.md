<!--
 * @Author: hibana2077 hibana2077@gmail.com
 * @Date: 2023-06-27 22:15:36
 * @LastEditors: hibana2077 hibana2077@gmail.com
 * @LastEditTime: 2023-07-04 11:14:00
 * @FilePath: \fintech_studies\ai_tpsl\README.md
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
-->
# AI TPSL

## 1. Introduction

## 2. Usage

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