'''
Author: hibana2077 hibana2077@gmail.com
Date: 2023-12-21 23:42:28
LastEditors: hibana2077 hibana2077@gmail.com
LastEditTime: 2023-12-22 00:24:06
FilePath: \fintech_studies\mamba_buy_point_classification\test2.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
import torch.nn as nn

# BCELoss test

criterion = nn.BCELoss()

ans = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
pred = torch.tensor([[0.9, 0.1], [0.1, 0.9]])

loss = criterion(pred, ans)

print(loss)