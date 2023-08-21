import torch
import torch.nn as nn
lr = 0.1
x = torch.tensor([5.0, 10.0]).reshape(1, 2)
w1 = torch.tensor([[0.1, 0.15, 0.2], [0.25, 0.3, 0.35]], requires_grad=True)
b1 = 0.35
w2 = torch.tensor([[0.4, 0.45], [0.5, 0.55], [0.6, 0.65]], requires_grad=True)
b2 = 0.65
y = torch.tensor([0.01, 0.99]).reshape(1, 2)
for i in range(100):
    net_h = torch.matmul(x, w1) + b1
    out_h = torch.sigmoid(net_h)
    net_o = torch.matmul(out_h, w2) + b2
    out_o = torch.sigmoid(net_o)
    # out_o.backward(torch.ones_like(out_o))  # 注意一开始就是此处出错，改成loss.backward()
    loss = torch.sum(torch.square(out_o - y))
    print(loss)
    loss.backward()
    w1_grad = lr * w1.grad
    w2_grad = lr * w2.grad
    with torch.no_grad():  # 注意加上这个后就不会对w1进行梯度跟踪
        w1 -= w1_grad  # 原地操作， 不能写成 w1 = w1 - w1_grad 
        w2 -= w2_grad
        # w1 = w1 - w1_grad 
        # w2 = w2 - w2_grad 
        w1.grad.zero_()
        w2.grad.zero_()  # 注意此处梯度要清零，否则会梯度会累计， 在这个例子里面不清零也没事