# 配置环境，导入库函数
import os
# os.environ['DEVICE_ID'] = '0'
import numpy as np
import mindspore as ms
from mindspore import nn
from mindspore import context
context.set_context(mode=context.GRAPH_MODE, device_target="Ascend") # CPU

x = np.arange(-5, 5, 0.3)[:32].reshape((32, 1))
y = -5 * x + 0.1 * np.random.normal(loc=0.0, scale=20.0, size=x.shape)

net = nn.Dense(1, 1)
loss_fn = nn.loss.MSELoss()
opt = nn.optim.SGD(net.trainable_params(), learning_rate = 0.01)
with_loss = nn.WithLossCell(net, loss_fn)
train_step = nn.TrainOneStepCell(with_loss, opt).set_train()

# 观察x和y的形状
print(x.shape) # (32, 1)
print(y.shape) # (32, 1)

for epoch in range(20):
    loss = train_step(ms.Tensor(x, ms.float32), ms.Tensor(y, ms.float32))
    print('epoch: {0}, loss is {1}'.format(epoch, loss))

wb = [x.asnumpy() for x in net.trainable_params()]
w, b = np.squeeze(wb[0]), np.squeeze(wb[1])
print('The true linear function is y = -5 * x + 0.1')
print('The trained linear model is y = {0} * x + {1}'.format(w, b))

for i in range(-10, 11, 5):
    print('x = {0}, predicted y = {1}'.format(i, net(ms.Tensor([[i]], ms.float32))))

# 作图
from matplotlib import pyplot as plt
plt.scatter(x, y, label='Samples')
plt.plot(x, w * x + b, c = 'r', label='True function')
plt.plot(x, -5 * x + 0.1, c='b', label='Trained model')
plt.legend()
plt.show()