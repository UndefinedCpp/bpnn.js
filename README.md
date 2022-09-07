# bpnn.js
BP神经网络的javascript实现

## 使用

```javascript
// 1. 导入模块 Import the module
const bpnn = require('path/to/bpnn.js');
// 2. 准备训练样本 / Prepare training data
//    这里是一个训练进位的demo / This is a demo of learning to carry in in addition
//    i 是输入数据集，o 是预期输出 / 'i' is input and 'o' is expected output
const i = [
    [0.5, 0.2], [0.9, 0.8],
    [0.5, 0.2], [0, 0.2],
    [0.9, 0], [0.2, 0.6],
    [0.7, 0.4], [0.1, 0.4],
    [0.2, 0.4], [0.5, 0.7]
];
const o = [
    [0], [1], [0],
    [0], [0], [0],
    [1], [0], [0],
    [1]
];
// 3. 创建网络 / Create instance
const opt = { layers: [2, 4, 4, 1] };
const instance = new bpnn.Network(opt);
// 4. 训练网络 / Train the network
instance.train(i, o);
// 5. 预测 / Predict
console.assert(instance.predict([0.3, 0.8])[0] > 0.8);  // true
```
