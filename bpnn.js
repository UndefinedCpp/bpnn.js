/**
 * 这是一个 BP 神经网络的 JavaScript 极简实现
 * 
 * A minimized implementation of BP neural network
 */
module.exports.VERSION = '0.1.0';
b = false;
const preset = {
    /** sigmoid激活函数 */
    sigmoid(x) {
        return 1.0 / (1.0 + Math.exp(-x));
    },
    /** ReLU激活函数 */
    relu(x) {
        return Math.max(0, x);
    },
    /** tanh激活函数 */
    tanh(x) {
        return Math.tanh(x);
    },
    softsign(x) {
        return x / (1 + Math.abs(x));
    },
    stddif(x) {
        return x * (1 - x);
    }
};
module.exports.preset = preset;


class Network {
    /**
     * @param {NetworkOptions} opts 
     */
    constructor(opts) {
        // 读取配置
        const { layers, activateFn, diffFn, learningRate, iterate, eps } = opts;
        // 配置隐层
        if (!(layers instanceof Array)) {
            throw new TypeError('parameter "layers" is not an array');
        }
        this.layerN = layers.length;
        this.layers = layers;
        // 输出函数
        this.fn = activateFn ?? preset.sigmoid;
        // 偏差函数
        this.fd = diffFn ?? preset.stddif;
        // weight 和 bias
        this.w = [];
        this.b = [];
        // 学习速率
        this.μ = learningRate ?? 0.2;
        // 迭代次数
        this.iter = iterate ?? 5000;
        // 误差
        this.error = 0.0;
        // 允许误差
        this.eps = eps ?? 0.0001;
        // 随机初始化 weight 和 bias
        for (let l = 1; l < this.layerN; l++) {
            let _w = [];
            let _b = [];
            for (let j = 0; j < layers[l]; j++) {
                let temp = [];
                for (let i = 0; i < layers[l - 1]; i++) {
                    temp[i] = Math.random();
                }
                _w.push(temp);
                _b.push(Math.random());
            }
            this.w[l] = _w;
            this.b[l] = _b;
        }
    }

    // 向前传播
    forward(x) {
        let y = [];
        y[0] = x;
        for (let l = 1; l < this.layerN; l++) {
            y[l] = [];
            for (let j = 0; j < this.layers[l]; j++) {
                // 计算输入综合 Sigma[w_p * o_p + theta]
                let u = 0.0;
                for (let i = 0; i < this.layers[l - 1]; i++) {
                    u = u + this.w[l][j][i] * y[l - 1][i];
                }
                u = u + this.b[l][j];
                // 激活函数映射
                y[l][j] = this.fn(u);
            }
        }
        return y;
    }

    /**
     * 计算网络的误差
     * @param {number[]} d 
     * @param {number[][]} y 
     * @returns {number[][]}
     */
    delta(d, y) {
        let delta = [];
        let last = [];
        for (let j = 0; j < this.layers[this.layerN - 1]; j++) {
            last[j] = (d[j] - y[this.layerN - 1][j]) * this.fd(y[this.layerN - 1][j]);
        }
        delta[this.layerN - 1] = last;
        for (let l = this.layerN - 2; l > 0; l--) {
            delta[l] = [];
            for (let j = 0; j < this.layers[l]; j++) {
                delta[l][j] = 0.0;
                for (let i = 0; i < this.layers[l + 1]; i++) {
                    delta[l][j] += delta[l + 1][i] * this.w[l + 1][i][j];
                }
                delta[l][j] = this.fd(y[l][j]) * delta[l][j];
            }
        }
        return delta;
    }

    /**
     * 根据误差调整权重
     * @param {number[][]} y 正向传播的输出权重
     * @param {number[][]} delta 误差
     */
    update(y, delta) {
        for (let l = 0; l < this.layerN; l++) {
            for (let j = 0; j < this.layers[l]; j++) {
                for (let i = 0; i < this.layers[l - 1]; i++) {
                    this.w[l][j][i] += this.μ * delta[l][j] * y[l - 1][i];
                    this.b[l][j] += this.μ * delta[l][j];
                }
            }
        }
    }

    /**
     * 训练网络
     * @param {number[][]} x 样例输入 
     * @param {number[][]} d 预期输出
     * @example
     * var in = [[0, 0], [0, 1], [1, 0], [1, 1]];
     * var out = [[0], [1], [1], [0]];
     * network.train(in, out);
     */
    train(x, d) {
        for (let p = 0; p < this.iter; p++) {
            this.error = 0;
            for (let i = 0; i < x.length; i++) {
                let y = this.forward(x[i]);
                let delta = this.delta(d[i], y);
                this.update(y, delta);
                let ep = 0.0;
                let l1 = this.layerN - 1;
                for (let l = 0; l < this.layers[l1]; l++) {
                    ep += (d[i][l] - y[l1][l]) * (d[i][l] - y[l1][l]);
                }
                this.error += ep / 2.0;
            }
            if (this.error < this.eps) {
                break;
            }
        }
    }

    predict(x) {
        let y = this.forward(x);
        return y[this.layerN - 1];
    }
}
module.exports.Network = Network;