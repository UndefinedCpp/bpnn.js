interface NetworkOptions {
    /**
     * 神经网络的神经元配置
     */
    layers: number[];

    /**
     * 激活函数
     */
    activateFn?: (x: number) => number;

    /**
     * 偏差函数
     */
    diffFn?: (x: number) => number;

    /**
     * 学习速率
     */
    learningRate?: number;

    /**
     * 迭代次数
     */
    iterate?: number;

    /**
     * 目标误差
     */
    eps?: number;
}