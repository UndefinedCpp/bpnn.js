const m = require('./bpnn.js');
const Network = m.Network;

function test(opts, i, o, num) {
    const n = new Network(opts);
    console.time('bpnet train ' + num);
    n.train(i, o);
    console.timeEnd('bpnet train ' + num);
    return n;
}

// 学习进位
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
const model = test({ layers: [2, 5, 5, 1] }, i, o, 0);
console.log('Training done, testing accuracy');

let correctNum = 0;
for (let i = 0; i < 1000; ++i) {
    const p = () => Math.floor(Math.random() * 100) / 100;
    const testPair = [p(), p()];
    const predicted = model.predict(testPair)[0];
    const actual = testPair[0] + testPair[1] >= 1.0;
    if ((predicted >= 0.5) === actual) {
        correctNum++;
    }
    else {
        console.log(`[!] Error: `, testPair, 'is expected to be', actual, 'but got', predicted);
    }
}
console.log(correctNum / 10, '% correct!');