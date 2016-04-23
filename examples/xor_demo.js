var Neural = require('../index');

var neural = Neural()
    .learn([
      { input: [0, 0], output: [0] },
      { input: [0, 1], output: [1] },
      { input: [1, 1], output: [1] },
      { input: [1, 0], output: [0] }
  ]);

console.log(neural.predict([1, 0])); // ~0
console.log(neural.predict([1, 1])); // ~1
