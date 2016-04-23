var Neural = require('./neur');

var result = Neural()
    .learn([
      { input: [0.03, 0.7, 0.5], output: [1, 0] },
      { input: [0.16, 0.09, 0.2], output: [0, 1] },
      { input: [0.5, 0.5, 1.0], output: [0, 1] }
    ])
    .predict([1, 0.4, 0]);

console.log(result);
