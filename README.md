# neur

Simple neural network implementation in JS

## Installation

Install via npm:

```
npm install neur
```

## Usage

**Step 1:** Initialize a network

```js
var Neural = require('neur');
var neural = Neural();
```

or 

```js
var neural = require('neur')();
```

You can config the network by passing the options to the constructor:

```js
var neural = Neural({
  learningRate: 0.7,
  iterations: 10000,
  hiddenUnits: 3
});
```

By default, we have a network with `1` hidden layer which has `3` neurons in total. And it will train the input data `10,000` times.

**Step 2:** Train a network

```js
neural.learn([
  { input: [ <input-values> ], output: [ <output-values> ] },
  ...
]);
```

**Step 3:** Predict

```js
var result = neural.predict([ <predict-input-values> ]);
```

## Examples

### Example 1: Basic using

```js
var Neural = require('neur');

var neural = Neural()
    .learn([
      { input: [0, 0], output: [0] },
      { input: [0, 1], output: [1] },
      { input: [1, 1], output: [1] },
      { input: [1, 0], output: [0] }
  ]);

console.log(neural.predict([1, 0])); // ~0
console.log(neural.predict([1, 1])); // ~1
```

### Example 2: Use Model mapping to train complex data

```js
var Neural = require('neur');

var color = Neural().model({ r: 0, g: 0, b: 0 });
var guess = Neural().model({ black: 0, white: 0 });

var result = Neural()
    .learn([
      {
          input: color.in({ r: 0.03, g: 0.7, b: 0.5 }),
          output: guess.in({ black: 1, white: 0 })
      },
      {
          input: color.in({ r: 0.16, g: 0.09, b: 0.2 }),
          output: guess.in({ black: 0, white: 1 })
      },
      {
          input: color.in({ r: 0.5, g: 0.5, b: 1.0 }),
          output: guess.in({ black: 0, white: 1 })
      }
    ])
    .predict(color.in({ r: 1, g: 0.4, b: 0 }));

console.log(guess.out(result)); // { black: ~0, white: ~1 }
```
