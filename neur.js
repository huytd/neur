var sigmoid = require('sigmoid');
var sigmoidPrime = require('sigmoid-prime');
var Matrix = require('node-matrix');
var sample = require('samples');

var scalar = Matrix.multiplyScalar;
var dot = Matrix.multiplyElements;
var multiply = Matrix.multiply;
var subtract = Matrix.subtract;
var add = Matrix.add;

module.exports = Neural;

function Neural(opts) {
    if (!(this instanceof Neural)) return new Neural(opts);
    opts = opts || {};
    this.activate = sigmoid;
    this.activatePrime = sigmoidPrime;

    this.learningRate = opts.learningRate || 0.7;
    this.iterations = opts.iterations || 10000;
    this.hiddenUnits = opts.hiddenUnits || 3;
}

Neural.prototype.forward = function (examples) {
    var activate = this.activate;
    var weights = this.weights;
    var ret = {};

    ret.hiddenSum = multiply(weights.inputHidden, examples.input);
    ret.hiddenResult = ret.hiddenSum.transform(activate);
    ret.outputSum = multiply(weights.hiddenOutput, ret.hiddenResult);
    ret.outputResult = ret.outputSum.transform(activate);

    return ret;
};

Neural.prototype.back = function(examples, results) {
  var activatePrime = this.activatePrime;
  var learningRate = this.learningRate;
  var weights = this.weights;

  // compute weight adjustments
  var errorOutputLayer = subtract(examples.output, results.outputResult);
  var deltaOutputLayer = dot(results.outputSum.transform(activatePrime), errorOutputLayer);
  var hiddenOutputChanges = scalar(multiply(deltaOutputLayer, results.hiddenResult.transpose()), learningRate);
  var deltaHiddenLayer = dot(multiply(weights.hiddenOutput.transpose(), deltaOutputLayer), results.hiddenSum.transform(activatePrime));
  var inputHiddenChanges = scalar(multiply(deltaHiddenLayer, examples.input.transpose()), learningRate);

  // adjust weights
  weights.inputHidden = add(weights.inputHidden, inputHiddenChanges);
  weights.hiddenOutput = add(weights.hiddenOutput, hiddenOutputChanges);

  return errorOutputLayer;
};

Neural.prototype.normalize = function (data) {
    var ret = { input: [], output: [] };

    for (var i = 0; i < data.length; i++) {
      var datum = data[i];

      ret.output.push(datum.output);
      ret.input.push(datum.input);
    }

    ret.output = Matrix(ret.output);
    ret.input = Matrix(ret.input);

    return ret;
};

Neural.prototype.learn = function(examples) {
  examples = this.normalize(examples);

  this.weights = {
    inputHidden: Matrix({
      columns: this.hiddenUnits,
      rows: examples.input[0].length,
      values: sample
    }),
    hiddenOutput: Matrix({
      columns: examples.output[0].length,
      rows: this.hiddenUnits,
      values: sample
    })
  };

  for (var i = 0; i < this.iterations; i++) {
    var results = this.forward(examples);
    var errors = this.back(examples, results);
  }

  return this;
};

Neural.prototype.predict = function (data) {
    var results = this.forward({ input: Matrix([data]) });
    return results.outputResult[0];
};

Neural.prototype.model = function (input) {
    return {
        _ref: input,
        in: function(src) {
            var output = [];
            for (key in src) {
                output = output.concat(src[key]);
            }
            return output;
        },
        out: function(src) {
            var output = Object.assign({}, this._ref);
            var keys = Object.keys(output);
            for (var i = 0; i < src.length; i++) {
                output[keys[i]] = src[i];
            }
            return output;
        }
    }
};
