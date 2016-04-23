var Neural = require('./neur');

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

console.log(guess.out(result));
