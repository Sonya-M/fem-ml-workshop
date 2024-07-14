const tf = require('@tensorflow/tfjs');
// https://cs231n.github.io/convolutional-networks/
const kernelSize = [3, 3]; // int or int array
const nFilters = 32; // number of filters the data goes thru in each step/layer
const nClasses = 2;

const model = tf.sequential();
model.add(tf.layers.conv2d({
  inputShape: [28, 28, 4], // 4 for RGBA,
  filters: nFilters,
  kernelSize,
  activation: 'relu', // 'softmax' usu. for last layer
})
);

model.add(tf.layers.maxPooling2d({
  poolSize: [2, 2],
})
);

model.add(tf.layers.flatten()); // takes output of prev layer and flattens it into a 1D array - that's what a dense layer expects, which is next

model.add(tf.layers.dense({
  units: 10, // n output units
  activation: 'relu',
}));

model.add(tf.layers.dense({
  units: nClasses,
  activation: 'softmax',
})) // for last layer, units = nClasses/labels, activation = 'softmax'

// compile with optimizer, loss function, and metrics
const optimizer = tf.train.adam(0.001); // learning rate
model.compile({
  optimizer,
  loss: 'categoricalCrossentropy', // the smaller the loss, the higher the accuracy
  metrics: ['accuracy'],
});

module.exports = model;