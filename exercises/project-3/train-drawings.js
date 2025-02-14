const tf = require('@tensorflow/tfjs-node-gpu');
const { loadData, getTrainData, getTestData } = require('./get-data');
const model = require('./create-model');

async function train() {
  loadData();
  const { images: trainImages, labels: trainLabels } = getTrainData();

  model.summary(); // Print a text summary of the Sequential model's layers.

  /** sample output:
__________________________________________________________________________________________
Layer (type)                Input Shape               Output shape              Param #
==========================================================================================
conv2d_Conv2D1 (Conv2D)     [[null,28,28,4]]          [null,26,26,32]           1184
__________________________________________________________________________________________
max_pooling2d_MaxPooling2D1 [[null,26,26,32]]         [null,13,13,32]           0
__________________________________________________________________________________________
flatten_Flatten1 (Flatten)  [[null,13,13,32]]         [null,5408]               0
__________________________________________________________________________________________
dense_Dense1 (Dense)        [[null,5408]]             [null,10]                 54090
__________________________________________________________________________________________
dense_Dense2 (Dense)        [[null,10]]               [null,2]                  22
==========================================================================================
Total params: 55296
Trainable params: 55296
Non-trainable params: 0
__________________________________________________________________________________________
   */



  await model.fit(trainImages, trainLabels, { epochs: 10, batchSize: 5, validationSplit: 0.2 });
  // validationSplit: fraction of the training data to be used as validation data. The model will set apart this fraction of the training data, will not train on it, and will evaluate the loss and any model metrics on this data at the end of each epoch. The validation data is selected from the last samples in the x and y data provided, before shuffling.
  // not the same as testing data


  const { images: testImages, labels: testLabels } = getTestData();
  const evalOutput = model.evaluate(testImages, testLabels);
  const loss = evalOutput[0].dataSync()[0].toFixed(3);
  const accuracy = evalOutput[1].dataSync()[0].toFixed(3);

  console.log({ loss, accuracy });

  await model.save('file://public/model')
}

train();

// run this file with node train-drawings.js
// the accuracy changes after each epoch, and the final values can be different each time you run it
// it's not a pure fn
// it's pretty unpredictable if you don't understand what is actually happening under the hood, but you can stop once you see a good accuracy value 