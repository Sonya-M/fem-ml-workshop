'use strict';
import * as tf from '@tensorflow/tfjs';
import * as tfd from '@tensorflow/tfjs-data';

const recordButtons = document.getElementsByClassName('record-button');
const buttonContainer = document.getElementById('buttons-container');
const trainButton = document.getElementById('train');
const predictButton = document.getElementById('predict')
const statusElement = document.getElementById('status');

let webcam, initialModel, mouseDown, newModel;

const totals = [0, 0]; // for our labels (we have only 2)
const labels = ['glasses', 'no-glasses'];
const learningRate = 0.0001; // how frequently the model's weights are changed during training
const batchSizeFraction = 0.4;
const epochs = 30; // steps to train model
const denseUnits = 100; // n outputs for layer

let isTraining = false;
let isPredicting = false;

async function loadModel() {
  // for transfer learning, the model used is mobilenet, not coco-ssd
  const mobilenet = await tf.loadLayersModel(
    "https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json"
  );
  // loadLayersModel allows to extract different layers of the ml model ;
  // we'll load a specific layer of the model instead of the model itself,
  // and reuse it to train our model
  const layer = mobilenet.getLayer("conv_pw_13_relu");
  // this particular layer is used for transfer learning
  return tf.model({ inputs: mobilenet.inputs, outputs: layer.output }); // create model from that layer??
  // this will end up being the first layer of our model that we'll build on top of

}

async function init() {
  webcam = await tfd.webcam(document.getElementById('webcam')); // @tensorflow/tfjs-data
  initialModel = await loadModel();
  statusElement.style.display = 'none';
  document.getElementById('controller').style.display = 'block';
}

init();
buttonContainer.onmousedown = (e) => {
  // 0 and 1 here are indices of the elements in the `labels` array above
  if (e.target == recordButtons[0]) {
    handleAddExample(0);
  } else {
    handleAddExample(1);
  }
}

buttonContainer.onmouseup = function () {
  mouseDown = false;
}

async function handleAddExample(labelIndex) {
  mouseDown = true;
  const total = document.getElementById(labels[labelIndex] + '-total');

  while (mouseDown) {
    addExample(labelIndex);
    total.innerText = ++totals[labelIndex];

    await tf.nextFrame() // so we don't block the browser?
  }
}

let xs, xy; //xs: example data; xy: labels attached to pieces of data
async function addExample(index) {
  let img = await getImage();
  let example = initialModel.predict(img); // will be kept in memory and concatenated with subsequent examples until it's ready to be used to create new models

  const y = tf.tidy(() => {
    return tf.oneHot(tf.tensor1d([index]).toInt(), labels.length); // transfer img into 1d array 
  });

  if (xs == undefined) {
    xs = tf.keep(example);
    xy = tf.keep(y)
  } else {
    const prevX = xs;
    xs = tf.keep(prevX.concat(example, 0));

    const prevY = xy;
    xy = tf.keep(prevY.concat(y, 0));

    // clean up memory
    prevX.dispose();
    prevY.dispose();
    y.dispose();
    img.dispose();
  }
}

async function getImage() {
  const img = await webcam.capture();
  const resizedImg = tf.tidy(() => {
    return img.expandDims(0).toFloat().div(127).sub(1);
  }); // tf.tidy() cleans up memory after the function is done
  img.dispose(); // Variables do not get cleaned up when inside a tidy(). To dispose variables, use tf.disposeVariables or call dispose() directly on variables.
  return resizedImg;
}

trainButton.onclick = async function () {
  train();
  statusElement.style.display = 'block';
  statusElement.innerText = 'Training...';
}

async function train() {
  isTraining = true;
  if (!xs) {
    throw new Error('No examples added before training')
  }
  // create tensorflow.js model
  // https://js.tensorflow.org/api/latest/

  // step #1: choose alg - sequential or functional and add layers
  newModel = tf.sequential({
    layers: [
      tf.layers.flatten({ inputShape: initialModel.outputs[0].shape.slice(1) }),
      tf.layers.dense({ units: denseUnits, activation: 'relu', kernelInitializer: 'varianceScaling', useBias: true }),
      tf.layers.dense({ units: labels.length, kernelInitializer: 'varianceScaling', useBias: true, activation: 'softmax' }) // last layer should have as many units as there are labels/classes
    ]
  });
  // step #2: compile with optimizer 
  const optimizer = tf.train.adam(learningRate);
  newModel.compile({ optimizer: optimizer, loss: 'categoricalCrossentropy' });

  // step #2a: split data into batches
  const batchSize = Math.floor(xs.shape[0] * batchSizeFraction);
  if (!(batchSize > 0)) {
    throw new Error(`Batch size is 0 or NaN. Please choose a non-zero fraction.`);
  }

  // step #3: train model
  newModel.fit(xs, xy, {
    batchSize,
    epochs,
    callbacks: {
      onBatchEnd: async (batch, logs) => {
        statusElement.innerText = `Loss: ${logs.loss.toFixed(5)}`;
        // await tf.nextFrame();
      }
    }
  })
  isTraining = false;
}


predictButton.onclick = async function () {
  let keepGoing = true;
  setTimeout(() => {
    keepGoing = false;
  }, 10000);
  isPredicting = true;
  while (isPredicting && keepGoing) {
    const img = await getImage();

    const initialPrediction = initialModel.predict(img);
    const predictions = newModel.predict(initialPrediction);

    const predictedClass = predictions.as1D().argMax();
    const data = await predictedClass.data();
    const classId = data[0];
    console.debug({ data })
    console.log(labels[classId]);

    img.dispose();
    await tf.nextFrame();
  }
}

// currently the model is lost when the page is refreshed
// see https://js.tensorflow.org/api/latest/#tf.LayersModel.save