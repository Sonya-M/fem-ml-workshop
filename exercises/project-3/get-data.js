const tf = require('@tensorflow/tfjs-node-gpu'); // will create tensors from images
const fs = require('fs');
const path = require('path');

const trainImagesDir = './data/train';
const testImagesDir = './data/test';

let trainData, testData;

function loadImages(dataDir) {
  const images = [];
  const labels = [];

  let files = fs.readdirSync(dataDir);
  for (let i = 0; i < files.length; i++) {
    let filePath = path.join(dataDir, files[i]);
    let buffer = fs.readFileSync(filePath);
    let imgTensor = tf.node.decodeImage(buffer).resizeNearestNeighbor([28, 28]).expandDims(); // resizeNearestNeighbor: transform image from 200x200 to 28x28 -  use same for building model
    // expandDims: add a dimension (with null/0 val) to the tensor to match the model input shape
    images.push(imgTensor);

    const circle = files[i].toLowerCase().endsWith('circle.png');
    const triangle = files[i].toLowerCase().endsWith('triangle.png');

    // cannot pass strings to tensorflow, so we use idcs
    if (circle) {
      labels.push(0); // idx 0 for circle
    } else if (triangle) {
      labels.push(1);
    }
  }
  return [images, labels];
}

function loadData() {
  console.debug('Loading images...');
  trainData = loadImages(trainImagesDir);
  testData = loadImages(testImagesDir);
  console.debug('Images loaded');
}

function getTrainData() {
  return {
    images: tf.concat(trainData[0]),
    labels: tf.oneHot(tf.tensor1d(trainData[1], 'float32'), 2) // last arg is the number of classes
  }
}

function getTestData() {
  return {
    images: tf.concat(testData[0]),
    labels: tf.oneHot(tf.tensor1d(testData[1], 'float32'), 2),
  }
}

module.exports = {
  loadData, getTestData, getTrainData
}