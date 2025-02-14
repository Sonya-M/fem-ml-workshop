import { clearRect, displayPrediction, getCanvas, resetCanvas } from "./utils.js";

const clearButton = document.getElementById('clear-button');
const predictButton = document.getElementById('check-button');

clearButton.addEventListener('click', () => {
  resetCanvas();
  const predictionElem = document.getElementsByClassName('prediction')[0];
  predictionElem.textContent = '';
  clearRect();
});

let model;
const modelPath = './model/model.json';

async function loadModel(path) {
  // tf imported in index.html
  if (!model) model = await tf.loadLayersModel(path)
}

predictButton.addEventListener('click', async function () {
  const canvas = getCanvas();
  const drawing = canvas.toDataURL();
  const newImg = document.getElementsByClassName('imageToCheck')[0];
  newImg.src = drawing;
  newImg.addEventListener('load', async function () {
    predict(newImg)
  });
  resetCanvas();
});

async function predict(img) {
  img.width = 200;
  img.height = 200;

  const processedImg = await tf.browser.fromPixelsAsync(img, 4);
  const resizedImg = tf.image.resizeNearestNeighbor(processedImg, [28, 28]);
  const castImg = tf.cast(resizedImg, 'float32');
  let shape;
  const predictions = await model.predict(tf.reshape(castImg, (shape = [1, 28, 28, 4]))).data();

  const label = predictions.indexOf(Math.max(...predictions));
  displayPrediction(label);
}

loadModel(modelPath);