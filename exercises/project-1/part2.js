import '@tensorflow/tfjs'
import * as cocoSsd from '@tensorflow-models/coco-ssd'
import { showResult, startWebcam, takePicture } from './utils';


const webcamBtn = document.getElementById('webcam');
const captureBtn = document.getElementById('pause');
const video = document.querySelector('video');

let model;

async function init() {
  model = await cocoSsd.load();
}

webcamBtn.onclick = function () {
  startWebcam(video);
}

captureBtn.onclick = function () {
  takePicture(video, predict);
}

async function predict(img) {
  const predictions = await model.detect(img);
  console.log(predictions);
  showResult(predictions);
}

init();