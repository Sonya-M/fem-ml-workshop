import '@tensorflow/tfjs'
import '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-webgl';
import * as faceDetection from '@tensorflow-models/face-detection';
import { startWebcam, takePicture, drawFaceBox } from './utils';

const webcamBtn = document.getElementById('webcam');
const captureBtn = document.getElementById('pause');
const video = document.querySelector('video');

let model, detector;

async function init() {
  model = faceDetection.SupportedModels.MediaPipeFaceDetector;

  detector = await faceDetection.createDetector(model, {
    runtime: 'tfjs',
  })
}

webcamBtn.onclick = function () {
  startWebcam(video);
}

captureBtn.onclick = function () {
  takePicture(video, predict);
}

async function predict(img) {
  const faces = await detector.estimateFaces(img, { flipHorizontal: false, });
  console.log(faces)
  drawFaceBox(img, faces);
}

init();