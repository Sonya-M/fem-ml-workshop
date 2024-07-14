import '@tensorflow/tfjs'
import * as cocoSsd from '@tensorflow-models/coco-ssd'
import { handleFilePicker, showResult } from './utils';

let model;

async function init() {
  model = await cocoSsd.load();
  handleFilePicker(predict);
}

async function predict(img) {
  const predictions = await model.detect(img);
  console.log(predictions);
  showResult(predictions);
}

init();
