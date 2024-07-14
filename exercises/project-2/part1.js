const path = './tm-my-image-model/';
const startBtn = document.getElementById('start');

startBtn.onclick = () => { init(); }
let model, webcam;


async function init() {
  const modelPath = path + 'model.json';
  const metadataPath = path + 'metadata.json';

  model = await tmImage.load(modelPath, metadataPath); // teachable-machine script imported in index.html

  let maxPredictions = model.getTotalClasses();

  webcam = new tmImage.Webcam(200, 200, /* flip: */ false); // for left/right, set flip to true
  await webcam.setup(); // request access to camera
  await webcam.play(); // start webcam feed in browser

  window.requestAnimationFrame(loop)

  document.getElementById('webcam-container').appendChild(webcam.canvas);
}


async function loop() {
  webcam.update() // update webcam frame
  // when camera is updated, need to predict what is on the screen
  await predict();
  window.requestAnimationFrame(loop)
}

async function predict() {
  const predictions = await model.predict(webcam.canvas) // pass the image
  console.log(predictions);
  /* sample output:
  [
  {className: 'with-glasses', probability: 0.09430024027824402},
  {className: 'without-glasses', probability: 0.9056997299194336}
  ]
  */
  const topPrediction = predictions.toSorted((a, b) => a.probability - b.probability).at(-1);
  console.log(`prediction: ${topPrediction.className} (probability: ${topPrediction.probability})`)
}