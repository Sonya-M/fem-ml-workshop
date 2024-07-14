import { clearRect, resetCanvas } from "./utils.js";

const clearButton = document.getElementById('clear-button');
clearButton.addEventListener('click', () => {
  resetCanvas();
  const predictionElem = document.getElementsByClassName('prediction')[0];
  predictionElem.textContent = '';
  clearRect();
});