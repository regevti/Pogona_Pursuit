export function handlePrediction(options, ctx, canvasParams) {
  options = JSON.parse(options)
  console.log(`Prediction detected coords: ${options.hit_point}, time2hit:${options.time2hit}`)
  ctx.fillRect(canvasParams.width / 2, canvasParams.height / 2, 300, 200)
  let t = setTimeout(() => {
    ctx.clearRect(0, 0, canvasParams.width, canvasParams.height)
    clearTimeout(t)
  }, 500)
}


export function showPogona(canvas, numFrames) {
  numFrames = Number(numFrames)
  const image = new Image(60, 45)
  image.onload = drawImageActualSize
  image.src = '/static/pogona0.jpg'

  function drawImageActualSize() {
    let canvasOriginalHeight = canvas.height
    canvas.height = naturalHeight
    ctx.drawImage(this, canvas.width / 4, 0)
    if (numFrames > 0) {
      let t = setTimeout(() => {
        ctx.clearRect(0, 0, canvas.width, canvas.height)
        canvas.height = canvasOriginalHeight
        clearTimeout(t)
      }, 1000 * numFrames / 60) // for 2 frames
    }
  }
}