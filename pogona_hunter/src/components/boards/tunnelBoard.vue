<template>
  <div class="board-canvas-wrapper" oncontextmenu="return false;" v-on:mousedown="analyzeScreenTouch">
  <div id="bugs-board">
    <audio ref="audio1">
      <source src="../../assets/sounds/2.mp3" type="audio/mpeg">
    </audio>
    <canvas id="backgroundCanvas" v-bind:style="{background: bugsSettings.backgroundColor}"
            v-bind:height="canvasParams.height" v-bind:width="canvasParams.width"></canvas>
    <canvas id="tunnelCanvas" v-bind:height="tunnelHeight" v-bind:width="tunnelWidth"></canvas>
    <canvas id="bugCanvas" v-bind:height="canvasParams.height" v-bind:width="canvasParams.width"
            v-on:mousedown="setCanvasClick($event)">
      <tunnel-bug v-for="(value, index) in bugsProps"
                 :key="index"
                 :bugsSettings="bugsSettings"
                 ref="bugChild"
                 v-on:bugRetreated="endTrial">
      </tunnel-bug>
    </canvas>
  </div>
</div>
</template>

<script>
import boardsMixin from './boardsMixin'
import tunnelBug from '../bugs/tunnelBug.vue'
export default {
  name: 'tunnelBoard',
  components: {tunnelBug},
  mixins: [boardsMixin],
  computed: {
    tunnelHeight() {
      return window.innerHeight
    },
    tunnelWidth() {
      return this.canvasParams.width / 10
    }
  },
  mounted() {
    const canvas = document.getElementById('tunnelCanvas')
    const context = canvas.getContext('2d')
    const img = new Image()
    img.onload = () => {
      context.drawImage(img, 0, 0, canvas.width, canvas.height)
    }
    img.src = require('@/assets/wooden-logs-wall.jpg')
  },
  methods: {
    initDrawing() {
      let canvas = document.getElementById('tunnelCanvas')
      canvas.style.left = `${this.canvas.width / 2 - this.tunnelWidth}px`
      let ctx = canvas.getContext('2d')
      ctx.fillRect(0, 0, canvas.width, canvas.height)
    }
  }
}
</script>

<style scoped>

#bugCanvas {
  padding: 0;
  z-index: 1;
  display: block;
  position: absolute;
  bottom: 0;
  top: auto;
}

#tunnelCanvas {
  padding: 0;
  z-index: 2;
  display: block;
  position: absolute;
  top: auto;
  bottom: 0;
}

</style>