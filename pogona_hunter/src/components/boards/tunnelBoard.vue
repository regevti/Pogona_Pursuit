<template>
  <div class="board-canvas-wrapper" oncontextmenu="return false;" v-on:mousedown="analyzeScreenTouch">
  <div id="bugs-board">
    <audio ref="audio1">
      <source src="../../assets/sounds/2.mp3" type="audio/mpeg">
    </audio>
    <canvas id="backgroundCanvas" v-bind:style="{background: bugsSettings.backgroundColor}"
            v-bind:height="canvasParams.height" v-bind:width="canvasParams.width"></canvas>
    <canvas id="tunnelCanvas" v-bind:height="tunnelHeight" v-bind:width="canvasParams.width"></canvas>
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
      return this.canvasParams.height / 2
    }
  },
  methods: {
    initDrawing() {
      let canvas = document.getElementById('tunnelCanvas')
      canvas.style.top = `${this.canvas.height / 4}px`
      // canvas.height = `${this.canvas.height / 2}px`
      // canvas.width = `${this.canvas.width}px`
      let ctx = canvas.getContext('2d')
      // ctx.fillRect(0, this.canvas.height / 4, this.canvas.width, this.canvas.height / 2)
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