<template>
  <div class="board-canvas-wrapper" oncontextmenu="return false;" v-on:mousedown="analyzeScreenTouch">
    <div id="bugs-board">
      <audio ref="audio1">
        <source src="../../assets/sounds/2.mp3" type="audio/mpeg">
      </audio>
      <canvas id="backgroundCanvas" v-bind:style="{background: bugsSettings.backgroundColor}"
              v-bind:height="canvasParams.height" v-bind:width="canvasParams.width"></canvas>
      <canvas id="bugCanvas" v-bind:height="canvasParams.height" v-bind:width="canvasParams.width"
              v-on:mousedown="setCanvasClick($event)">
                <holes-bug v-for="(value, index) in bugsProps"
                   :key="index"
                   :bugsSettings="bugsSettings"
                   :exit-hole-pos="exitHolePos"
                   :entrance-hole-pos="entranceHolePos"
                   ref="bugChild"
                   v-on:bugRetreated="endTrial">
                </holes-bug>
      </canvas>
    </div>
  </div>
</template>

<script>
import holesBug from '../bugs/holesBug.vue'
import boardsMixin from './boardsMixin'

export default {
  name: 'holesBoard',
  components: {holesBug},
  mixins: [boardsMixin],
  data() {
    return {
      bugsSettings: { // extends the mixin's bugSettings
        holeSize: [200, 200],
        exitHole: 'bottomRight',
        entranceHole: null
      },
      pad: 100 // padding for holes
    }
  },
  computed: {
    holesPositions: function () {
      let [canvasW, canvasH] = [this.canvas.width, this.canvas.height]
      let [holeW, holeH] = this.bugsSettings.holeSize
      return {
        bottomLeft: [this.pad, canvasH - holeH - this.pad],
        bottomRight: [canvasW - holeW - this.pad, canvasH - holeH - this.pad]
      }
    },
    exitHolePos: function () {
      return this.holesPositions[this.bugsSettings.exitHole]
    },
    entranceHolePos: function () {
      let entranceHole = this.bugsSettings.exitHole === 'bottomLeft' ? 'bottomRight' : 'bottomLeft'
      return this.holesPositions[entranceHole]
    }
  },
  methods: {
    initDrawing() {
      let image = new Image()
      let canvas = document.getElementById('backgroundCanvas')
      let ctx = canvas.getContext('2d')
      let [holeW, holeH] = this.bugsSettings.holeSize
      let that = this
      image.onload = function () {
        ctx.drawImage(image, that.exitHolePos[0], that.exitHolePos[1], holeW, holeH)
        ctx.drawImage(image, that.entranceHolePos[0], that.entranceHolePos[1], holeW, holeH)
      }
      image.src = require('@/assets/hole2.png')
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
</style>