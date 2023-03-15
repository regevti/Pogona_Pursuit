<template>
  <div>
    <img ref="bugImg" :src="bugImgSrc" alt=""/>
  </div>
</template>

<script>
import bugsMixin from './bugsMixin'
import {randomRange} from '../../js/helpers'

export default {
  name: 'tunnelBug',
  mixins: [bugsMixin],
  data() {
    return {
      directionAngle: 0
    }
  },
  computed: {
    initialYs: function () {
      return [this.canvas.height / 2]
    }
  },
  methods: {
    initiateStartPosition() {
      this.x = 0
      this.y = this.initialYs[randomRange(0, this.initialYs.length)]
    },
    straightMove(noiseWeight = null) {
      let xNoise = this.y > this.canvas.height / 2 ? 0 : 0.2 * this.getRandomNoise()
      let speedWeight = this.y < this.canvas.height / 2 ? 0.4 : 1
      this.dx = (this.vx * speedWeight) + xNoise
      this.dy = (this.vy * speedWeight)
      this.x += this.dx
      this.y += this.dy
    }
  }
}
</script>

<style scoped>

</style>