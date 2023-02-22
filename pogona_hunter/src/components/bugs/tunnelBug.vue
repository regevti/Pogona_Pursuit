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
  computed: {
    initialXs: function () {
      return [this.canvas.width / 4, 2 * this.canvas.width / 4, 3 * this.canvas.width / 4]
    }
  },
  methods: {
    initiateStartPosition() {
      this.x = this.initialXs[randomRange(0, this.initialXs.length)]
      this.y = this.startPosition[1]
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