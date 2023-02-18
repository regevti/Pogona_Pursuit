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
    }
  }
}
</script>

<style scoped>

</style>