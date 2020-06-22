<template>
    <div class="board-canvas-wrapper">
        <div v-on:change="initBoard">
            <label for="bugType">Bug Type:</label>
            <select id="bugType" v-model="bugType">
                <option v-for="option in bugTypeOptions" v-bind:value="option.value" v-bind:key="option.value">
                    {{ option.text }}
                </option>
            </select>
            <label for="numOfBugs">Number of Bugs: </label>
            <input v-model.number="numOfBugs" id="numOfBugs" type="number" style="width: 2em">
        </div>
        <canvas id="canvas" v-bind:width="canvasParams.width" v-bind:height="canvasParams.height"
                v-on:click="setCanvasClick($event)"
                v-on:click.right="changeTrajectory($event)" oncontextmenu="return false;">
            <bug v-for="(value, index) in bugsProps"
                 :key="index"
                 :x0="value.x"
                 :y0="value.y"
                 :radius="value.radius"
                 :bugType="value.bugType"
                 ref="bugChild">
            </bug>
        </canvas>
        <h3 style="margin: 10px auto 0">SCORE: {{$store.state.score}}</h3>
    </div>
</template>

<script>
  import bug from './bug'
  import {distance, randomRange} from '../js/helpers'
  import {mapState} from 'vuex'

  export default {
    name: 'board',
    components: {bug},
    data() {
      return {
        bugsProps: [],
        bugType: 'cockroach',
        numOfBugs: 1
      }
    },
    mounted() {
      this.canvas = document.getElementById('canvas')
      this.ctx = this.canvas.getContext('2d')
      this.initBoard()
      window.addEventListener('keypress', e => {
        this.changeTrajectory(e.code)
      })
    },
    computed: {
      ...mapState(['timeBetweenTrial', 'bloodDuration', 'canvasParams', 'radiusRange', 'bugTypeOptions'])
    },
    methods: {
      initBoard() {
        if (this.animationHandler) {
          this.$refs.bugChild = []
          cancelAnimationFrame(this.animationHandler)
        }
        this.spawnBugs(this.numOfBugs)
        this.$nextTick(function () {
          console.log('start animation...')
          this.animate()
        })
      },
      animate() {
        try {
          this.animationHandler = requestAnimationFrame(this.animate)
          this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height)
          this.$refs.bugChild.forEach(bug => bug.move(this.$refs.bugChild))
        } catch (e) {
          console.log(e)
          cancelAnimationFrame(this.animationHandler)
        }
      },
      setCanvasClick(event) {
        let x = event.x
        let y = event.y
        x -= this.canvas.offsetLeft
        y -= this.canvas.offsetTop
        console.log(x, y)

        for (let i = 0; i < this.$refs.bugChild.length; i++) {
          if (distance(x, y, this.$refs.bugChild[i].x, this.$refs.bugChild[i].y) <= this.$refs.bugChild[i].radius / 2) {
            this.destruct(i)
          }
        }
      },
      destruct(bugIndex) {
        if (this.$refs.bugChild[bugIndex].isDead) {
          return
        }
        this.$refs.bugChild[bugIndex].isDead = true
        this.$store.commit('increment')
        const bloodTimeout = setTimeout(() => {
          this.$refs.bugChild = this.$refs.bugChild.filter((items, index) => bugIndex !== index)
          if (this.$refs.bugChild.length === 0) {
            const startNewGameTimeout = setTimeout(() => {
              cancelAnimationFrame(this.animationHandler)
              this.initBoard()
              clearTimeout(startNewGameTimeout)
            }, this.timeBetweenTrial)
          }
          clearTimeout(bloodTimeout)
        }, this.bloodDuration)
      },
      spawnBugs(noOfBugs) {
        const radius = randomRange(this.radiusRange.min, this.radiusRange.max)
        for (let i = 0; i < noOfBugs; i++) {
          let x = randomRange(radius, this.canvas.width - radius)
          let y = randomRange(radius, this.canvas.height - radius)
          let properties = {
            x: x,
            y: y,
            radius: radius,
            bugType: this.bugType,
            bugId: `${this.bugType}${i}`
          }

          if (i !== 0) {
            for (let j = 0; j < i; j++) {
              let d = distance(x, y, this.bugsProps[j].x, this.bugsProps[j].y)
              if (d <= radius + this.bugsProps[j].radius) {
                x = randomRange(radius, this.canvas.width - radius)
                y = randomRange(radius, this.canvas.height - radius)
                j = -1
              }
            }
          }
          this.bugsProps.push(properties)
        }
      },
      changeTrajectory(event) {
        console.log('rightClick', event.x, event.y)
        this.$refs.bugChild.forEach(bug => bug.escape(event.x, event.y))
      }
    }
  }
</script>

<style scoped>

    canvas {
        padding: 0;
        margin: 20px auto 0;
        display: block;
        background: #e8eaf6;
    }

</style>
