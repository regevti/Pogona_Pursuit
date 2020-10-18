<template>
  <div class="board-canvas-wrapper" oncontextmenu="return false;">
    <p style="float: right">SCORE: {{ $store.state.score }}</p>
    <Slide style="z-index: 20;">
      <div>
        <form id='game-configuration' v-on:change="initBoard">
          <h1>Pogona Hunter</h1>
          <div class="row">
            <label for="bugType">Bug Type:</label>
            <select id="bugType" v-model="bugTypes">
              <option v-for="option in Object.keys(bugTypeOptions)" v-bind:value="option"
                      v-bind:key="option">
                {{ bugTypeOptions[option].text }}
              </option>
            </select>
          </div>
          <div class="row">
            <label for="movementType">Movement Type:</label>
            <select id="movementType" v-model="movementType">
              <option v-for="option in movementTypeOptions" v-bind:value="option"
                      v-bind:key="option">
                {{ option }}
              </option>
            </select>
          </div>
          <div class="row">
            <label for="numOfBugs">Number of Bugs: </label>
            <input v-model.number="numOfBugs" id="numOfBugs" type="number" style="width: 2em">
          </div>
          <div class="row">
            <label for="canvasHeight">Canvas Height: </label>
            <input v-model.number="canvasParams.height" id="canvasHeight" type="number" style="width: 4em">
          </div>
          <div class="row">
            <label for="time-in-edge">Time In Edge: </label>
            <input v-model.number="timeInEdge" id="time-in-edge" type="number" style="width: 4em">
          </div>
          <div class="row">
            <label for="time-between-bugs">Time Between Bugs: </label>
            <input v-model.number="timeBetweenBugs" id="time-between-bugs" type="number"
                   style="width: 4em">
          </div>
          <div class="row">
            <label for="blood-duration">Blood Duration: </label>
            <input v-model.number="bloodDuration" id="blood-duration" type="number" style="width: 4em">
          </div>
          <div class="row">
            <label for="radius-min">Radius Range: </label>
            <input v-model.number="currentBugOptions.radiusRange.min" id="radius-min" type="number"
                   style="width: 3em">
            <input v-model.number="currentBugOptions.radiusRange.max" id="radius-max" type="number"
                   style="width: 3em">
          </div>
          <div class="row">
            <label for="speed">Speed: </label>
            <input v-model.number="currentBugOptions.speed" id="speed" type="number"
                   style="width: 3em">
          </div>
          <p>Written by Reggev Eyal</p>
        </form>
      </div>
    </Slide>
    <!--        <canvas id="canvas" v-bind:width="canvasParams.width" v-bind:height="canvasParams.height"-->
    <!--                v-on:touchstart="setCanvasTouch($event)" style="z-index: 10;">-->
    <canvas id="canvas" v-bind:height="canvasParams.height" v-bind:width="canvasParams.width"
            v-on:mousedown="setCanvasClick($event)" style="z-index: 10;"
            v-on:click.right="changeTrajectory($event)">
      <bug v-for="(value, index) in bugsProps"
           :key="index"
           :x0="value.x"
           :y0="value.y"
           :radius="value.radius"
           :bugTypes="bugTypes"
           :timeInEdge="timeInEdge"
           :speed="currentBugOptions.speed"
           :movementType="movementType"
           ref="bugChild">
      </bug>
    </canvas>
  </div>
</template>

<script>
import bug from './bug'
import {distance, randomRange} from '@/js/helpers'
import {Slide} from 'vue-burger-menu'

export default {
  name: 'board',
  components: {bug, Slide},
  data() {
    return {
      bugsProps: [],
      bugTypeOptions: require('@/config.json')['bugTypes'],
      bugTypes: 'cockroach',
      rewardBugs: ['cockroach'],
      movementTypeOptions: require('@/config.json')['movementTypes'],
      movementType: 'line',
      numOfBugs: 0,
      isStopOnReward: false,
      isHandlingTouch: false,
      timeBetweenBugs: 2000,
      bloodDuration: 2000,
      timeInEdge: 2000,
      trajectoryLog: [],
      canvasParams: {
        width: window.innerWidth - 20,
        height: Math.round(window.innerHeight / 1.5)
      }
    }
  },
  mounted() {
    this.$mqtt.subscribe('event/log/prediction')
    this.$mqtt.subscribe('event/command/+')
    this.canvas = document.getElementById('canvas')
    this.ctx = this.canvas.getContext('2d')
    this.initBoard()
    window.addEventListener('keypress', e => {
      this.changeTrajectory(e.code)
    })
  },
  mqtt: {
    'event/command/hide_bugs'(data) {
      this.clearBoard()
    },
    'event/command/init_bugs'(options) {
      options = JSON.parse(options)
      console.log(options)
      this.numOfBugs = Number(options.numOfBugs) ? Number(options.numOfBugs) : 1
      this.bugTypes = options.bugTypes ? options.bugTypes : this.bugTypes
      this.rewardBugs = options.rewardBugs ? options.rewardBugs : this.bugTypes
      this.isStopOnReward = options.isStopOnReward ? Boolean(options.isStopOnReward) : false
      this.movementType = options.movementType ? options.movementType : this.movementType
      this.timeBetweenBugs = options.timeBetweenBugs !== undefined ? Number(options.timeBetweenBugs) * 1000 : this.timeBetweenBugs
      this.currentBugOptions.speed = options.speed ? Number(options.speed) : this.currentBugOptions.speed
      let isLogTrajectory = options.isLogTrajectory ? Boolean(options.isLogTrajectory) : false
      this.$store.commit('reset_score')
      this.initBoard(isLogTrajectory)
    },
    'event/command/show_pogona'(numFrames) {
      numFrames = Number(numFrames)
      const image = new Image(60, 45)
      image.onload = drawImageActualSize
      image.src = '/static/pogona0.jpg'
      let that = this

      function drawImageActualSize() {
        let canvasOriginalHeight = that.canvas.height
        that.canvas.height = this.naturalHeight
        that.ctx.drawImage(this, that.canvas.width / 4, 0)
        if (numFrames > 0) {
          let t = setTimeout(() => {
            that.ctx.clearRect(0, 0, that.canvas.width, that.canvas.height)
            that.canvas.height = canvasOriginalHeight
            clearTimeout(t)
          }, 1000 * numFrames / 60) // for 2 frames
        }
      }
    },
    'event/log/prediction'(options) {
      options = JSON.parse(options)
      console.log(`Prediction detected coords: ${options.hit_point}, time2hit:${options.time2hit}`)
      this.ctx.fillRect(this.canvasParams.width / 2, this.canvasParams.height / 2, 300, 200)
      let t = setTimeout(() => {
        this.ctx.clearRect(0, 0, this.canvasParams.width, this.canvasParams.height)
        clearTimeout(t)
      }, 500)
    }
  },
  computed: {
    currentBugOptions: function () {
      let bugType = Array.isArray(this.bugTypes) ? this.bugTypes[0] : this.bugTypes
      return this.bugTypeOptions[bugType]
    }
  },
  methods: {
    initBoard(isLogTrajectory = false) {
      if (this.animationHandler) {
        this.$refs.bugChild = []
        cancelAnimationFrame(this.animationHandler)
      }
      if (isLogTrajectory) {
        this.startLogTrajectory()
      }
      this.spawnBugs(this.numOfBugs)
      this.$nextTick(function () {
        console.log('start animation...')
        this.animate()
      })
    },
    clearBoard() {
      this.numOfBugs = 0
      if (this.animationHandler) {
        this.$refs.bugChild = []
        cancelAnimationFrame(this.animationHandler)
      }
      if (this.trajectoryLogInterval) {
        this.endLogTrajectory()
      }
      this.$nextTick(function () {
        console.log('Clear board')
        this.animate()
      })
    },
    animate() {
      try {
        this.animationHandler = requestAnimationFrame(this.animate)
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height)
        this.$refs.bugChild.forEach(bug => bug.move(this.$refs.bugChild))
      } catch (e) {
        // console.log(e)
        cancelAnimationFrame(this.animationHandler)
      }
    },
    setCanvasTouch(event) {
      for (let touch of event.touches) {
        this.handleTouchEvent(touch.screenX, touch.screenY)
      }
    },
    setCanvasClick(event) {
      this.handleTouchEvent(event.x, event.y)
    },
    handleTouchEvent(x, y) {
      if (this.isHandlingTouch || !this.$refs.bugChild) { return }
      this.isHandlingTouch = true
      x -= this.canvas.offsetLeft
      y -= this.canvas.offsetTop
      console.log(x, y)
      for (let i = 0; i < this.$refs.bugChild.length; i++) {
        let isHit = false
        let bug = this.$refs.bugChild[i]
        let isRewardBug = this.rewardBugs.includes(bug.currentBugType)
        if (distance(x, y, bug.x, bug.y) <= bug.radius / 1.5) {
          this.destruct(i, x, y, isRewardBug)
          isHit = true
        }
        this.$mqtt.publish('event/log/touch', JSON.stringify({
          x: x,
          y: y,
          bug_x: bug.x,
          bug_y: bug.y,
          is_hit: isHit,
          is_reward_bug: isRewardBug,
          bug_type: bug.currentBugType
        }))
      }
      this.isHandlingTouch = false
    },
    destruct(bugIndex, x, y, isRewardBug) {
      if (this.$refs.bugChild[bugIndex].isDead) {
        return
      }
      this.$refs.bugChild[bugIndex].isDead = true
      if (isRewardBug) {
        this.$store.commit('increment')
      }
      const bloodTimeout = setTimeout(() => {
        this.$refs.bugChild = this.$refs.bugChild.filter((items, index) => bugIndex !== index)
        if (this.$refs.bugChild.length === 0) {
          if (this.isStopOnReward && isRewardBug) {
            this.clearBoard()
            this.$mqtt.publish('event/command/end_trial', '')
          } else {
            const startNewGameTimeout = setTimeout(() => {
              this.initBoard()
              clearTimeout(startNewGameTimeout)
            }, this.timeBetweenBugs)
          }
        }
        clearTimeout(bloodTimeout)
      }, this.bloodDuration)
    },
    spawnBugs(noOfBugs) {
      const radius = randomRange(this.currentBugOptions.radiusRange.min, this.currentBugOptions.radiusRange.max)
      for (let i = 0; i < noOfBugs; i++) {
        let x = randomRange(radius, this.canvas.width - radius)
        let y = randomRange(radius, this.canvas.height - radius)
        let properties = {
          x: x,
          y: y,
          radius: radius,
          bugId: `${this.bugTypes}${i}`
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
    },
    startLogTrajectory() {
      console.log('trajectory log started...')
      this.trajectoryLogInterval = setInterval(() => {
        let bug = this.$refs.bugChild[0]
        if (bug && bug.isInsideBoard()) {
          this.trajectoryLog.push({
            time: Date.now(),
            x: bug.x,
            y: bug.y,
            bug_type: bug.currentBugType
          })
        }
      }, 1000 / 60)
    },
    endLogTrajectory() {
      clearInterval(this.trajectoryLogInterval)
      this.trajectoryLogInterval = null
      this.$mqtt.publish('event/log/trajectory', JSON.stringify(this.trajectoryLog))
      console.log('sent trajectory through mqtt...')
      this.trajectoryLog = []
    }
  }
}
</script>s

<style scoped>

canvas {
  padding: 0;
  /*margin: 20px auto 0;*/
  display: block;
  background: #e8eaf6;
  position: absolute;
  bottom: 10px;
}

</style>
