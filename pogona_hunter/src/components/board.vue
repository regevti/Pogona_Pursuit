<template>
  <div class="board-canvas-wrapper" oncontextmenu="return false;">
    <div id="bugs-board" v-if="!isMedia">
      <audio ref="audio1"><source src="@/assets/sounds/2.mp3" type="audio/mpeg"></audio>
      <p style="float: right">SCORE: {{ $store.state.score }}</p>
      <slide-menu v-on:init="initBoard"
                  v-bind:bugSettings="bugsSettings"
                  v-bind:configOptions="configOptions"
                  v-bind:canvasParams="canvasParams">
      </slide-menu>
      <!--        <canvas id="canvas" v-bind:width="canvasParams.width" v-bind:height="canvasParams.height"-->
      <!--                v-on:touchstart="setCanvasTouch($event)" style="z-index: 10;">-->
      <canvas id="canvas" v-bind:height="canvasParams.height" v-bind:width="canvasParams.width"
              v-on:mousedown="setCanvasClick($event)" style="z-index: 100;"
              v-on:click.right="changeTrajectory($event)">
        <bug v-for="(value, index) in bugsProps"
             :key="index"
             :x0="value.x"
             :y0="value.y"
             :bugsSettings="bugsSettings"
             ref="bugChild">
        </bug>
      </canvas>
    </div>
    <media v-if="isMedia" :url="mediaUrl" ref="mediaElement"></media>
  </div>
</template>

<script>
import bug from './bug'
import {distance, randomRange} from '@/js/helpers'
import {handlePrediction, showPogona} from '../js/predictions'
import SlideMenu from './slideMenu'
import media from './media'

export default {
  name: 'board',
  components: {SlideMenu, bug, media},
  data() {
    return {
      configOptions: require('@/config.json'),
      bugsProps: [],
      bugsSettings: {
        numOfBugs: 0,
        bugTypes: ['cockroach'],
        rewardBugs: 'cockroach',
        movementType: 'line',
        speed: 0, // if 0 config default for bug will be used
        radiusSize: 0, // if 0 config default for bug will be used
        isStopOnReward: false,
        isAntiClockWise: false,
        timeBetweenBugs: 2000,
        bloodDuration: 2000,
        timeInEdge: 2000
      },
      mediaUrl: '',
      isMedia: false,
      isHandlingTouch: false,
      trajectoryLog: [],
      canvasParams: {
        width: window.innerWidth,
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
      Object.assign(this.bugsSettings, options)
      this.$store.commit('reset_score')
      this.initBoard(!!options['isLogTrajectory'])
    },
    'event/command/hide_media'() {
      if (this.isMedia) {
        this.$mqtt.publish('event/log/video_frames', JSON.stringify(this.$refs.mediaElement.framesLog))
        this.isMedia = false
      }
    },
    'event/command/init_media'(options) {
      options = JSON.parse(options)
      this.clearBoard()
      this.mediaUrl = options.url
      console.log(this.mediaUrl)
      this.isMedia = true
    },
    'event/command/show_pogona'(numFrames) {
      showPogona(this.canvas, numFrames)
    },
    'event/log/prediction'(options) {
      handlePrediction(options, this.ctx, this.canvasParams)
    }
  },
  computed: {
    currentBugOptions: function () {
      let bugType = Array.isArray(this.bugsSettings.bugTypes) ? this.bugsSettings.bugTypes[0] : this.bugsSettings.bugTypes
      return this.configOptions.bugTypes[bugType]
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
      this.spawnBugs(this.bugsSettings.numOfBugs)
      this.$nextTick(function () {
        console.log('start animation...')
        this.animate()
      })
    },
    clearBoard() {
      this.bugsSettings.numOfBugs = 0
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
      console.log(x, y)
      if (this.isHandlingTouch || !this.$refs.bugChild) { return }
      this.isHandlingTouch = true
      x -= this.canvas.offsetLeft
      y -= this.canvas.offsetTop
      for (let i = 0; i < this.$refs.bugChild.length; i++) {
        let isHit = false
        let bug = this.$refs.bugChild[i]
        let isRewardBug = this.bugsSettings.rewardBugs.includes(bug.currentBugType)
        if (bug.isHit(x, y)) {
          this.destruct(i, x, y, isRewardBug)
          isHit = true
        }
        this.$mqtt.publish('event/log/touch', JSON.stringify({
          time: Date.now(),
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
        this.$refs.audio1.play()
        this.$store.commit('increment')
      }
      const bloodTimeout = setTimeout(() => {
        this.$refs.bugChild = this.$refs.bugChild.filter((items, index) => bugIndex !== index)
        if (this.$refs.bugChild.length === 0) {
          if (this.bugsSettings.isStopOnReward && isRewardBug) {
            this.clearBoard()
          } else {
            const startNewGameTimeout = setTimeout(() => {
              this.initBoard()
              clearTimeout(startNewGameTimeout)
            }, this.bugsSettings.timeBetweenBugs)
          }
        }
        clearTimeout(bloodTimeout)
      }, this.bugsSettings.bloodDuration)
    },
    spawnBugs(noOfBugs) {
      const minDistance = 100
      for (let i = 0; i < noOfBugs; i++) {
        let x = randomRange(0, this.canvas.width)
        let y = randomRange(0, this.canvas.height)
        let properties = {
          x: x,
          y: y,
          bugId: `${this.bugsSettings.bugTypes}${i}`
        }
        if (i !== 0) {
          for (let j = 0; j < i; j++) {
            let d = distance(x, y, this.bugsProps[j].x, this.bugsProps[j].y)
            if (d <= minDistance) {
              x = randomRange(0, this.canvas.width)
              y = randomRange(0, this.canvas.height)
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
</script>

<style scoped>

canvas {
  padding: 0;
  /*margin: 20px auto 0;*/
  display: block;
  background: #e8eaf6;
  position: absolute;
  bottom: 0;
}

</style>
