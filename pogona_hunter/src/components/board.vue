<template>
  <div class="board-canvas-wrapper" oncontextmenu="return false;" v-on:mousedown="logTouch">
    <div id="bugs-board" v-if="!isMedia">
      <audio ref="audio1"><source src="@/assets/sounds/2.mp3" type="audio/mpeg"></audio>
      <p style="float: right">SCORE: {{ $store.state.score }}</p>
      <slide-menu v-on:init="initBoard"
                  v-bind:bugSettings="bugsSettings"
                  v-bind:configOptions="configOptions"
                  v-bind:canvasParams="canvasParams">
      </slide-menu>
      <canvas id="holesCanvas" v-bind:style="{background: bugsSettings.backgroundColor}"
              v-bind:height="canvasParams.height" v-bind:width="canvasParams.width"></canvas>
      <canvas id="bugCanvas" v-bind:height="canvasParams.height" v-bind:width="canvasParams.width"
              v-on:mousedown="setCanvasClick($event)">
        <hole-bugs v-for="(value, index) in bugsProps"
                   :key="index"
                   :bugsSettings="bugsSettings"
                   :exit-hole-pos="exitHolePos"
                   :entrance-hole-pos="entranceHolePos"
              ref="bugChild"></hole-bugs>
<!--        <bug v-for="(value, index) in bugsProps"-->
<!--             :key="index"-->
<!--             :x0="value.x"-->
<!--             :y0="value.y"-->
<!--             :bugsSettings="bugsSettings"-->
<!--             ref="bugChild">-->
<!--        </bug>-->
      </canvas>
    </div>
    <media v-if="isMedia" :url="mediaUrl" ref="mediaElement"></media>
  </div>
</template>

<script>
import bug from './bug'
import holeBugs from './holeBugs'
import {distance, randomRange} from '@/js/helpers'
import {handlePrediction, showPogona} from '../js/predictions'
import SlideMenu from './slideMenu'
import media from './media'

export default {
  name: 'board',
  components: {SlideMenu, bug, media, holeBugs},
  data() {
    return {
      configOptions: require('@/config.json'),
      bugsProps: [],
      bugsSettings: {
        numOfBugs: 0,
        numTrials: 2, // null = endless trials
        trialDuration: 10,
        iti: 5,
        bugTypes: ['cockroach'],
        rewardBugs: 'cockroach',
        movementType: 'circle',
        speed: 0, // if 0 config default for bug will be used
        bugSize: 0, // if 0 config default for bug will be used
        bloodDuration: 2000,
        backgroundColor: '#e8eaf6',
        holeSize: [200, 200],
        exitHole: 'bottomRight',
        entranceHole: null
        // timeInEdge: 2000,
        // isStopOnReward: false,
        // isAntiClockWise: false,
        // targetDrift: 'leftBottom'
        // bugHeight: 100, // relevant only for horizontal movements
      },
      mediaUrl: '',
      holeImgSrc: '',
      trial_id: 1,
      pad: 100, // padding for holes
      isMedia: false,
      isHandlingTouch: false,
      trajectoryLog: [],
      touchesCounter: 0,
      canvasParams: {
        width: window.innerWidth,
        height: window.innerHeight
        // height: Math.round(window.innerHeight / 1.5)
      }
    }
  },
  mounted() {
    this.$mqtt.subscribe('event/log/prediction')
    this.$mqtt.subscribe('event/command/+')
    this.canvas = document.getElementById('bugCanvas')
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
      location.reload()
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
    },
    'event/command/reload_app'(options) {
      location.reload()
    }
  },
  computed: {
    currentBugOptions: function () {
      let bugType = Array.isArray(this.bugsSettings.bugTypes) ? this.bugsSettings.bugTypes[0] : this.bugsSettings.bugTypes
      return this.configOptions.bugTypes[bugType]
    },
    holesPositions: function() {
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
    initBoard(isLogTrajectory = false) {
      if (this.animationHandler) {
        this.$refs.bugChild = []
        cancelAnimationFrame(this.animationHandler)
      }
      if (isLogTrajectory) {
        this.startLogTrajectory()
      }
      this.drawHoles()
      this.spawnBugs(this.bugsSettings.numOfBugs)
      this.$nextTick(function () {
        console.log('start animation...')
        this.animate()
      })
    },
    drawHoles() {
      let image = new Image()
      let canvas = document.getElementById('holesCanvas')
      let ctx = canvas.getContext('2d')
      let [holeW, holeH] = this.bugsSettings.holeSize
      let that = this
      image.onload = function () {
        ctx.drawImage(image, that.exitHolePos[0], that.exitHolePos[1], holeW, holeH)
        ctx.drawImage(image, that.entranceHolePos[0], that.entranceHolePos[1], holeW, holeH)
      }
      image.src = require('@/assets/hole2.png')
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
      this.trial_id = 1
      this.$nextTick(function () {
        console.log('Clear board')
        this.animate()
      })
    },
    animate() {
      if (!this.$refs.bugChild) {
          return
      }
      try {
        this.animationHandler = requestAnimationFrame(this.animate)
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height)
        this.$refs.bugChild.forEach(bug => bug.move(this.$refs.bugChild))
      } catch (e) {
        console.log(e)
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
        if (bug.isDead) {
          continue
        }
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
          bug_type: bug.currentBugType,
          bug_size: bug.currentBugSize
        }))
      }
      this.isHandlingTouch = false
    },
    destruct(bugIndex, x, y, isRewardBug) {
      this.$refs.bugChild[bugIndex].isDead = true
      if (isRewardBug) {
        this.$refs.audio1.play()
        this.$store.commit('increment')
      }
      const bloodTimeout = setTimeout(() => {
        this.$refs.bugChild = this.$refs.bugChild.filter((items, index) => bugIndex !== index)
        if (this.$refs.bugChild.length === 0) {
          this.trial_id++
          if (this.bugsSettings.numTrials && this.trial_id > this.bugsSettings.numTrials) {
            this.$mqtt.publish('event/command/end_app_wait', '')
            this.clearBoard()
          } else {
            const startNewGameTimeout = setTimeout(() => {
              this.initBoard()
              clearTimeout(startNewGameTimeout)
            }, this.bugsSettings.iti)
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
        if (bug) {
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
    },
    logTouch(event) {
      if (this.touchesCounter === 0) {
        let that = this
        let t = setTimeout(() => {
          that.touchesCounter = 0
          clearTimeout(t)
        }, 4000)
      }
      this.touchesCounter++
      if (this.touchesCounter > 5) {
        console.log('climbing!')
      }
    }
  }
}
</script>

<style scoped>

#bugCanvas {
  padding: 0;
  z-index: 100;
  /*margin: 20px auto 0;*/
  display: block;
  position: absolute;
  bottom: 0;
  top: auto;
}

#holesCanvas {
  padding: 0;
  /*margin: 20px auto 0;*/
  display: block;
  position: absolute;
  bottom: 0;
  top: auto;
}

</style>
