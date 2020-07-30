<template>
    <div class="board-canvas-wrapper" oncontextmenu="return false;">
        <Slide style="z-index: 20;">
            <div>
                <form id='game-configuration' v-on:change="initBoard">
                    <h1>Pogona Hunter</h1>
                    <div class="row">
                        <label for="bugType">Bug Type:</label>
                        <select id="bugType" v-model="bugType">
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
                        <label for="time-between-trial">Time Between Trials: </label>
                        <input v-model.number="timeBetweenTrial" id="time-between-trial" type="number"
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
                    <h3 style="margin-top: 3em">SCORE: {{$store.state.score}}</h3>
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
                 :bugType="bugType"
                 :timeInEdge="timeInEdge"
                 :speed="currentBugOptions.speed"
                 :numImagesPerBug="currentBugOptions.numImagesPerBug"
                 :isStatic="currentBugOptions.isStatic"
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
        bugTypeOptions: require('@/store/bugs.json'),
        bugType: 'cockroach',
        movementTypeOptions: ['line', 'circle'],
        movementType: 'line',
        numOfBugs: 1,
        timeBetweenTrial: 2000,
        bloodDuration: 2000,
        timeInEdge: 2000,
        canvasParams: {
          width: window.innerWidth - 20,
          height: Math.round(window.innerHeight / 1.5)
        }
      }
    },
    mounted() {
      this.$mqtt.subscribe('event/detect/+')
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
        this.numOfBugs = 0
        this.initBoard()
      },
      'event/command/init_bugs'(numOfBugs) {
        numOfBugs = Number(numOfBugs) ? Number(numOfBugs) : 1
        this.numOfBugs = numOfBugs
        this.initBoard()
      },
      'event/command/bug_speed'(speed) {
        this.speed = Number(speed)
        this.initBoard()
      },
      'event/command/bug_type'(bugType) {
        if (Object.values(this.bugTypeOptions).indexOf(bugType) !== -1) {
          this.bugType = bugType
          this.initBoard()
        }
      }
    },
    computed: {
      currentBugOptions: function () {
        return this.bugTypeOptions[this.bugType]
      }
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
      setCanvasTouch(event) {
        for (let touch of event.touches) {
          this.handleTouchEvent(touch.screenX, touch.screenY)
        }
      },
      setCanvasClick(event) {
        this.handleTouchEvent(event.x, event.y)
      },
      handleTouchEvent(x, y) {
        x -= this.canvas.offsetLeft
        y -= this.canvas.offsetTop
        console.log(x, y)

        for (let i = 0; i < this.$refs.bugChild.length; i++) {
          this.$mqtt.publish('event/log/touch', JSON.stringify({
            x: x,
            y: y,
            bug_x: this.$refs.bugChild[i].x,
            bug_y: this.$refs.bugChild[i].y
          }))
          if (distance(x, y, this.$refs.bugChild[i].x, this.$refs.bugChild[i].y) <= this.$refs.bugChild[i].radius / 1.5) {
            this.destruct(i, x, y)
          }
        }
      },
      destruct(bugIndex, x, y) {
        if (this.$refs.bugChild[bugIndex].isDead) {
          return
        }
        this.$refs.bugChild[bugIndex].isDead = true
        this.$store.commit('increment')
        this.$mqtt.publish('event/log/hit', JSON.stringify({x: x, y: y}))
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
        const radius = randomRange(this.currentBugOptions.radiusRange.min, this.currentBugOptions.radiusRange.max)
        for (let i = 0; i < noOfBugs; i++) {
          let x = randomRange(radius, this.canvas.width - radius)
          let y = randomRange(radius, this.canvas.height - radius)
          let properties = {
            x: x,
            y: y,
            radius: radius,
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
        /*margin: 20px auto 0;*/
        display: block;
        background: #e8eaf6;
        position: absolute;
        bottom: 10px;
    }

</style>
