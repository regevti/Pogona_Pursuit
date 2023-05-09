<template>
  <div>
    <img ref="bugImg" :src="bugImgSrc" alt=""/>
  </div>
</template>

<script>
import {randomRange} from '@/js/helpers'
import bugsMixin from './bugsMixin'

export default {
  name: 'holeBugs',
  mixins: [bugsMixin],
  data() {
    return {
      edgesPolicy: 'inside',
      framesUntilExitFromEntranceHole: 100
    }
  },
  props: {
    bugsSettings: Object,
    exitHolePos: Array,
    entranceHolePos: Array
  },
  computed: {
    holeSize: function () {
      return this.bugsSettings.holeSize
    },
    currentSpeed: function () {
      if (this.bugsSettings && this.bugsSettings.speed) {
        return this.bugsSettings.speed
      }
      return this.bugTypeOptions[this.currentBugType].speed
    },
    jump_distance: function () {
      return this.currentBugSize * 1.5
    },
    upper_edge: function () {
      let edge = this.currentBugSize / 2
      if (this.isJumpUpMovement) {
        edge += this.jump_distance
      }
      return edge
    },
    numFramesToRetreat: function () {
      return (this.bugsSettings.trialDuration || 1) * 60
    },
    isRightExit: function () {
      return this.bugsSettings.exitHole === 'bottomRight'
    },
    isLeftExit: function () {
      return this.bugsSettings.exitHole === 'bottomLeft'
    },
    isMoveInCircles: function () {
      return this.bugsSettings.movementType === 'circle'
    },
    isHalfCircleMovement: function () {
      return this.bugsSettings.movementType === 'half_circle'
    },
    isRandomMovement: function () {
      return this.bugsSettings.movementType === 'random'
    },
    isLowHorizontalMovement: function () {
      return this.bugsSettings.movementType === 'low_horizontal'
    },
    isJumpUpMovement: function () {
      return this.bugsSettings.movementType === 'jump_up'
    },
    isNoisyLowHorizontalMovement: function () {
      return this.bugsSettings.movementType === 'low_horizontal_noise'
    },
    isRandomSpeeds: function () {
      return this.bugsSettings.movementType === 'random_speeds'
    },
    isCounterClockWise: function () {
      return this.isLeftExit
    }
  },
  methods: {
    move() {
      if (this.isDead || this.isRetreated) {
        this.draw()
        return
      }
      this.frameCounter++
      this.edgeDetection()
      this.checkHoleRetreat()
      // half-circle
      if (this.isHalfCircleMovement) {
        this.circularMove()
      // circle
      } else if (this.isMoveInCircles && !this.isHoleRetreatStarted) {
        this.circularMove()
      // low horizontal noise
      } else if (this.isNoisyLowHorizontalMovement) {
        this.checkNoisyTrack()
        if (this.isNoisyPartReached) {
          this.noisyMove()
        } else {
          this.straightMove(0)
        }
      // low horizontal
      } else if (this.isLowHorizontalMovement || this.isJumpUpMovement) {
        this.straightMove(0)
      // random
      } else {
        this.straightMove()
      }
      this.draw()
    },
    edgeDetection() {
      if (this.isChangingDirection) {
        return
      }
      // borders
      let radius = this.currentBugSize / 2
      if (this.x < radius || this.x > this.canvas.width - radius ||
          this.y < this.upper_edge || this.y > this.canvas.height - radius) {
        this.setNextAngle()
      // holes edges
      } else if (this.frameCounter > 100 && this.isInsideHoleBoundaries()) {
        if ((this.isHoleRetreatStarted && this.isInsideExitHoleBoundaries()) || !(this.isRandomMovement || this.isRandomSpeeds)) {
          this.hideBug()
        } else {
          this.setNextAngle()
        }
      } else {
        return
      }
      this.changeDirectionTimeout()
    },
    setNextAngle(angle = null) {
      if (this.isNoisyLowHorizontalMovement && this.isNoisyPartReached) {
        return
      }
      let nextAngle = angle
      if (!angle) {
        let openAngles = this.getNotBlockedAngles()
        openAngles = openAngles.sort()
        for (let i = 0; i < openAngles.length - 1; i++) {
          // in order to maintain the continuity in angles range, in cases of missing angles add 2π to the angles
          // right before the missing ones.
          if ((openAngles[i + 1] - openAngles[i]) > (Math.PI / 2)) {
            openAngles[i] += 2 * Math.PI
          }
        }
        openAngles = openAngles.sort()
        nextAngle = Math.random() * (openAngles[openAngles.length - 1] - openAngles[0]) + openAngles[0]
      }
      this.vx = this.currentSpeed * Math.cos(nextAngle)
      this.vy = this.currentSpeed * Math.sin(nextAngle)
    },
    initiateStartPosition() {
      this.x = this.entranceHolePos[0] + (this.bugsSettings.holeSize[0] / 2)
      this.y = this.entranceHolePos[1] + (this.bugsSettings.holeSize[1] / 2)
      this.xTarget = this.exitHolePos[0] + (this.bugsSettings.holeSize[0] / 2)
      this.yTarget = this.exitHolePos[1] + (this.bugsSettings.holeSize[1] / 2)
      this.isRetreated = false
      this.isHoleRetreatStarted = false
      this.isCircleTrackReached = true
      this.lowHorizontalNoiseStart = (this.x + this.xTarget) / 2
      this.isNoisyPartReached = false
      this.frameCounter = 0
      switch (this.bugsSettings.movementType) {
        case 'circle':
          this.theta = this.isRightExit ? (Math.PI + (Math.PI / 5)) : (Math.PI + (2 * Math.PI / 3))
          this.r = (Math.abs(this.xTarget - this.x) / 5)
          this.r0 = [(this.x + this.xTarget) / 2, this.y / 2]
          break
        case 'half_circle':
          this.theta = this.isCounterClockWise ? (Math.PI + (Math.PI / 4)) : (Math.PI + (Math.PI / 4))
          this.r = (Math.abs(this.xTarget - this.x) / 2)
          this.r0 = [(this.x + this.xTarget) / 2, this.y + (this.r / 2.3)]
          break
        case 'low_horizontal':
        case 'jump_up':
          this.directionAngle = this.isRightExit ? 2 * Math.PI : Math.PI
          this.startRetreat()
          break
        case 'low_horizontal_noise':
          this.directionAngle = this.isRightExit ? 2 * Math.PI : Math.PI
          this.setRetreatSpeeds()
          break
        case 'random_speeds':
          this.directionAngle = randomRange(3 * Math.PI / 4, 2 * Math.PI)
          this.bugsSettings.speed = randomRange(2, 10)
          this.setNextAngle()
          break
        default:
          this.directionAngle = randomRange(3 * Math.PI / 4, 2 * Math.PI)
          this.setNextAngle()
      }
    },
    circularMove() {
      this.theta += Math.abs(this.currentSpeed) * Math.sqrt(2) / this.r
      this.x = this.r0[0] + (this.r * Math.cos(this.theta)) * (this.isCounterClockWise ? -1 : 1)
      this.y = this.r0[1] + this.r * Math.sin(this.theta)
    },
    noisyMove() {
      let randNoise = this.getRandomNoise()
      this.dx = this.vx + 0.5 * randNoise
      this.dy = 0.00008 * (this.yTarget - this.y) + 0.9 * randNoise + 0.65 * this.dy
      this.x += this.dx
      this.y += this.dy
    },
    jump() {
      if (!this.isJumpUpMovement) {
        return
      }
      this.y -= this.jump_distance
    },
    isInsideHoleBoundaries() {
      return this.isInsideEntranceHoleBoundaries() || this.isInsideExitHoleBoundaries()
    },
    isInsideEntranceHoleBoundaries() {
      return this.entranceHolePos[0] <= this.x && this.x <= (this.entranceHolePos[0] + this.holeSize[0]) &&
          this.entranceHolePos[1] <= this.y && this.y <= (this.entranceHolePos[1] + this.holeSize[1])
    },
    isInsideExitHoleBoundaries() {
      return this.exitHolePos[0] <= this.x && this.x <= (this.exitHolePos[0] + this.holeSize[0]) &&
          this.exitHolePos[1] <= this.y && this.y <= (this.exitHolePos[1] + this.holeSize[1])
    },
    checkHoleRetreat() {
      if (!this.isHoleRetreatStarted && this.frameCounter > this.numFramesToRetreat) {
        this.startRetreat()
      }
    },
    startRetreat() {
      if (!this.isHoleRetreatStarted) {
        this.setRetreatSpeeds()
        this.isHoleRetreatStarted = true
      }
    },
    setRetreatSpeeds() {
      let xd = this.xTarget - this.x
      let yd = this.yTarget - this.y
      let T = yd / xd
      this.vx = Math.sign(xd) * (this.currentSpeed / Math.sqrt(1 + T ** 2))
      this.vy = Math.sign(yd) * Math.sqrt((this.currentSpeed ** 2) - (this.vx ** 2))
    },
    checkNoisyTrack() {
      if (this.isHoleRetreatStarted) {
        return
      }
      if (!this.isNoisyPartReached) {
        if (((this.exitHolePos[0] > this.lowHorizontalNoiseStart) && (this.x > this.lowHorizontalNoiseStart)) ||
        ((this.exitHolePos[0] < this.lowHorizontalNoiseStart) && (this.x < this.lowHorizontalNoiseStart))) {
            this.isNoisyPartReached = true
          }
      }
      if (((this.exitHolePos[0] > this.lowHorizontalNoiseStart) && (this.x > this.exitHolePos[0] - 10)) ||
         ((this.exitHolePos[0] < this.lowHorizontalNoiseStart) && (this.x < this.exitHolePos[0] + 10))) {
        this.isNoisyPartReached = false
        this.startRetreat()
      }
    }
  }
}
</script>

<style scoped>

</style>
