<template>
  <div>
    <img ref="bugImg" :src="bugImgSrc" alt=""/>
  </div>
</template>

<script>
import {distance, randBM, randomRange} from '@/js/helpers'

export default {
  name: 'holeBugs',
  data() {
    return {
      bugTypeOptions: require('@/config.json')['bugTypes'],
      targetDriftsOptions: require('@/config.json')['targetDrifts'],
      bugImages: [],
      currentBugType: undefined,
      currentBugSize: undefined,
      bugImgSrc: '',
      holeImgSrc: '',
      randomNoise: 0,
      frameCounter: 0,
      framesUntilExitFromEntranceHole: 100,
      minDistFromObstacle: 300,
      bordersAngles: {
        top: 3 * Math.PI / 2,
        bottom: Math.PI / 2,
        left: Math.PI,
        right: 0
      }
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
    stepsPerImage: function () {
      return this.bugTypeOptions[this.currentBugType].stepsPerImage
    },
    currentSpeed: function () {
      if (this.bugsSettings && this.bugsSettings.speed) {
        return this.bugsSettings.speed
      }
      return this.bugTypeOptions[this.currentBugType].speed
    },
    currentAngle: function () {
      return Math.atan2(this.vx, this.vy)
    },
    numImagesPerBug: function () {
      return this.bugTypeOptions[this.currentBugType].numImagesPerBug
    },
    numFramesToRetreat: function () {
      return (this.bugsSettings.trialDuration || 1) * 60
    },
    timeBetweenBugs: function () {
      return (this.bugsSettings.iti || 2) * 1000
    },
    isMoveInCircles: function () {
      return this.bugsSettings.movementType === 'circle'
    },
    isRandomMovement: function () {
      return this.bugsSettings.movementType === 'random'
    }
  },
  mounted() {
    this.canvas = this.$parent.canvas
    this.ctx = this.canvas.getContext('2d')
    if (!Array.isArray(this.bugsSettings.bugTypes)) {
      this.bugsSettings.bugTypes = [this.bugsSettings.bugTypes]
    }
    this.initBug()
  },
  methods: {
    initBug() {
      this.loadNextBugType()
      this.randomNoiseCount = 0
      this.frameCounter = 0
      this.step = 0
      this.isDead = false
      this.isRetreated = false
      this.isHoleRetreatStarted = false
      this.isChangingDirection = false
      this.currentBugSize = this.getRadiusSize()
      this.x = this.entranceHolePos[0] + (this.bugsSettings.holeSize[0] / 2)
      this.y = this.entranceHolePos[1] + (this.bugsSettings.holeSize[1] / 2)
      this.xTarget = this.exitHolePos[0] + (this.bugsSettings.holeSize[0] / 2)
      this.yTarget = this.exitHolePos[1] + (this.bugsSettings.holeSize[1] / 2)
      switch (this.bugsSettings.movementType) {
        case 'circle':
          this.theta = Math.PI + (Math.PI / 4) // - Math.PI / 20 // used for circular movement
          this.r = (Math.abs(this.xTarget - this.x) / 2)
          this.r0 = [(this.x + this.xTarget) / 2, this.y + (this.r / 2.3)]
          break
        case 'low_horizontal':
          this.startRetreat()
          break
        default:
          this.setNextAngle()
      }
    },
    move() {
      if (this.isDead || this.isRetreated) {
        this.draw()
        return
      }
      this.frameCounter++
      this.edgeDetection()
      this.checkHoleRetreat()
      switch (this.bugsSettings.movementType) {
      case 'circle':
        this.theta += Math.abs(this.currentSpeed) * Math.sqrt(2) / this.r
        this.x = this.r0[0] + (this.r * Math.cos(this.theta)) * (this.bugsSettings.isAntiClockWise ? -1 : 1)
        this.y = this.r0[1] + this.r * Math.sin(this.theta)
        break
      default:
        this.straightMove()
      }
      this.draw()
    },
    straightMove() {
      let randNoise = this.getRandomNoise()
      this.dx = this.vx + 0.5 * randNoise
      this.dy = this.vy + 0.5 * randNoise
      this.x += this.dx
      this.y += this.dy
    },
    edgeDetection() {
      if (this.isChangingDirection) {
        return
      }
      // borders
      let radius = this.currentBugSize / 2
      if (this.x < radius || this.x > this.canvas.width - radius ||
          this.y < radius || this.y > this.canvas.height - radius) {
        this.setNextAngle()
      // holes edges
      } else if (this.frameCounter > 100 && this.isInsideHoleBoundaries()) {
        if ((this.isHoleRetreatStarted && this.isInsideExitHoleBoundaries()) || !this.isRandomMovement) {
          this.hideBug()
        } else {
          this.setNextAngle()
        }
      } else {
        return
      }
      this.changeDirectionTimeout()
    },
    draw() {
      let imgIndex = Math.floor(this.step / this.stepsPerImage)
      this.bugImgSrc = this.getImageSrc(`/${this.currentBugType}${imgIndex}.png`)
      this.drawBug()
      this.step++
      if (this.step > (this.numImagesPerBug - 1) * this.stepsPerImage) {
        this.step = 0
      }
    },
    drawBug() {
      if (this.isRetreated) {
        return
      }
      try {
        let bugImage = this.isDead ? this.getDeadImage() : this.$refs.bugImg
        this.ctx.setTransform(1, 0, 0, 1, this.x, this.y)
        this.ctx.rotate(this.getAngleRadians())
        // drawImage(image, dx, dy, dWidth, dHeight)
        this.ctx.drawImage(bugImage, -this.currentBugSize / 2, -this.currentBugSize / 2, this.currentBugSize, this.currentBugSize)
        this.ctx.setTransform(1, 0, 0, 1, 0, 0)
      } catch (e) {
        console.error(e)
      }
    },
    getNotBlockedAngles() {
      let angles = []
      let borderDistances = {
        top: this.y,
        bottom: this.canvas.height - this.y,
        left: this.x,
        right: this.canvas.width - this.x
      }
      for (const [key, angle] of Object.entries(this.bordersAngles)) {
        if (borderDistances[key] > this.minDistFromObstacle) {
          // push to angles array the radian angles of the walls that are far, so these can be used
          // to determine the range of next possible angles
          angles.push(angle)
        }
      }
      return angles
    },
    setNextAngle() {
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
      let nextAngle = Math.random() * (openAngles[openAngles.length - 1] - openAngles[0]) + openAngles[0]
      this.vx = this.currentSpeed * Math.cos(nextAngle)
      this.vy = this.currentSpeed * Math.sin(nextAngle)
    },
    hideBug() {
      // hide bug when exit hole reached
      let fadeTimeout = setTimeout(() => {
        this.isRetreated = true
        this.$emit('bugRetreated')
        let initTimeout = setTimeout(() => {
          this.initBug()
          clearTimeout(initTimeout)
        }, this.timeBetweenBugs)
        clearTimeout(fadeTimeout)
      }, 100)
    },
    changeDirectionTimeout() {
      this.isChangingDirection = true
      let t = setTimeout(() => {
        this.isChangingDirection = false
        clearTimeout(t)
      }, 300)
    },
    isHit(x, y) {
      return distance(x, y, this.x, this.y) <= this.currentBugSize / 1.5
    },
    rotate(dx, dy, angle) {
      return {
        dx: dx * Math.cos(angle) - dy * Math.sin(angle),
        dy: dx * Math.sin(angle) + dy * Math.cos(angle)
      }
    },
    getAngleRadians() {
      if (this.isMoveInCircles) {
        return Math.atan2(this.y - this.r0[1], this.x - this.r0[0]) + (this.bugsSettings.isAntiClockWise ? 0 : Math.PI)
      }
      return Math.atan2(this.dy, this.dx) + Math.PI / 2
    },
    getRandomNoise() {
      if (this.randomNoiseCount > 20) {
        this.randomNoiseCount = 0
        this.randomNoise = randBM()
      }
      this.randomNoiseCount++
      return this.randomNoise
    },
    loadNextBugType() {
      if (this.bugsSettings.bugTypes.length === 1) {
        this.currentBugType = this.bugsSettings.bugTypes[0]
        return
      }
      let nextBugOptions = this.bugsSettings.bugTypes.filter(bug => bug !== this.currentBugType)
      let nextIndex = randomRange(0, nextBugOptions.length)
      this.currentBugType = nextBugOptions[nextIndex]
    },
    getRadiusSize() {
      if (this.bugsSettings.bugSize) {
        return this.bugsSettings.bugSize
      }
      let currentBugOptions = this.bugTypeOptions[this.currentBugType]
      return randomRange(currentBugOptions.radiusRange.min, currentBugOptions.radiusRange.max)
    },
    getImageSrc(fileName) {
      return require('@/assets' + fileName)
    },
    getDeadImage() {
      let img = new Image()
      img.src = this.getImageSrc(`/${this.currentBugType}_dead.png`)
      return img
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
        let xd = this.xTarget - this.x
        let yd = this.yTarget - this.y
        let T = yd / xd
        this.vx = Math.sign(xd) * (this.currentSpeed / Math.sqrt(1 + T ** 2))
        this.vy = Math.sign(yd) * Math.sqrt((this.currentSpeed ** 2) - (this.vx ** 2))
        this.isHoleRetreatStarted = true
      }
    }
  }
}
</script>

<style scoped>

</style>
