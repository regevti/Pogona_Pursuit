import {distance, randBM, randomRange} from '../../js/helpers'

export default {
  data() {
    return {
      y: 0,
      bugTypeOptions: require('../../config.json')['bugTypes'],
      targetDriftsOptions: require('../../config.json')['targetDrifts'],
      bugImages: [],
      currentBugType: undefined,
      currentBugSize: undefined,
      bugImgSrc: '',
      randomNoise: 0,
      directionAngle: Math.PI / 2,
      edgesPolicy: 'outside_end', // options: ['inside', 'outside_back', 'outside_end']
      // inside edgePolicy Parameters:
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
    bugsSettings: Object
  },
  computed: {
    startPosition: function () {
      return [this.canvas.width / 2, this.currentBugSize / 2]
    },
    isInsidePolicy: function () {
      return this.edgesPolicy === 'inside'
    },
    isOutsideEndPolicy: function () {
      return this.edgesPolicy === 'outside_end'
    },
    isOutsideBackPolicy: function () {
      return this.edgesPolicy === 'outside_back'
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
    numImagesPerBug: function () {
      return this.bugTypeOptions[this.currentBugType].numImagesPerBug
    },
    timeBetweenBugs: function () {
      return (this.bugsSettings.iti || 2) * 1000
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
      this.step = 0
      this.isDead = false
      this.isChangingDirection = false
      this.currentBugSize = this.getRadiusSize()
      this.initiateStartPosition()
      this.setNextAngle(this.directionAngle)
    },
    move() {
      if (this.isDead || this.isRetreated) {
        // the drawBug function will draw the dead bug
        this.draw()
        return
      }
      this.edgeDetection()
      this.straightMove()
      this.draw()
    },
    edgeDetection() {
      if (this.isChangingDirection) {
        return
      }
      // borders
      let radius = this.currentBugSize / 2
      if (!this.isInsidePolicy) {
        radius = -this.currentBugSize
      }
      if (this.x < radius || this.x > this.canvas.width - radius ||
          this.y < radius || this.y > this.canvas.height - radius) {
        this.setAfterEdgeAngle()
      }
    },
    setAfterEdgeAngle() {
      let nextAngle = this.directionAngle
      if (this.isInsidePolicy || this.isOutsideBackPolicy) {
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
      } else if (this.isOutsideEndPolicy) {
        this.hideBug()
      }
      this.setNextAngle(nextAngle)
    },
    setNextAngle(angle) {
      this.vx = this.currentSpeed * Math.cos(angle)
      this.vy = this.currentSpeed * Math.sin(angle)
    },
    initiateStartPosition() {
      this.x = this.startPosition[0]
      this.y = this.startPosition[1]
    },
    straightMove(noiseWeight = null) {
      let noise = noiseWeight ? 0 : noiseWeight * this.getRandomNoise()
      this.dx = this.vx + noise
      this.dy = this.vy + noise
      this.x += this.dx
      this.y += this.dy
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
    hideBug() {
      // hide bug when exit hole reached
      let fadeTimeoutValue = this.isOutsideEndPolicy ? 0 : 100
      let fadeTimeout = setTimeout(() => {
        this.isRetreated = true
        this.$emit('bugRetreated')
        let initTimeout = setTimeout(() => {
          this.initBug()
          clearTimeout(initTimeout)
        }, this.timeBetweenBugs)
        clearTimeout(fadeTimeout)
      }, fadeTimeoutValue)
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
      if (this.isHalfCircleMovement || (this.isMoveInCircles && this.isCircleTrackReached && !this.isHoleRetreatStarted)) {
        return Math.atan2(this.y - this.r0[1], this.x - this.r0[0]) + (this.isCounterClockWise ? 0 : Math.PI)
      }
      return Math.atan2(this.dy, this.dx) + Math.PI / 2
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
    changeDirectionTimeout() {
      this.isChangingDirection = true
      let t = setTimeout(() => {
        this.isChangingDirection = false
        clearTimeout(t)
      }, 300)
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
    }
  }
}
