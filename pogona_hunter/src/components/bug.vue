<template>
  <img ref="bugImg" :src="imgSrc" alt=""/>
</template>

<script>
import {distance, plusOrMinus, randomRange} from '@/js/helpers'

export default {
  name: 'bug',
  data() {
    return {
      bugTypeOptions: require('@/config.json')['bugTypes'],
      bugImages: [],
      currentBugType: undefined,
      currentRadiusSize: undefined,
      imgSrc: '',
      mass: 1
    }
  },
  props: {
    x0: Number,
    y0: Number,
    bugsSettings: Object
  },
  computed: {
    isMoveInCircles: function () {
      return this.bugsSettings.movementType === 'circle'
    },
    isRandomLine: function () {
      return this.bugsSettings.movementType === 'line'
    },
    isLowHorizontal: function () {
      return this.bugsSettings.movementType === 'low_horizontal'
    },
    numImagesPerBug: function () {
      return this.bugTypeOptions[this.currentBugType].numImagesPerBug
    },
    isStatic: function () {
      return this.bugTypeOptions[this.currentBugType].isStatic
    },
    stepsPerImage: function () {
      return this.bugTypeOptions[this.currentBugType].stepsPerImage
    },
    currentSpeed: function () {
      if (this.bugsSettings && this.bugsSettings.speed) {
        return this.bugsSettings.speed
      }
      return this.bugTypeOptions[this.currentBugType].speed
    }
  },
  mounted() {
    this.canvas = this.$parent.canvas
    this.ctx = this.canvas.getContext('2d')
    this.isOutEdged = false
    this.isDead = false
    if (!Array.isArray(this.bugsSettings.bugTypes)) {
      this.bugsSettings.bugTypes = [this.bugsSettings.bugTypes]
    }
    if (this.isMoveInCircles) {
      this.r0 = [this.canvas.width / 2, this.canvas.height]
      this.r = Math.min(this.canvas.width / 2, this.canvas.height) * 0.75
    }
    this.initBug(this.x0, this.y0, plusOrMinus(), plusOrMinus())
  },
  methods: {
    initBug(x, y, directionX, directionY) {
      this.getNextBugType()
      if (this.isLowHorizontal) {
        this.x = this.canvas.width
        this.y = this.canvas.height - 80
        this.dx = -1 * this.currentSpeed
        this.dy = 0
      } else if (this.isRandomLine) {
        this.x = x
        this.y = y
        this.dx = Math.sign(directionX) * this.currentSpeed * Math.sqrt(2)
        this.dy = Math.sign(directionY) * this.currentSpeed * Math.sqrt(2)
      }
      this.step = 0
      this.theta = Math.PI - Math.PI / 10
      this.currentRadiusSize = this.getRadiusSize()
    },
    move(particles) {
      if (this.isDead || this.isStatic || this.isOutEdged) {
        this.draw()
        return
      }
      this.edgeDetection()
      if (this.isMoveInCircles) {
        this.theta += Math.abs(this.currentSpeed) * Math.sqrt(2) / this.r
        this.x = this.r0[0] + (this.r * Math.cos(this.theta)) * (this.bugsSettings.isAntiClockWise ? -1 : 1)
        this.y = this.r0[1] + this.r * Math.sin(this.theta)
      } else {
        // move in straight lines
        this.x += this.dx
        this.y += this.dy
      }
      this.draw()

      for (let i = 0; i < particles.length; i++) {
        if (this === particles[i]) continue

        if (distance(this.x, this.y, particles[i].x, particles[i].y) <= this.currentRadiusSize + particles[i].currentRadiusSize) {
          this.collisionEffect(particles[i])
        }
      }
    },
    draw() {
      this.ctx.beginPath()
      let imgIndex = Math.floor(this.step / this.stepsPerImage)
      this.imgSrc = this.getImageSrc(`/${this.currentBugType}${imgIndex}.png`)
      this.drawImage()
      this.step++
      if (this.step > (this.numImagesPerBug - 1) * this.stepsPerImage) {
        this.step = 0
      }
      this.ctx.fill()
      this.ctx.closePath()
    },
    drawImage() {
      try {
        let bugImage = this.isDead ? this.getDeadImage() : this.$refs.bugImg
        this.ctx.setTransform(1, 0, 0, 1, this.x, this.y)
        this.ctx.rotate(this.getAngleRadians())
        // drawImage(image, dx, dy, dWidth, dHeight)
        this.ctx.drawImage(bugImage, -this.currentRadiusSize / 2, -this.currentRadiusSize / 2, this.currentRadiusSize, this.currentRadiusSize)
        this.ctx.setTransform(1, 0, 0, 1, 0, 0)
        // this.ctx.arc(this.x, this.y, this.currentRadiusSize / 1.5, 0, 2 * Math.PI, false)
        // this.ctx.fillStyle = 'rgba(255,0,0,0.3)'
        // this.ctx.fill()
      } catch (e) {
        console.error(e)
      }
    },
    edgeTimeout(dx, dy) {
      this.isOutEdged = true
      const initAfterEdgeTimeout = setTimeout(() => {
        this.initBug(this.xEdge(), this.yEdge(), dx, dy)
        const outEdgeTimeout = setTimeout(() => {
          this.isOutEdged = false
          clearTimeout(outEdgeTimeout)
        }, 50)
        clearTimeout(initAfterEdgeTimeout)
      }, this.bugsSettings.timeInEdge)
    },
    getImageSrc(fileName) {
      return require('@/assets' + fileName)
    },
    getDeadImage() {
      let img = new Image()
      img.src = this.getImageSrc(`/${this.currentBugType}_dead.png`)
      return img
    },
    getNextBugType() {
      if (this.bugsSettings.bugTypes.length === 1) {
        this.currentBugType = this.bugsSettings.bugTypes[0]
        return
      }
      let nextBugOptions = this.bugsSettings.bugTypes.filter(bug => bug !== this.currentBugType)
      let nextIndex = randomRange(0, nextBugOptions.length)
      this.currentBugType = nextBugOptions[nextIndex]
    },
    getRadiusSize() {
      if (this.bugsSettings.radiusSize) {
        return this.bugsSettings.radiusSize
      }
      let currentBugOptions = this.bugTypeOptions[this.currentBugType]
      return randomRange(currentBugOptions.radiusRange.min, currentBugOptions.radiusRange.max)
    },
    xEdge() {
      if (this.isMoveInCircles) {
        return this.r0[0] - Math.sqrt(this.r ** 2 - (this.canvas.height - this.r0[1]) ** 2)
      }
      if (this.x > this.canvas.width) {
        return this.canvas.width
      } else if (this.x < 0) {
        return 0
      }
      return this.x
    },
    yEdge() {
      if (this.y > this.canvas.height) {
        return this.canvas.height
      } else if (this.y < 0) {
        return 0
      }
      return this.y
    },
    edgeDetection() {
      if (this.isOutEdged) {
        return
      }
      const margin = this.currentRadiusSize
      if (this.x >= this.canvas.width + margin || this.x <= -margin) {
        this.edgeTimeout(-this.dx, this.dy)
      } else if (this.y >= this.canvas.height + margin || this.y <= -margin) {
        this.edgeTimeout(this.dx, -this.dy)
      }
    },
    isInsideBoard() {
      return this.x > 0 && this.x < this.canvas.width && this.y > 0 && this.y < this.canvas.height
    },
    isHit(x, y) {
      return distance(x, y, this.x, this.y) <= this.currentRadiusSize / 1.5
    },
    escape(xe, ye) {
      let dist = distance(this.x, this.y, xe, ye)
      if (dist < this.canvas.height / 1.5) {
        this.dx = Math.abs(this.dx) * Math.sign(this.x - xe)
        this.dy = Math.abs(this.dy) * Math.sign(this.y - ye)
      }
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
    collisionEffect(otherBug) {
      // collision between 2 bugs
      const angle = -Math.atan2(otherBug.y - this.y, otherBug.x - this.x)
      const u1 = this.rotate(this.dx, this.dy, angle)
      const u2 = this.rotate(otherBug.dx, otherBug.dy, angle)
      const v1 = {
        dx: ((this.mass - otherBug.mass) * u1.dx / (this.mass + otherBug.mass)) + (2 * otherBug.mass * u2.dx / (this.mass + otherBug.mass)),
        dy: u1.dy
      }
      const rotatedv1 = this.rotate(v1.dx, v1.dy, -angle)
      this.dx = rotatedv1.dx
      this.dy = rotatedv1.dy
      // const v2 = {
      //   dx: ((this.mass - otherBug.mass) * u2.dx / (this.mass + otherBug.mass)) + (2 * otherBug.mass * u1.dx / (this.mass + otherBug.mass)),
      //   dy: u2.dy
      // }
      // const rotatedv2 = this.rotate(v2.dx, v2.dy, -angle)
      // otherBug.dx = rotatedv2.dx
      // otherBug.dy = rotatedv2.dy
    }
  }
}
</script>
