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
      currentBugType: '',
      imgSrc: '',
      mass: 1
    }
  },
  props: {
    x0: Number,
    y0: Number,
    radius: Number,
    bugTypes: Array[String],
    timeInEdge: Number,
    speed: Number,
    movementType: String
  },
  computed: {
    isMoveInCircles: function () {
      return this.movementType === 'circle'
    },
    numImagesPerBug: function () {
      return this.bugTypeOptions[this.currentBugType].numImagesPerBug
    },
    isStatic: function () {
      return this.bugTypeOptions[this.currentBugType].isStatic
    },
    stepsPerImage: function () {
      return this.bugTypeOptions[this.currentBugType].stepsPerImage
    }
  },
  mounted() {
    this.canvas = this.$parent.canvas
    this.ctx = this.canvas.getContext('2d')
    this.isOutEdged = false
    this.isDead = false
    this.dx = plusOrMinus() * this.speed
    this.dy = plusOrMinus() * this.speed
    this.initBug(this.x0, this.y0, this.dx, this.dy)
  },
  methods: {
    getImageSrc(fileName) {
      return require('@/assets' + fileName)
    },
    getDeadImage() {
      let img = new Image()
      img.src = this.getImageSrc(`/${this.currentBugType}_dead.png`)
      return img
    },
    getNextBugType() {
      if (!Array.isArray(this.bugTypes)) {
        this.currentBugType = this.bugTypes
        return
      }
      let nextBugOptions = this.bugTypes.filter(bug => bug !== this.currentBugType)
      let nextIndex = randomRange(0, nextBugOptions.length)
      this.currentBugType = nextBugOptions[nextIndex]
    },
    initBug(x, y, dx, dy) {
      if (this.isMoveInCircles) {
        this.r0 = [this.canvas.width / 2, this.canvas.height]
        this.r = Math.min(this.canvas.width / 2, this.canvas.height) * 0.75
      }
      this.x = x
      this.y = y
      this.dx = dx
      this.dy = dy
      this.step = 0
      this.theta = Math.PI
      this.getNextBugType()
    },
    move(particles) {
      if (this.isDead || this.isStatic || this.isOutEdged) {
        this.draw()
        return
      }
      this.edgeDetection()
      if (this.isMoveInCircles) {
        this.theta += Math.abs(this.dx) * Math.sqrt(2) / this.r
        this.x = this.r0[0] + this.r * Math.cos(this.theta)
        this.y = this.r0[1] + this.r * Math.sin(this.theta)
      } else {
        // move in straight lines
        this.x += this.dx
        this.y += this.dy
      }
      this.draw()

      for (let i = 0; i < particles.length; i++) {
        if (this === particles[i]) continue

        if (distance(this.x, this.y, particles[i].x, particles[i].y) <= this.radius + particles[i].radius) {
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
        this.ctx.drawImage(bugImage, -this.radius / 2, -this.radius / 2, this.radius, this.radius)
        this.ctx.setTransform(1, 0, 0, 1, 0, 0)
      } catch (e) {
        console.error(e)
      }
    },
    edgeTimeout(dx, dy) {
      this.isOutEdged = true
      let x = this.xEdge()
      let y = this.yEdge()
      const initAfterEdgeTimeout = setTimeout(() => {
        this.initBug(x, y, dx, dy)
        const outEdgeTimeout = setTimeout(() => {
          this.isOutEdged = false
          clearTimeout(outEdgeTimeout)
        }, 50)
        clearTimeout(initAfterEdgeTimeout)
      }, this.timeInEdge)
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
      const margin = this.radius
      if (this.x >= this.canvas.width + margin || this.x <= -margin) {
        this.edgeTimeout(-this.dx, this.dy)
      } else if (this.y >= this.canvas.height + margin || this.y <= -margin) {
        this.edgeTimeout(this.dx, -this.dy)
      }
    },
    isInsideBoard() {
      return this.x > 0 && this.x < this.canvas.width && this.y > 0 && this.y < this.canvas.height
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
        return Math.atan2(this.y - this.r0[1], this.x - this.r0[0]) + Math.PI
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
