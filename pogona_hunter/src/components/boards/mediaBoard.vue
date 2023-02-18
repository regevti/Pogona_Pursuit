<template>
  <div id="wrapper">
    <img v-if="!isVideoFile()" :src="url" alt=""/>
    <FrameVideo v-if="isVideoFile()"
          id="frame-video"
          ref="videoElement"
          :src="url"
          :autoplay="autoplay"
          :muted="muted"
          @frameupdate="onFrameUpdate"
      />
    <canvas id="media-canvas" v-bind:height="canvasHeight" v-bind:width="canvasWidth"
              v-on:mousedown="handleTouchEvent($event)" style="z-index: 10;">
    </canvas>
  </div>
</template>

<script>
import FrameVideo from './frameVideo'
export default {
  name: 'media',
  components: {
    FrameVideo
  },
  props: {
    url: String
  },
  data() {
    return {
      frameId: 1,
      framesLog: [],
      autoplay: 'autoplay',
      muted: true,
      canvasWidth: window.innerWidth,
      canvasHeight: window.innerHeight
    }
  },
  mounted() {
    this.canvas = document.getElementById('media-canvas')
    if (this.isVideoFile()) {
      let video = this.$refs.videoElement.getVideoElement()
      video.setAttribute('loop', 'true')
    }
  },
  methods: {
    onFrameUpdate(event) {
      if (event.SMPTE === '00:00:00:00') {
        this.frameId = 1
      }
      this.framesLog.push({
        time: Date.now(),
        frame: this.frameId
      })
      this.frameId++
    },
    isVideoFile() {
      let url = this.url.toLowerCase()
      return url.endsWith('.avi') || url.endsWith('.mp4')
    },
    handleTouchEvent(event) {
      let x = event.x - this.canvas.offsetLeft
      let y = event.y - this.canvas.offsetTop
      console.log(x, y)
      this.$mqtt.publish('event/log/touch', JSON.stringify({
          time: Date.now(),
          x: x,
          y: y,
          frame_id: this.frameId
        }))
    }
  }
}
</script>

<style>
#frame-video, div {
  position: absolute;
  /*top: 0;*/
  bottom: 0;
  width: 100%;
  height: 100%;
  overflow: hidden;
}

#wrapper, img {
  width: 100%;
  height: 100%;
}

video {
  /* Make video to at least 100% wide and tall */
  min-width: 100%;
  min-height: 100%;

  /* Setting width & height to auto prevents the browser from stretching or squishing the video */
  width: auto;
  height: auto;

  /* Center the video */
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
}
#media-canvas {
  padding: 0;
  top: 0;
  /*margin: 20px auto 0;*/
  display: block;
  background: #e8eaf6;
  /*position: absolute;*/
  bottom: 10px;
}
</style>