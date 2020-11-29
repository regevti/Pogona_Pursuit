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
        @ended="onEnded"
    />
<!--    <video v-if="isVideoFile()" autoplay loop>-->
<!--      <source :src="url" :type="videoType">-->
<!--    </video>-->
  </div>
</template>

<script>
import FrameVideo from 'vue-frame-video'
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
      muted: true
    }
  },
  mounted() {
    let video = this.$refs.videoElement.getVideoElement()
    video.setAttribute('loop', 'true')
  },
  methods: {
    onFrameUpdate() {
      this.framesLog.push({
        time: Date.now(),
        frame: this.frameId
      })
      this.frameId++
    },
    onEnded() {
      this.frameId = 1
    },
    isVideoFile() {
      let url = this.url.toLowerCase()
      return url.endsWith('.avi') || url.endsWith('.mp4')
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
</style>