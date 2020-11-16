<template>
  <div id="wrapper">
    <img v-if="!isVideoFile()" :src="url" alt=""/>
    <video v-if="isVideoFile()" autoplay loop>
      <source :src="url" :type="videoType">
    </video>
  </div>
</template>

<script>
export default {
  name: 'media',
  props: {
    url: String
  },
  computed: {
    videoType() {
      if (this.url.endsWith('.avi')) {
        return 'video/avi'
      } else if (this.url.endsWith('.mp4')) {
        return 'video/mp4'
      }
    }
  },
  methods: {
    isVideoFile() {
      return this.url.endsWith('.avi') || this.url.endsWith('.mp4')
    }
  }
}
</script>

<style scoped>
#wrapper {
  position: absolute;
  top: 0;
  bottom: 0;
  width: 100%;
  height: 100%;
  overflow: hidden;
}

#wrapper, img {
  width: 100%;
  height: 100%;
}

#wrapper, video {
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