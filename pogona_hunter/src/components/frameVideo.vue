<template>
  <div>
    <video
      ref="videoElement"
      :id="id"
      :src="src"
      :controls="controls"
      :autoplay="autoplay"
      :muted="muted"
      :initial-time="initialTime"
      width="300"
      height="255"
      preload="auto"
    />

    <canvas
      ref="canvasElement"
      id="player-capture-temp"
      style="display: none;"
    />
  </div>
</template>

<script>
import VideoFrameController from '../js/frameVideo/VideoFrameController.js'
import BaseVideoAdapter from '../js/frameVideo/BaseVideoAdapter.js'
import ErrorTypes from '../js/frameVideo/ErrorTypes.js'

export default {
  name: 'vue-frame-video',
  props: {
    id: {
      type: String,
      required: false,
      default: 'player-container-id'
    },
    src: {
      type: String,
      required: false,
      default: ''
    },
    autoplay: {
      type: Boolean,
      required: false,
      default: false
    },
    loop: {
      type: Boolean,
      required: false,
      default: false
    },
    muted: {
      type: Boolean,
      required: false,
      default: false
    },
    initialTime: {
      type: Number,
      required: false,
      default: 0
    },
    controls: {
      type: Boolean,
      required: false,
      default: false
    },
    frameRate: {
      type: Number,
      required: false,
      default: VideoFrameController.FrameRates.film
    },
    playsinline: {
      type: Boolean,
      default: false
    }
  },

  watch: {
    src(newValue) {
      this.videoFrameController.once(
        VideoFrameController.Events.canplay,
        this._setInitTime
      )
    },
    frameRate(newValue) {
      this.videoFrameController.frame = newValue
    },
    playsinline(newValue) {
      let video = this.$refs.videoElement
      if (newValue) {
        video.setAttribute('playsinline', this.playsinline)
        video.setAttribute('webkit-playsinline', this.playsinline)
        video.setAttribute('x5-playsinline', this.playsinline)
        video.setAttribute('x5-video-player-type', 'h5')
        video.setAttribute('x5-video-player-fullscreen', false)
      } else {
        video.removeAttribute('playsinline', this.playsinline)
        video.removeAttribute('webkit-playsinline', this.playsinline)
        video.removeAttribute('x5-playsinline', this.playsinline)
        video.removeAttribute('x5-video-player-type', 'h5')
        video.removeAttribute('x5-video-player-fullscreen', false)
      }
    }
  },

  data: function() {
    return {
      SMPTE: '00:00:00:00',
      logicPlaying: false
    }
  },

  mounted: function() {
    let self = this
    let video = this.$refs.videoElement
    this.videoFrameController = new VideoFrameController({
      videoController: new BaseVideoAdapter(video),
      frameRate: this.frameRate
    })
    self._registerListener() // 初始化监听
    if (self.initialTime) {
      this.videoFrameController.once(
        VideoFrameController.Events.canplay,
        self._setInitTime
      )
    }
    if (this.playsinline) {
      video.setAttribute('playsinline', this.playsinline)
      video.setAttribute('webkit-playsinline', this.playsinline)
      video.setAttribute('x5-playsinline', this.playsinline)
      video.setAttribute('x5-video-player-type', 'h5')
      video.setAttribute('x5-video-player-fullscreen', false)
    }
  },

  destroyed: function() {
    this._removeListener()
    this.videoFrameController.dispose()
    this.videoFrameController = null
  },

  methods: {
    getVideoElement: function() {
      return this.$refs.videoElement
    },
    captureIMG: function(width, height, scale = 1) {
      let video = this.$refs.videoElement
      let canvas = this.$refs.canvasElement
      let ctx = canvas.getContext('2d')
      width = width || video.videoWidth
      height = height || video.videoWidth
      canvas.width = width * scale
      canvas.height = height * scale
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
      var base64 = '' // base64
      base64 = canvas.toDataURL('image/png')
      return base64
    },
    onPlay: function(event) {
      this.logicPlaying = true
      this.$emit(VideoFrameController.Events.play)
    },
    onPause: function(event) {
      this.logicPlaying = false
      this.$emit(VideoFrameController.Events.pause)
    },
    onEnded: function(event) {
      this.logicPlaying = false
      this.$emit(VideoFrameController.Events.ended)
    },
    onWaiting: function(event) {
      this.$emit(VideoFrameController.Events.waiting)
    },
    onTimeUpdate: function(event) {
      this.$emit(VideoFrameController.Events.timeupdate, {
        currentTime: this.currentTime(),
        duration: this.duration()
      })
    },
    onFullScreenChange: function(event) {
      let param = { fullScreen: this.isFullscreen() }
      this.$emit(VideoFrameController.Events.fullscreenchange, param)
    },
    onFrameUpdate: function(event) {
      let currentSMPTE = this.videoFrameController.getCurrentSMPTE()
      if (currentSMPTE !== this.SMPTE) {
        this.SMPTE = currentSMPTE
        this.$emit(VideoFrameController.Events.frameupdate, {
          SMPTE: currentSMPTE
        })
      }
    },
    onProgress: function(event) {
      this.$emit(VideoFrameController.Events.progress, {
        buffered: this.bufferedPercent()
      })
    },
    onError: function(event) {
      let errorCode =
        event && event.target && event.target.error && event.target.error.code
      if (errorCode === ErrorTypes.ErrorRet.MEDIA_ERR_SRC_NOT_SUPPORTED.ret) {
        if (this.src === '') {
          errorCode = ErrorTypes.ErrorRet.MEDIA_NO_AVAILABLE_SRC.ret
        }
      }
      this.$emit(VideoFrameController.Events.error, {
        ret: errorCode,
        msg: ErrorTypes.RetMsgMap[errorCode]
      })
    },
    onDurationchange: function(event) {
      this.$emit(VideoFrameController.Events.durationchange, {
        duration: this.duration(),
        SMPTE: this.videoFrameController.convertSecondsToSMPTE(this.duration())
      })
    },
    onCanplay: function(event) {
      this.$emit(VideoFrameController.Events.canplay)
    },
    onCanplayThrough: function(event) {
      this.$emit(VideoFrameController.Events.canplaythrough)
    },
    convertSecondsToSMPTE: function(seconds) {
      return this.videoFrameController.convertSecondsToSMPTE(seconds)
    },
    convertSecondToFrameNumber: function(seconds) {
      return this.videoFrameController.convertSecondToFrameNumber(seconds)
    },
    convertSMPTEToMilliseconds: function(SMPTE) {
      return this.videoFrameController.convertSMPTEToMilliseconds(SMPTE)
    },
    convertFrameNumberToSMPTE: function(FrameNumbe) {
      return this.videoFrameController.convertFrameNumberToSMPTE(FrameNumbe)
    },
    _setInitTime(event) {
      this.currentTime(this.initialTime)
    },
    isPlaying() {
      return this.logicPlaying
    },
    play: function() {
      this.videoFrameController.play()
    },
    pause: function() {
      this.videoFrameController.pause()
    },
    currentTime: function(seconds) {
      if (seconds === 0 || seconds) {
        this.videoFrameController.currentTime = seconds
      }
      return this.videoFrameController.currentTime
    },
    duration: function() {
      return this.videoFrameController.duration()
    },
    volume: function(percent) {
      return this.videoFrameController.volume(percent)
    },
    poster: function(src) {
      return this.videoFrameController.poster(src)
    },
    requestFullscreen: function() {
      this.videoFrameController.requestFullscreen()
    },
    exitFullscreen: function() {
      this.videoFrameController.exitFullscreen()
    },
    isFullscreen: function() {
      return this.videoFrameController.isFullscreen()
    },
    buffered: function() {
      return this.videoFrameController.buffered()
    },
    bufferedPercent: function() {
      return this.videoFrameController.bufferedPercent()
    },
    dispose: function() {
      this.videoFrameController.dispose()
    },
    on: function(type, listerner) {
      this.videoFrameController.on(type, listerner)
    },
    off: function(type, listerner) {
      this.videoFrameController.off(type, listerner)
    },
    once: function(type, listerner) {
      this.videoFrameController.once(type, listerner)
    },
    seekForward: function() {
      this.videoFrameController.seekForward()
    },
    seekBackward: function() {
      this.videoFrameController.seekBackward()
    },
    _registerListener: function() {
      // let self = this
      this.videoFrameController.on(
        VideoFrameController.Events.play,
        this.onPlay
      )
      this.videoFrameController.on(
        VideoFrameController.Events.pause,
        this.onPause
      )
      this.videoFrameController.on(
        VideoFrameController.Events.timeupdate,
        this.onTimeUpdate
      )
      this.videoFrameController.on(
        VideoFrameController.Events.fullscreenchange,
        this.onFullScreenChange
      )
      this.videoFrameController.on(
        VideoFrameController.Events.ended,
        this.onEnded
      )
      this.videoFrameController.on(
        VideoFrameController.Events.waiting,
        this.onWaiting
      )
      this.videoFrameController.on(
        VideoFrameController.Events.progress,
        this.onProgress
      )
      this.videoFrameController.on(
        VideoFrameController.Events.error,
        this.onError
      )
      this.videoFrameController.on(
        VideoFrameController.Events.frameupdate,
        this.onFrameUpdate
      )
      this.videoFrameController.on(
        VideoFrameController.Events.durationchange,
        this.onDurationchange
      )
      this.videoFrameController.on(
        VideoFrameController.Events.canplay,
        this.onCanplay
      )
      this.videoFrameController.on(
        VideoFrameController.Events.canplaythrough,
        this.onCanplayThrough
      )
    },
    _removeListener: function() {
      // let self = this
      this.videoFrameController.off(
        VideoFrameController.Events.play,
        this.onPlay
      )
      this.videoFrameController.off(
        VideoFrameController.Events.pause,
        this.onPause
      )
      this.videoFrameController.off(
        VideoFrameController.Events.timeupdate,
        this.onTimeUpdate
      )
      this.videoFrameController.off(
        VideoFrameController.Events.fullscreenchange,
        this.onFullScreenChange
      )
      this.videoFrameController.off(
        VideoFrameController.Events.ended,
        this.onEnded
      )
      this.videoFrameController.off(
        VideoFrameController.Events.waiting,
        this.onWaiting
      )
      this.videoFrameController.off(
        VideoFrameController.Events.progress,
        this.onProgress
      )
      this.videoFrameController.off(
        VideoFrameController.Events.error,
        this.onError
      )
      this.videoFrameController.off(
        VideoFrameController.Events.frameupdate,
        this.onFrameUpdate
      )
      this.videoFrameController.off(
        VideoFrameController.Events.durationchange,
        this.onDurationchange
      )
      this.videoFrameController.off(
        VideoFrameController.Events.canplay,
        this.onCanplay
      )
      this.videoFrameController.off(
        VideoFrameController.Events.canplaythrough,
        this.onCanplayThrough
      )
    }
  }
}
</script>
