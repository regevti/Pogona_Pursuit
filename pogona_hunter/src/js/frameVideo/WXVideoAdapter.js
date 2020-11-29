import BaseVideoAdapter from './BaseVideoAdapter'

export default class WXVideoController extends BaseVideoAdapter {
  constructor(videoContext) {
    super(videoContext)
    if (!videoContext) {
      throw new Error('videoElement empty')
    }
    this.wxTimeData = { currentTime: 0, duration: 0 }
    this.isPlaying = false
    this.passStartTime = null
    this._ended = false
  }

  // 覆盖父类方法
  initListener() {}

  removeListener() {}

  set currentTime(seconds) {
    this.videoElement.seek(seconds)
    // 小程序环境下，如果没有首次timeupdate,seek后不会触发timeupdate调用，手动更新time
    this.wxTimeData.currentTime = seconds
  }

  get currentTime() {
    let time = this.wxTimeData.currentTime ? this.wxTimeData.currentTime : 0
    if (this.isPlaying && this.passStartTime) {
      let nowTime = Date.now()
      let passTIme = nowTime - this.passStartTime
      time += passTIme / 1000
    }
    return time
  }

  timeupdate({ currentTime, duration }) {
    this.wxTimeData = { currentTime, duration }
    this.passStartTime = Date.now()
  }

  requestFullscreen() {
    this.videoElement.requestFullScreen()
  }

  exitFullscreen() {
    this.videoElement.exitFullScreen()
  }

  isFullscreen() {
    return this._isFullscreen
  }

  onFullscreenchange(event) {
    let { fullScreen } = event
    this._isFullscreen = fullScreen
  }

  duration() {
    return this.wxTimeData.duration
  }

  forceUpdateDuration(value) {
    this.wxTimeData.duration = value
  }

  onProgess(event) {
    let { buffered } = event
    this._bufferedPercent = buffered
  }

  buffered() {
    return this._bufferedPercent
  }

  set playing(isPlaying) {
    if (isPlaying === this.isPlaying) {
      return
    }
    if (isPlaying === true) {
      this.passStartTime = Date.now()
    } else {
      let nowTime = Date.now()
      let passTIme = nowTime - this.passStartTime
      this.wxTimeData.currentTime =
        this.wxTimeData.currentTime + passTIme / 1000
      this.passStartTime = 0
    }
    this.isPlaying = isPlaying
  }

  get paused() {
    return !this.isPlaying
  }

  get ended() {
    return this._paused
  }

  set ended(value) {
    this._ended = value
  }
}
