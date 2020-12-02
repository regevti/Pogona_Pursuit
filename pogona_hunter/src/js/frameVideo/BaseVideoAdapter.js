import HijackEventImpl from './HijackEventImpl.js'
import Logger from './Logger.js'

export default class BaseVideoController {
  // todo get fix css head and remove all if/else for full screen
  constructor(videoElement) {
    if (!videoElement) {
      throw new Error('videoElement empty')
    }
    Logger.log('BaseVideoController init')
    this.videoElement = videoElement
    this._isFullscreen = false
    this._timeRanges = null
    this._bufferedPercent = 0
    this._hijackEventImpl = new HijackEventImpl([
      'fullscreenchange',
      'progress'
    ])
    this.initListener()
  }

  dispose() {
    this._hijackEventImpl.dispose()
    this._hijackEventImpl = null
    this.removeListener()
    this.videoElement = null
  }

  initListener() {
    this.onFullScreenChange = this.onFullScreenChange.bind(this)
    this.onProgess = this.onProgess.bind(this)
    const ele = this.videoElement
    let eventType = null
    if (ele.webkitRequestFullScreen) {
      eventType = 'webkitfullscreenchange'
    } else if (ele.mozRequestFullScreen) {
      eventType = 'mozfullscreenchange'
    } else if (ele.requestFullscreen) {
      eventType = 'fullscreenchange'
    } else if (ele.msRequestFullscreen) {
      document.onmsfullscreenchange = this.onFullScreenChange
    }
    if (eventType) {
      ele.addEventListener(eventType, this.onFullScreenChange)
    }

    ele.addEventListener('progress', this.onProgess)
  }

  removeListener() {
    const ele = this.videoElement
    let eventType = null
    ele.removeEventListener('progress', this.onProgess)
    if (ele.webkitRequestFullScreen) {
      eventType = 'webkitfullscreenchange'
    } else if (ele.mozRequestFullScreen) {
      eventType = 'mozfullscreenchange'
    } else if (ele.requestFullscreen) {
      eventType = 'fullscreenchange'
    } else if (ele.msRequestFullscreen) {
      document.onmsfullscreenchange = null
    }
    if (eventType) {
      ele.removeEventListener(eventType, this.onFullScreenChange)
    }
  }

  onFullScreenChange(event) {
    let self = this
    self._isFullscreen = !self._isFullscreen
    this._hijackEventImpl.dispatchEvent('fullscreenchange')
  }

  onProgess(event) {
    let video = this.videoElement
    let timeRanges = video.buffered
    this._timeRanges = timeRanges
    let timeRangeLength = timeRanges.length
    let totalLoaded = 0
    for (let i = 0; i < timeRangeLength; ++i) {
      totalLoaded += timeRanges.end(i) - timeRanges.start(i)
    }
    this._bufferedPercent = totalLoaded / this.duration()
    if (this._bufferedPercent > 0.9999) {
      this._bufferedPercent = 1
    }
    this._hijackEventImpl.dispatchEvent('progress')
  }

  play() {
    this.videoElement.play()
  }

  pause() {
    this.videoElement.pause()
  }

  set currentTime(seconds) {
    this.videoElement.currentTime = seconds
  }

  get currentTime() {
    return this.videoElement.currentTime
  }

  get paused() {
    return this.videoElement.paused
  }

  get ended() {
    return this.videoElement.ended
  }

  duration() {
    return this.videoElement.duration
  }

  set volume(percent) {
    this.videoElement.volume = percent
  }

  get volume() {
    return this.videoElement.volume
  }

  set poster(src) {
    this.videoElement.poster = src
  }

  get poster() {
    return this.videoElement.poster
  }

  requestFullscreen() {
    const ele = this.videoElement
    if (ele.webkitRequestFullScreen) {
      ele.webkitRequestFullScreen()
    } else if (ele.mozRequestFullScreen) {
      ele.mozRequestFullScreen()
    } else if (ele.requestFullscreen) {
      ele.requestFullscreen()
    } else if (ele.msRequestFullscreen) {
      ele.msRequestFullscreen()
    }
  }

  exitFullscreen() {
    const ele = this.videoElement
    if (ele.webkitCancelFullScreen) {
      ele.webkitCancelFullScreen()
    } else if (ele.mozCancelFullScreen) {
      ele.mozCancelFullScreen()
    } else if (ele.exitFullscreen) {
      ele.exitFullscreen()
    } else if (ele.msCancelFullScreen) {
      ele.msCancelFullScreen()
    }
  }

  isFullscreen() {
    let fullScreenEle =
      document.fullscreenElement ||
      document.msFullscreenElement ||
      document.mozFullScreenElement ||
      document.webkitFullscreenElement

    if (fullScreenEle === this.videoElement) {
      return true
    }

    return this._isFullscreen
  }

  buffered() {
    // onProgess 并不是稳定有的回调，所以每次即时获取 buffered
    // return this._timeRanges
    return this.videoElement.buffered
  }

  bufferedPercent() {
    return this._bufferedPercent
  }

  on(type, listerner) {
    if (this._hijackEventImpl.hijackEvent(type, listerner)) {
      return
    }
    this.videoElement.addEventListener(type, listerner)
  }

  off(type, listerner) {
    if (this._hijackEventImpl.offHijackEvent(type, listerner)) {
      return
    }
    this.videoElement.removeEventListener(type, listerner)
  }

  once(type, listerner) {
    let self = this
    if (this._hijackEventImpl.onceHijackEvent(type, listerner)) {
      return
    }
    let onceCallBackFun = function(event) {
      listerner.call(self, event)
      self.off(type, onceCallBackFun)
    }
    self.videoElement.addEventListener(type, onceCallBackFun)
  }

  get src() {
    return this.videoElement.src
  }

  set src(url) {
    this.videoElement.src = url
  }
}
