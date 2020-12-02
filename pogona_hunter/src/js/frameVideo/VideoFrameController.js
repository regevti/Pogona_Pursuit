import HijackEventImpl from './HijackEventImpl.js'
import Logger from './Logger'

const FrameRates = {
  film: 24
}

const Direction = {
  forward: 1,
  backward: -1
}

const Events = {
  play: 'play',
  playing: 'playing',
  loadstart: 'loadstart',
  durationchange: 'durationchange',
  loadedmetadata: 'loadedmetadata',
  loadeddata: 'loadeddata',
  progress: 'progress',
  canplay: 'canplay',
  canplaythrough: 'canplaythrough',
  error: 'error',
  pause: 'pause',
  ratechange: 'ratechange',
  seeked: 'seeked',
  seeking: 'seeking',
  timeupdate: 'timeupdate',
  volumechange: 'volumechange',
  waiting: 'waiting',
  ended: 'ended',
  resolutionswitching: 'resolutionswitching',
  resolutionswitched: 'resolutionswitched',
  fullscreenchange: 'fullscreenchange',
  frameupdate: 'frameupdate'
}

export default class VideoFrameController {
  constructor({ videoController, frameRate }) {
    if (!videoController) {
      throw new Error('videoController empty')
    }
    Logger.log('VideoFrameController init')
    this.videoController = videoController
    this._frameRate = frameRate || FrameRates.film
    this._hijackEventImpl = new HijackEventImpl([Events.frameupdate])

    this.initListener()
  }

  initListener() {
    this._callBackFrameChange = this.onFrameChange
  }

  onFrameChange() {
    this._hijackEventImpl.dispatchEvent(Events.frameupdate)
  }

  static get FrameRates() {
    return FrameRates
  }

  static get Events() {
    return Events
  }

  /**
   * 销毁
   */
  dispose() {
    this.stopListen()
    this._hijackEventImpl.dispose()
    this._hijackEventImpl = null
    this.videoController.dispose()
    this.videoController = null
  }

  _wrapTime(time) {
    return time < 10 ? '0' + time : time
  }

  listen(tick) {
    let _videoFrameController = this
    if (this.interval) {
      throw new Error('videoFrameController已在监听frameChange')
    }
    this.interval = setInterval(function() {
      if (_videoFrameController.paused || _videoFrameController.ended) {
        return
      }
      _videoFrameController._callBackFrameChange(_videoFrameController)
    }, tick || 1000 / _videoFrameController.frameRate / 2)
  }

  _handleHijackEvent(eventType) {
    switch (eventType) {
      case Events.frameupdate:
        this.listen()
        break
    }
  }

  _callBackFrameChange(videoFrameController) {
    let result = videoFrameController.getCurrentSMPTE()
    videoFrameController.listenCallBack &&
      videoFrameController.listenCallBack(result)
  }

  stopListen() {
    if (this.interval) {
      clearInterval(this.interval)
      this.interval = null
    }
  }
  /**
   * 向前frames帧.
   *
   * @param  {Number} frames - 向前的帧数，默认1帧
   * @param  {Function} callback - 回调函数
   */
  seekForward(frames, callback) {
    if (!frames) {
      frames = 1
    }
    this._seek(Direction.forward, Number(frames))
    return callback ? callback() : true
  }

  /**
   * 向后frames帧.
   *
   * @param  {Number} frames - 向后的帧数，默认1帧
   * @param  {Function} callback - 回调函数
   */
  seekBackward(frames, callback) {
    if (!frames) {
      frames = 1
    }
    this._seek(Direction.backward, Number(frames))
    return callback ? callback() : true
  }

  /**
   * private
   * 视频跳跃
   *
   * @param  {String} direction - 前进或者后退，VideoFrameController.Direction中的值
   * @param  {Number} frames - 前进或后退的帧数
   */
  _seek(direction, frames) {
    if (!this.videoController.paused) {
      this.videoController.pause()
    }
    let frame = Number(this.getCurrentFrameNumber())
    frame = direction === Direction.backward ? frame - frames : frame + frames
    let SMPTE = this.convertFrameNumberToSMPTE(frame)
    // this.videoController.currentTime = ((((direction === Direction.backward ? (frame - frames) : (frame + frames))) / this.frameRate) + 0.00001);
    // this.currentTime = ((((direction === Direction.backward ? (frame - frames) : (frame + frames))) / this.frameRate));
    this.seekTo({ SMPTE: SMPTE })
    // this.currentTime = this.currentTime - 0.002;
  }

  /**
   * 返回从0秒起到当前播放位置的帧数
   *
   * @return {Number} - 帧
   */
  getCurrentFrameNumber() {
    return Math.floor(this.currentTime.toFixed(5) * this._frameRate)
  }

  /**
   * 返回当前播放位置的SMPTE格式时间字符串
   *
   * @return {String} HH:MM:SS:FF
   */
  getCurrentSMPTE() {
    let seconds = this.currentTime
    return this.convertSecondsToSMPTE(seconds)
  }

  /**
   * 返回当前视频所有帧数
   */
  getTotalFrames() {
    return this.convertSecondToFrameNumber(this.videoController.duration())
  }

  /**
   * 返回当前视频播放的时间，单位秒
   *
   * @return {Number}
   */
  getCurrentTime() {
    return this.videoController.currentTime
  }

  getCurrentSMPTETime() {
    const SMPTE = this.getCurrentSMPTE()
    return this.convertSMPTEToMilliseconds(SMPTE) / 1000
  }

  /**
   * SMPTE to 毫秒
   * @param {Number}
   */
  convertSMPTEToMilliseconds(SMPTE) {
    var frames = Number(SMPTE.split(':')[3])
    var milliseconds = (1000 / this._frameRate) * (isNaN(frames) ? 0 : frames)
    function toSeconds(SMPTE) {
      let time = SMPTE.split(':')
      return Number(time[0]) * 60 * 60 + Number(time[1]) * 60 + Number(time[2])
    }
    let ms = toSeconds(SMPTE) * 1000 + milliseconds
    ms = Math.round(ms)
    return ms
  }

  /**
   * Frame to SMPTE
   * @param {Number}
   */
  convertFrameNumberToSMPTE(frameNumber) {
    frameNumber = Number(frameNumber)
    let fps = this._frameRate
    let _hour = fps * 60 * 60
    let _minute = fps * 60
    let _hours = (frameNumber / _hour).toFixed(0)
    let _minutes = Number((frameNumber / _minute).toString().split('.')[0]) % 60
    let _seconds = Number((frameNumber / fps).toString().split('.')[0]) % 60
    let wrap = this._wrapTime
    let SMPTE =
      wrap(_hours) +
      ':' +
      wrap(_minutes) +
      ':' +
      wrap(_seconds) +
      ':' +
      wrap(frameNumber % fps)
    return SMPTE
  }

  /**
   * Second to Frame
   * @param {Number}
   */
  convertSecondToFrameNumber(seconds) {
    return Math.floor(seconds.toFixed(5) * this._frameRate)
  }

  /**
   * Second to SMPTE
   * @param {Number}
   */
  convertSecondsToSMPTE(seconds) {
    let time = Number(seconds)
    let frameRate = this._frameRate
    let dt = new Date()
    let format = 'hh:mm:ss:ff'
    dt.setHours(0)
    dt.setMinutes(0)
    dt.setSeconds(0)
    dt.setMilliseconds(time * 1000)
    let wrap = this._wrapTime
    // 小于10时，前面补0
    return format.replace(/hh|mm|ss|ff/g, function(format) {
      switch (format) {
        case 'hh':
          return wrap(dt.getHours() < 13 ? dt.getHours() : dt.getHours() - 12)
        case 'mm':
          return wrap(dt.getMinutes())
        case 'ss':
          return wrap(dt.getSeconds())
        case 'ff':
          return wrap(Math.round((time % 1) * frameRate))
      }
    })
  }

  /**
   *
   * @param {void} config
   */
  seekTo(config) {
    config = config || {}
    let seekTime = 0
    let SMPTE
    var option = Object.keys(config)[0]

    switch (option) {
      case 'frame':
        SMPTE = this.convertFrameNumberToSMPTE(config[option])
        seekTime = this.convertSMPTEToMilliseconds(SMPTE) / 1000 + 0.001
        break
      case 'seconds':
        // SMPTE = this.convertSecondsToSMPTE(config[option]);
        // seekTime = ((this.convertSMPTEToMilliseconds(SMPTE) / 1000) + 0.001);
        seekTime = Number(config[option]) + 0.001
        break
      case 'milliseconds':
        seekTime = Number(config[option]) / 1000
        break
    }

    if (option === 'SMPTE' || option === 'time') {
      SMPTE = config[option]
      seekTime = this.convertSMPTEToMilliseconds(SMPTE) / 1000 + 0.001
      this.videoController.currentTime = seekTime
      Logger.log('seekTime Time ', seekTime)
      this._callBackFrameChange(this)
      return
    }

    if (!isNaN(seekTime)) {
      this.videoController.currentTime = seekTime
      Logger.log('seekTime Time ', seekTime)
      this._callBackFrameChange(this)
    }
  }

  get paused() {
    return this.videoController.paused
  }

  get ended() {
    return this.videoController.ended
  }

  get src() {
    return this.videoController.src
  }

  set src(url) {
    this.videoController.src = url
  }

  set frameRate(value) {
    this._frameRate = value
  }

  get frameRate() {
    return this._frameRate
  }

  play() {
    this.videoController.play()
  }

  pause() {
    this.videoController.pause()
    let SMPTE = this.getCurrentSMPTE()
    this.seekTo({ SMPTE: SMPTE })
  }

  duration() {
    return this.videoController.duration()
  }

  set currentTime(value) {
    Logger.log('set currentTime ', value)
    this.seekTo({ seconds: value })
  }

  get currentTime() {
    return Math.round(this.getCurrentTime() * 1000) / 1000
  }

  set volume(percent) {
    this.videoController.volume = percent
  }

  get volume() {
    return this.videoController.volume
  }

  set poster(src) {
    this.videoController.poster = src
  }

  get poster() {
    return this.videoController.poster
  }

  requestFullscreen() {
    this.videoController.requestFullscreen()
  }

  exitFullscreen() {
    this.videoController.exitFullscreen()
  }

  isFullscreen() {
    return this.videoController.isFullscreen()
  }

  buffered() {
    return this.videoController.buffered()
  }

  bufferedPercent() {
    return this.videoController.bufferedPercent()
  }

  on(type, listerner) {
    if (this._hijackEventImpl.hijackEvent(type, listerner)) {
      this._handleHijackEvent(type)
      return
    }
    this.videoController.on(type, listerner)
  }

  off(type, listerner) {
    if (this._hijackEventImpl.offHijackEvent(type, listerner)) {
      return
    }
    this.videoController.off(type, listerner)
  }

  once(type, listerner) {
    if (this._hijackEventImpl.onceHijackEvent(type, listerner)) {
      return
    }
    this.videoController.once(type, listerner)
  }
}
