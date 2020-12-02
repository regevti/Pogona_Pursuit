export default class HijackEventImpl {
  constructor(eventTypesArr) {
    this._hijackEventTypes = {}
    for (let key in eventTypesArr) {
      this.addHijackEventType(eventTypesArr[key])
    }
  }

  addHijackEventType(evnetType) {
    if (!this._hijackEventTypes[evnetType]) {
      this._hijackEventTypes[evnetType] = []
    }
  }

  removeHijackEventType(evnetType) {
    if (this._hijackEventTypes[evnetType]) {
      this._hijackEventTypes[evnetType].length = 0
    }
    delete this._hijackEventTypes[evnetType]
  }

  dispatchEvent(evnetType, event) {
    let listenerArr = this._hijackEventTypes[evnetType]
    for (let key in listenerArr) {
      listenerArr[key](event)
    }
  }

  dispose() {
    for (let key in this._hijackEventTypes) {
      this._hijackEventTypes[key].length = 0
    }
  }

  /**
   * 劫持事件
   */
  hijackEvent(type, listerner) {
    let listenerArr = this._hijackEventTypes[type]
    if (listenerArr && listenerArr.indexOf(listerner) < 0) {
      listenerArr.push(listerner)
    }
    return !!listenerArr
  }

  /**
   * 劫持事件，只执行一次
   */
  onceHijackEvent(type, listerner) {
    let self = this
    let listenerArr = this._hijackEventTypes[type]
    let onceCallBackFun = function(event) {
      listerner.call(self, event)
      self.offHijackEvent(type, onceCallBackFun)
    }
    this.hijackEvent(type, onceCallBackFun)
    return !!listenerArr
  }

  /**
   * 移除劫持事件
   */
  offHijackEvent(type, listerner) {
    let listenerArr = this._hijackEventTypes[type]
    if (listenerArr) {
      let idx = listenerArr.indexOf(listerner)
      if (idx >= 0) {
        this._hijackEventTypes[type].splice(idx, 1)
      }
    }
    return !!listenerArr
  }
}
