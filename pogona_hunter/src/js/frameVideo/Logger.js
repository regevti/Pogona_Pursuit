let needDebug = false

export function setDebug(open) {
  needDebug = !!open
}

const Logger = {
  log: function() {
    needDebug && console.log.apply(this, arguments)
  }
}

export default Logger
