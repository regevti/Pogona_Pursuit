import WebSocketClient from './WebSocketClient'

export const webSocket = {

  install (Vue, connection, options) {
    const socketClient = new WebSocketClient(connection, options)
    socketClient.connect()
    Vue.prototype.$socketClient = socketClient
  }

}
