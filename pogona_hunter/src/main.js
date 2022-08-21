/* eslint-disable */
// The Vue build version to load with the `import` command
// (runtime-only or standalone) has been set in webpack.base.conf with an alias.
import Vue from 'vue'
import App from './App'
import store from './store'
import {webSocket} from './js/websocket'
// import VueMqtt from 'vue-mqtt'

Vue.config.productionTip = false
Vue.use(webSocket, 'ws://127.0.0.1:6380', {
  reconnectEnabled: true,
  reconnectInterval: 5000 // time to reconnect in milliseconds
})
// Vue.use(VueMqtt, 'ws://localhost:9001')

/* eslint-disable no-new */
/* eslint-disable indent */
new Vue({
  el: '#app',
  store,
  components: { App },
  template: '<App/>'
})
