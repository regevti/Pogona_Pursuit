/* eslint-disable */
// The Vue build version to load with the `import` command
// (runtime-only or standalone) has been set in webpack.base.conf with an alias.
import Vue from 'vue'
import VueRouter from 'vue-router'
import App from './App'
import tunnelBoard from './components/boards/tunnelBoard.vue'
import holesBoard from './components/boards/holesBoard.vue'
import store from './store'
import {webSocket} from './js/websocket'
// import VueMqtt from 'vue-mqtt'

Vue.use(VueRouter)
const PageNotFound = { template: '<h1>Page not found</h1>' }
const routes = [
    { path: '/', component: PageNotFound },
    { path: '/tunnel', component: tunnelBoard},
    { path: '/holes', component: holesBoard}
]
const router = new VueRouter({
  mode: process.env.ROUTER_MODE,
  routes
})

Vue.config.productionTip = false
Vue.use(webSocket, 'ws://127.0.0.1:6380', {
  reconnectEnabled: true,
  reconnectInterval: 5000 // time to reconnect in milliseconds
})
// Vue.use(VueMqtt, 'ws://localhost:9001')

/* eslint-disable no-new */
/* eslint-disable indent */
new Vue({
  router,
  store,
  render: function(createElement){
        return createElement(App)
    }
}).$mount('#app')
