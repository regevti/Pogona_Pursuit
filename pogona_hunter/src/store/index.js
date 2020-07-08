import Vue from 'vue'
import Vuex from 'vuex'

Vue.use(Vuex)

export default new Vuex.Store({
  state: {
    score: 0,
    timeBetweenTrial: 2000,
    timeInEdge: 2000,
    bloodDuration: 2000,
    bugTypeOptions: [
      {text: 'Cockroach', value: 'cockroach'},
      {text: 'Cricket', value: 'cricket'}
    ],
    numImagesPerBug: 3,
    speedRange: {
      min: -2,
      max: 2
    },
    radiusRange: {
      min: 120,
      max: 140
    },
    canvasParams: {
      width: window.innerWidth,
      height: window.innerHeight
    }
  },
  mutations: {
    increment (state) {
      state.score++
    }
  }
})