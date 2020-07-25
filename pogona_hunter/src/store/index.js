import Vue from 'vue'
import Vuex from 'vuex'

Vue.use(Vuex)

export default new Vuex.Store({
  state: {
    score: 0,
    bugTypeOptions: [
      {text: 'Cockroach', value: 'cockroach'},
      {text: 'Cricket', value: 'cricket'}
    ],
    numImagesPerBug: 3,
    speedRange: {
      min: -2,
      max: 2
    }
  },
  mutations: {
    increment (state) {
      state.score++
    }
  }
})