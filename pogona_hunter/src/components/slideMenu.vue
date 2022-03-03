<template>
  <Slide style="z-index: 20;">
    <div>
      <form id='game-configuration' v-on:change="$emit('init')">
        <h1>Pogona Hunter</h1>

        <div class="row">
          <label for="bugType">Bug Type:</label>
          <select id="bugType" v-model="bugSettings.bugTypes" multiple="true">
            <option v-for="option in Object.keys(configOptions.bugTypes)" v-bind:value="option" v-bind:key="option">
              {{ configOptions.bugTypes[option].text }}
            </option>
          </select>
        </div>

        <div class="row">
          <label for="movementType">Movement Type:</label>
          <select id="movementType" v-model="bugSettings.movementType">
            <option v-for="option in Object.keys(configOptions.movementTypes)" v-bind:value="option"
                    v-bind:key="option">
              {{ configOptions.movementTypes[option] }}
            </option>
          </select>
        </div>

        <div class="row">
          <label for="numOfBugs">Number of Bugs: </label>
          <input v-model.number="bugSettings.numOfBugs" id="numOfBugs" type="number" style="width: 2em">
        </div>

        <div class="row">
          <label for="canvasHeight">Canvas Height: </label>
          <input v-model.number="canvasParams.height" id="canvasHeight" type="number" style="width: 4em">
        </div>

        <div class="row">
          <label for="time-in-edge">Time In Edge: </label>
          <input v-model.number="bugSettings.timeInEdge" id="time-in-edge" type="number" style="width: 4em">
        </div>

        <div class="row">
          <label for="time-between-bugs">Time Between Bugs: </label>
          <input v-model.number="bugSettings.timeBetweenBugs" id="time-between-bugs" type="number"
                 style="width: 4em">
        </div>

        <div class="row">
          <label for="blood-duration">Blood Duration: </label>
          <input v-model.number="bugSettings.bloodDuration" id="blood-duration" type="number" style="width: 4em">
        </div>

        <div class="row">
          <label for="radius-size">Bug Size: </label>
          <input v-model.number="bugSettings.bugSize" id="radius-size" type="number"
                 style="width: 3em">
        </div>

        <div class="row">
          <label for="speed">Speed: </label>
          <input v-model.number="bugSettings.speed" id="speed" type="number"
                 style="width: 3em">
        </div>

        <div class="row" v-if="bugSettings.movementType==='low_horizontal_noise'">
          <label for="noiseStartFraction">Noise Start Fraction: </label>
          <input v-model.number="bugSettings.noiseStartFraction" id="noiseStartFraction" type="number"
                 style="width: 3em" min="0" max="1" step="0.1">
        </div>

        <p>Written by Reggev Eyal</p>

      </form>

    </div>
  </Slide>
</template>

<script>
import { Slide } from 'vue-burger-menu'

export default {
  name: 'slide-menu',
  components: {Slide},
  props: {
    bugSettings: Object,
    configOptions: Object,
    canvasParams: Object
  }
}
</script>

<style scoped>

</style>