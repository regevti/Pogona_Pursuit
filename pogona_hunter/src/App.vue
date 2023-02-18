<template>
    <div id="app">
      <router-view :key="$route.path"></router-view>
    </div>
</template>

<script>
  export default {
    name: 'App',
    components: {}
  }
</script>

<style>
    #app {
        font-family: 'Avenir', Helvetica, Arial, sans-serif;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
        text-align: center;
        color: #2c3e50;
        /*margin-top: 60px;*/
    }

    body {
      margin: 0;
    }
    .bm-item-list {
        margin-left: 0;
    }
</style>
