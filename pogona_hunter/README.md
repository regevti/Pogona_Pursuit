# pogona_hunter

> A Vue.js project

## Build Setup

``` bash
# install dependencies
npm install

# serve with hot reload at localhost:8080
npm run dev

# build for production with minification
npm run build

# build for production and view the bundle analyzer report
npm run build --report

# run unit tests
npm run unit

# run all tests
npm test
```

For a detailed explanation on how things work, check out the [guide](http://vuejs-templates.github.io/webpack/) and [docs for vue-loader](http://vuejs.github.io/vue-loader).

## Pub/Sub Channels
### Listeners
Websocket subscribed to the pattern: "cmd/visual_app/*"
1. **cmd/visual_app/hide_bugs** - hide all bugs from canvas
2. **cmd/visual_app/hide_media** - stop media
3. **cmd/visual_app/init_media** - start media
4. **cmd/visual_app/show_pogona** -
5. **cmd/visual_app/prediction** -
6. **cmd/visual_app/reload_app** - Reload the application
7. **cmd/visual_app/reward_given** - Reward was given in the arena

### Publishes
1. **event/log/video_frames**
2. **event/log/touch**
3. **event/command/end_app_wait**
4. **event/log/trajectory**
5. **event/log/experiment**
6. **event/log/trials_times**
