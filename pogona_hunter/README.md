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
1. **log/metric/touch** - screen touch logging
2. **log/metric/trial** - logging of all trial data, including bug_trajectory or media
3. **cmd/visual_app/console** - publish message to management UI console

