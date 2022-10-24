const WebSocket = require('ws');
const redis = require('redis');

// Configuration: adapt to your environment
const REDIS_SERVER = "redis://127.0.0.1:6379";
const WEB_SOCKET_PORT = 6380;
const SUBSCRIPTION_PATTERN = 'cmd/visual_app/*'

// Connect to Redis and subscribe to "app:notifications" channel
let publisher = redis.createClient(REDIS_SERVER);
publisher.on('error', (err) => console.error('Redis Client Error', err));
publisher.connect()

// Create & Start the WebSocket server
const wsServer = new WebSocket.Server({ port : WEB_SOCKET_PORT });

// create the redis subscriber
let subscriber = publisher.duplicate();
subscriber.connect()

// Register event for client connection
wsServer.on('connection', function connection(ws, req) {
    // broadcast on web socket when receving a Redis PUB/SUB Event
    subscriber.pSubscribe(SUBSCRIPTION_PATTERN, (payload, channel) => {
        ws.send(JSON.stringify({channel: channel, payload: payload}));
        // console.log(`Subscriber: ${channel} - ${payload}`);
    });

    ws.on('message', function message(event) {
        let parsedMessage = JSON.parse(event);
        let channel = parsedMessage.channel
        if (channel.startsWith('set/')) {
            channel = channel.replace('set/', '')
            publisher.set(channel, parsedMessage.payload)
            // console.log(`Set "${channel}" to "${parsedMessage.payload}"`)
        }
        else {
            publisher.publish(channel, parsedMessage.payload);
            // console.log(`Published to channel ${channel}: ${parsedMessage.payload}`)
        }
    })

    console.log("WebSocket server started at ws://localhost:"+ WEB_SOCKET_PORT);
});

process.stdin.resume();//so the program will not close instantly

function exitHandler(options, exitCode) {
    subscriber.pUnsubscribe(SUBSCRIPTION_PATTERN)
    subscriber.disconnect()
    publisher.disconnect()
    console.log('disconnected from redis clients')
    if (options.cleanup) console.log('clean');
    if (exitCode || exitCode === 0) console.log(exitCode);
    if (options.exit) process.exit();
}

//do something when app is closing
process.on('exit', exitHandler.bind(null,{cleanup:true}));

//catches ctrl+c event
process.on('SIGINT', exitHandler.bind(null, {exit:true}));

// catches "kill pid" (for example: nodemon restart)
process.on('SIGUSR1', exitHandler.bind(null, {exit:true}));
process.on('SIGUSR2', exitHandler.bind(null, {exit:true}));

//catches uncaught exceptions
process.on('uncaughtException', exitHandler.bind(null, {exit:true}));