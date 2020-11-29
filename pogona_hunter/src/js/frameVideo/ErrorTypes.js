const ErrorRet = {
  MEDIA_NO_AVAILABLE_SRC: {
    ret: -1,
    msg: '播放器没有检测到可用的视频地址。'
  },
  MEDIA_ERR_ABORTED: {
    ret: 1,
    msg: '视频数据加载过程中被中断。'
  },
  MEDIA_ERR_NETWORK: {
    ret: 2,
    msg: '由于网络问题造成加载视频失败。'
  },
  MEDIA_ERR_DECODE: {
    ret: 3,
    msg: '视频解码时发生错误。'
  },
  MEDIA_ERR_SRC_NOT_SUPPORTED: {
    ret: 4,
    msg: '视频因格式不支持或者服务器或网络的问题无法加载。'
  }
}

const RetMsgMap = {}
for (let key in ErrorRet) {
  const RetData = ErrorRet[key]
  RetMsgMap[RetData.ret] = RetData.msg
}

export default {
  ErrorRet,
  RetMsgMap
}
