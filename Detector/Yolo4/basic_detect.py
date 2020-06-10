import darknet

net = darknet.load_net_custom("yolo-obj.cfg".encode("ascii"), "yolo-obj_best.weights".encode("ascii"), 0, 1)
meta = darknet.load_meta("obj.data".encode("ascii"))
detections = darknet.detect(net, meta, "test.jpg".encode("ascii"), 0.9)

print(detections)
