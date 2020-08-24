import darknet

net = darknet.load_net_custom("Detector/yolo4/Yolo-obj.cfg".encode("ascii"), "Detector/Yolo4/yolo-obj_best.weights".encode("ascii"), 0, 1)
meta = darknet.load_meta("Detector/yolo4/obj.data".encode("ascii"))
detections = darknet.detect(net, meta, "test.jpg".encode("ascii"), 0.9)

print(detections)
