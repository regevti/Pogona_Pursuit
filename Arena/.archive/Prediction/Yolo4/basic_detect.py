import darknet

net = darknet.load_net_custom("Prediction/Yolo4/yolo4_2306.cfg".encode("ascii"), "Prediction/Yolo4/yolo4_best_rgb.weights".encode("ascii"), 0, 1)
meta = darknet.load_meta("Prediction/Yolo4/obj.data".encode("ascii"))
detections = darknet.detect(net, meta, "test.jpg".encode("ascii"), 0.9)

print(detections)
