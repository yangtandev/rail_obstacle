from openvino.runtime import Core

ie = Core()
gpu_device = "GPU"
print(ie.available_devices)

