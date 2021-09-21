import http.server as http
import asyncio
import websockets
import socketserver
import multiprocessing
import cv2
import sys
from datetime import datetime as dt

from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from torch.autograd import Variable

# Keep track of our processes
PROCESSES = []

# Torch transform
p = transforms.Compose([transforms.Resize((96, 96)),
                        transforms.ToTensor(),
                        ])

# Torch model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
speech_detector = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=(3, 3)),
    nn.Conv2d(16, 64, kernel_size=(3, 3)),
    nn.ReLU(),
    nn.BatchNorm2d(64),
    nn.MaxPool2d((2, 2)),
    nn.Conv2d(64, 128, kernel_size=(3, 3)),
    nn.Conv2d(128, 256, kernel_size=(3, 3)),
    nn.ReLU(),
    nn.BatchNorm2d(256),
    nn.MaxPool2d((2, 2)),
    nn.Flatten(),
    nn.LazyLinear(512),
    nn.LazyLinear(1)
).to(device)
speech_detector.load_state_dict(torch.load("model2850.pt"))

# Face detection loading
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Is this working appropriately
def predict_image(image):
    image_tensor = p(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input = Variable(image_tensor)
    input = input.to(device)
    output = speech_detector(input)
    index = output.data.cpu().numpy() >= 0.7
    return index

def log(message):
    print("[LOG] " + str(dt.now()) + " - " + message)

def camera(man):
    log("Starting camera")
    vc = cv2.VideoCapture(0)

    if vc.isOpened():
        r, f = vc.read()
    else:
        r = False

    while r:
        cv2.waitKey(20)
        r, f = vc.read()

        # f = cv2.rotate(f, cv2.cv2.ROTATE_90_CLOCKWISE)
        faces = faceCascade.detectMultiScale(
                f,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

        image = ""
        if len(faces) == 0:
            image = f
        else:
            (x, y, w, h) = faces[0]
            if predict_image(Image.fromarray(f[y:y+h,x:x+w])):
                image = cv2.rectangle(
                    f, (x, y), (x+w, y+h), (0, 255, 0), 2)
            else:
                image = f
            print(predict_image(Image.fromarray(f[y:y+h,x:x+w])))

        # image = cv2.resize(image, (640, 480))
        # encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 65]
        man[0] = cv2.imencode('.jpg', image)[1]

# HTTP server handler
def server():
    server_address = ('0.0.0.0', 8000)
    if sys.version_info[1] < 7:
        class ThreadingHTTPServer(socketserver.ThreadingMixIn, http.HTTPServer):
            pass
        httpd = ThreadingHTTPServer(server_address, http.SimpleHTTPRequestHandler)
    else:
        httpd = http.ThreadingHTTPServer(server_address, http.SimpleHTTPRequestHandler)
    log("Server started")
    httpd.serve_forever()

def socket(man):
    # Will handle our websocket connections
    async def handler(websocket, path):
        log("Socket opened")
        try:
            while True:
                await asyncio.sleep(0.033) # 30 fps
                await websocket.send(man[0].tobytes())
        except websockets.exceptions.ConnectionClosed:
            log("Socket closed")

    log("Starting socket handler")
    # Create the awaitable object
    start_server = websockets.serve(ws_handler=handler, host='0.0.0.0', port=8585)
    # Start the server, add it to the event loop
    asyncio.get_event_loop().run_until_complete(start_server)
    # Registered our websocket connection handler, thus run event loop forever
    asyncio.get_event_loop().run_forever()


def main():
    # queue = multiprocessing.Queue()
    manager = multiprocessing.Manager()
    lst = manager.list()
    lst.append(None)
    # Host the page, creating the server
    http_server = multiprocessing.Process(target=server)
    # Set up our websocket handler
    socket_handler = multiprocessing.Process(target=socket, args=(lst,))
    # Set up our camera
    camera_handler = multiprocessing.Process(target=camera, args=(lst,))
    # Add 'em to our list
    PROCESSES.append(camera_handler)
    PROCESSES.append(http_server)
    PROCESSES.append(socket_handler)
    for p in PROCESSES:
        p.start()
    # Wait forever
    while True:
        pass

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        for p in PROCESSES:
            p.terminate()