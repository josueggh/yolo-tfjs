# YOLO-TS

[![npm version](https://img.shields.io/npm/v/yolo-ts.svg)](https://www.npmjs.com/package/yolo-ts)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

YOLO-TS is a **TypeScript-based YOLO object detection library** powered by [TensorFlow.js](https://www.tensorflow.org/js). It enables real-time object detection on images, videos, and live webcam streams directly in the browser.

## Features

- **Easy Integration:** Simple API to quickly add object detection to your projects.
- **Real-Time Detection:** Process images, videos, or live webcam feeds in real time.
- **Customizable:** Configure labels, detection thresholds, and even a custom color palette.
- **Lightweight & Modular:** Written in TypeScript for robust type-checking and maintainability.
- **CDN-Ready:** Publish on npm and serve via CDNs like [jsDelivr](https://www.jsdelivr.com/) or [unpkg](https://unpkg.com/).
- **Tested with YOLO Models:** Compatible with YOLOv8 and YOLO11.

## Live Examples

Check out the following demos to see YOLO-TS in action:

(![Image Detection](doc/room.jpg)
(![Image Detection](doc/girl.jpg)

- 🖼️ **Image Detection:** [Live Demo](https://yolots-examples.vercel.app/)
- 🎞️ **Video Detection:** [Live Demo](https://yolots-examples.vercel.app/video.html)
- 📷 **Webcam Detection:** [Live Demo](https://yolots-examples.vercel.app/webcam.html)



## Installation

Install via npm:

```bash
npm install yolo-ts
```

Or load directly from a CDN (UMD build):
```html
<script src="https://cdn.jsdelivr.net/npm/yolo-ts@latest/dist/yolo.umd.js"></script>
```

**Note**: YOLO-TS has peer dependencies on TensorFlow.js and its WebGL backend. When using the UMD build, load these libraries before your YOLO-TS bundle:
```html
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-webgl@4.22.0/dist/tf-backend-webgl.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/yolo-ts@latest/dist/yolo.umd.js"></script>
```


## API Overview

YOLO-TS exposes a single class, YOLO, with the following primary methods:

**setup(options)**
Configure the model with custom settings (e.g., model URL, labels, colors, display filters, and score thresholds).

**loadModel()**
Loads the YOLO model from the specified URL. Returns a promise that resolves to the loaded model.

**detect(source, model, canvasRef, callback)**
Processes an image, video, or canvas element for object detection and renders bounding boxes on the provided canvas.

**detectVideo(videoSource, model, canvasRef)**
Continuously processes video frames for real-time detection.


## Contributing

Contributions are welcome! If you’d like to improve YOLO-TS, please fork the repository and submit a pull request.

- Repository: https://github.com/josueggh/yolo-ts
- Issues: https://github.com/josueggh/yolo-ts/issues


