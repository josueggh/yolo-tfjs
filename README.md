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

![Image Detection](doc/room.jpg)
![Image Detection](doc/girl.jpg)

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


## Initial Configuration

The `setup` method in YOLO-TS allows for customization using the following configuration options:

```typescript
export interface YOLOConfig {
  modelUrl: string;
  labels?: string[]; // Optional, defaulting to COCO categories
  colors?: string[]; // Optional, custom colors for label display
  displayLabels?: Set<string> | null; // Optional, filter specific labels to be displayed
  scoreThreshold: number;
  scoreThreshold: number;
  boxLineWidth: number;
  boxLabels: boolean;
}
```
## API Overview

YOLO-TS exposes a single class, YOLO, with the following primary methods:

**setup(options)**
Configure the model with custom settings (e.g., model URL, labels, colors, display filters, and score thresholds).
```javascript
yolo.setup({
  modelUrl: "model/model.json",
  labels?: ["person", "car", "dog"],
  colors?: ["#FF0000", "#00FF00"],
  displayLabels?: new Set(["person", "dog"]),
  scoreThreshold: 0.3,
  boxLineWidth: 10,
  boxLabels: true,
});
```

**loadModel()**
Loads the YOLO model from the specified URL. Returns a promise that resolves to the loaded model.
```javascript
yolo.loadModel().then((model) => {
    console.log("Model loaded!", model)
  });
```

**detect(source, model, canvasRef, callback)**
Processes an image, video, or canvas element for object detection and renders bounding boxes on the provided canvas.
```javascript
yolo.detect(imageElement, model, canvas, (detections) => {
  console.log(detections);
});
```


**detectVideo(videoSource, model, canvasRef)**
Continuously processes video frames for real-time detection.
```javascript
yolo.detectVideo(videoElement, model, canvas);
```

---
## Exporting a YOLO Model for TensorFlow.js

If you have a trained YOLO model and want to use it with YOLO-TS, you need to export it in TensorFlow.js format. Here's how you can do it:

```python
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("yolo11n.pt")

# Export the model to TensorFlow.js format
model.export(format="tfjs")
```

## Contributing

Contributions are welcome! If you’d like to improve YOLO-TS, please fork the repository and submit a pull request.

- Repository: https://github.com/josueggh/yolo-ts
- Issues: https://github.com/josueggh/yolo-ts/issues


