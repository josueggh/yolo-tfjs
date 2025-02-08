import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgl";


/**
 * Interface representing the configuration for the YOLO object detection.
 */
export interface YOLOConfig {
  /**
   * The URL pointing to the TensorFlow.js YOLO model JSON file.
   * This model should be compatible with TensorFlow.js for in-browser execution.
   *
   * @example "https://example.com/model/model.json"
   */
  modelUrl: string;
  /**
   * Optional array of labels for object detection.
   * If omitted, the model will default to using COCO dataset labels.
   *
   * @example ["person", "car", "dog"]
   */
  labels: string[];
  /**
   * Optional array of colors for detected object labels.
   * The color mapping corresponds to the label array.
   * If omitted, a default color scheme is used.
   *
   * @example ["#FF0000", "#00FF00", "#0000FF"]
   */
  colors: string[];
  /**
   * Optional set of labels to filter displayed objects.
   * If null, all detected objects will be displayed.
   *
   * @example new Set(["person", "dog"])
   */
  displayLabels: Set<string> | null;
  /**
   * The minimum confidence score threshold for displaying detected objects.
   * Objects with a lower score will be ignored.
   *
   * @default 0.5
   * @example 0.3
   */
  scoreThreshold: number;
  /**
   * The width of the bounding box stroke for detected objects.
   * Adjusting this value controls how thick the detection box appears.
   *
   * @default 2
   * @example 10
   */
  boxLineWidth: number;
}


/**
 * Interface representing the loaded YOLO model.
 */
export interface YOLOModel {
  net: tf.GraphModel;
  inputShape: number[];
}


/**
 * YOLO class for object detection using TensorFlow.js.
 */
class YOLO {
  /**
   * Default configuration for the YOLO model.
   */
  public config: YOLOConfig = {
    modelUrl: "",
    labels: [
      "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
      "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
      "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
      "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
      "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
      "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
      "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
      "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
      "scissors", "teddy bear", "hair drier", "toothbrush"
    ],
    colors: [
      "#FF6B6B", "#FFA372", "#FFDC64", "#A3E635", "#66D3FA", "#82AAFF", "#C084FC", "#F472B6", "#FF9F43", "#FFB74D",
      "#AED581", "#4DD0E1", "#4FC3F7", "#9575CD", "#F06292", "#BA68C8", "#FF8A65", "#FFD54F", "#81C784", "#64B5F6",
    ],
    displayLabels: null,
    scoreThreshold: 0.5,
    boxLineWidth: 2,
  };

  /**
   * Sets up the YOLO configuration.
   *
   * @param options - Partial configuration options to override the defaults.
   *                  `displayLabels` can be provided as an array of strings.
   */
  public setup(options: Partial<Omit<YOLOConfig, "displayLabels">> & { displayLabels?: string[] }): void {
    if (options.modelUrl) this.config.modelUrl = options.modelUrl;
    if (options.labels) this.config.labels = options.labels;
    if (options.colors) this.config.colors = options.colors;
    if (options.displayLabels) this.config.displayLabels = new Set(options.displayLabels);
    if (options.scoreThreshold !== undefined) this.config.scoreThreshold = options.scoreThreshold;
    if (options.boxLineWidth) this.config.boxLineWidth = options.boxLineWidth;
  }

  /**
   * Loads the YOLO model from the specified URL.
   *
   * @returns A promise that resolves to the loaded YOLO model, or null if an error occurs.
   */
  public async loadModel(): Promise<YOLOModel | null> {
    await tf.ready();
    try {
      const net = await tf.loadGraphModel(this.config.modelUrl);
      // Cast net.inputs[0].shape as number[] so tf.ones receives a valid array.
      const dummyInput = tf.ones(net.inputs[0].shape as number[]);
      const warmupResults = net.execute(dummyInput);
      tf.dispose([dummyInput, warmupResults]);
      return {net, inputShape: net.inputs[0].shape as number[]};
    } catch (error) {
      console.error("Error loading model:", error);
      return null;
    }
  }

  /**
   * Detects objects in the given source (image, video, or canvas) and draws bounding boxes on the canvas.
   *
   * @param source - The source element (HTMLImageElement, HTMLVideoElement, or HTMLCanvasElement).
   * @param model - The loaded YOLO model.
   * @param canvasRef - The canvas element where the detection boxes will be rendered.
   * @param callback - A callback function that receives detection data (boxes, scores, classes).
   */
  public async detect(
    source: HTMLImageElement | HTMLVideoElement | HTMLCanvasElement,
    model: YOLOModel,
    canvasRef: HTMLCanvasElement,
    callback: (detections: {
      boxes: number[];
      scores: number[];
      classes: number[];
    }) => void = () => {}
  ): Promise<void> {
    // Extract model dimensions (assumed square, e.g. 640 x 640)
    const [modelWidth, modelHeight] = model.inputShape.slice(1, 3);
    tf.engine().startScope();

    // Preprocess: get both the input tensor and letterbox info.
    const [input, {scale, dx, dy, origWidth, origHeight}] = this.preprocess(source, modelWidth, modelHeight);

    // Run inference and process outputs.
    const res = model.net.execute(input) as tf.Tensor;
    const transposedRes = res.transpose([0, 2, 1]);

    const boxes = tf.tidy(() => {
      const xCenter = transposedRes.slice([0, 0, 0], [-1, -1, 1]);
      const yCenter = transposedRes.slice([0, 0, 1], [-1, -1, 1]);
      const width = transposedRes.slice([0, 0, 2], [-1, -1, 1]);
      const height = transposedRes.slice([0, 0, 3], [-1, -1, 1]);

      const x1 = tf.sub(xCenter, tf.div(width, 2));
      const y1 = tf.sub(yCenter, tf.div(height, 2));
      const x2 = tf.add(xCenter, tf.div(width, 2));
      const y2 = tf.add(yCenter, tf.div(height, 2));

      return tf.concat([y1, x1, y2, x2], 2).squeeze();
    });

    const [scores, classes] = tf.tidy(() => {
      const rawScores = transposedRes
        .slice([0, 0, 4], ([-1, -1, this.config.labels.length] as [number, number, number]))
        .squeeze([0]);
      return [rawScores.max(1), rawScores.argMax(1)];
    });

    // Perform non-max suppression
    const nmsIndices = await tf.image.nonMaxSuppressionAsync(
      boxes as tf.Tensor2D,
      scores as tf.Tensor1D,
      500,
      0.45,
      0.2
    );

    const boxesData = boxes.gather(nmsIndices, 0).dataSync();
    const scoresData = scores.gather(nmsIndices, 0).dataSync();
    const classesData = classes.gather(nmsIndices, 0).dataSync();

    // Adjust the box coordinates back to the original image.
    const adjustedBoxes: number[] = [];
    for (let i = 0; i < scoresData.length; i++) {
      let [y1, x1, y2, x2] = boxesData.slice(i * 4, (i + 1) * 4);
      // Remove the letterbox padding and revert the scaling.
      x1 = Math.max(0, (x1 - dx) / scale);
      x2 = Math.min(origWidth, (x2 - dx) / scale);
      y1 = Math.max(0, (y1 - dy) / scale);
      y2 = Math.min(origHeight, (y2 - dy) / scale);
      adjustedBoxes.push(y1, x1, y2, x2);
    }

    // Update the canvas size to match the original image
    if (source instanceof HTMLImageElement) {
      canvasRef.width = origWidth;
      canvasRef.height = origHeight;
    } else {
      canvasRef.width = origWidth;
      canvasRef.height = origHeight;
    }

    // Render the boxes on the canvas using the adjusted coordinates.
    this.renderBoxes(canvasRef, adjustedBoxes, scoresData, classesData, [1, 1]);

    // Pass detection data back via the callback.
    callback({boxes: adjustedBoxes, scores: Array.from(scoresData), classes: Array.from(classesData)});

    tf.dispose([res, transposedRes, boxes, scores, classes, nmsIndices]);
    tf.engine().endScope();
  }

  /**
   * Continuously detects objects in a video source and updates the canvas in real time.
   *
   * @param videoSource - The HTMLVideoElement that provides the video feed.
   * @param model - The loaded YOLO model.
   * @param canvasRef - The canvas element where detection boxes will be rendered.
   */
  public detectVideo(videoSource: HTMLVideoElement, model: YOLOModel, canvasRef: HTMLCanvasElement): void {
    const detectFrame = async () => {
      if (videoSource.videoWidth === 0 && videoSource.srcObject === null) {
        const ctx = canvasRef.getContext("2d");
        if (ctx) ctx.clearRect(0, 0, canvasRef.width, canvasRef.height);
        requestAnimationFrame(detectFrame);
        return;
      }
      await this.detect(videoSource, model, canvasRef, () => {
        requestAnimationFrame(detectFrame);
      });
    };
    detectFrame();
  }


  /**
   * Preprocesses the source image/video/canvas by resizing it with letterboxing
   * to maintain aspect ratio.
   *
   * @param source - The source element (HTMLImageElement, HTMLVideoElement, or HTMLCanvasElement).
   * @param modelWidth - The target width for the model input.
   * @param modelHeight - The target height for the model input.
   * @returns A tuple containing the preprocessed input tensor and an object with letterbox details:
   *          - scale: The scaling factor applied.
   *          - dx: Horizontal padding.
   *          - dy: Vertical padding.
   *          - origWidth: Original source width.
   *          - origHeight: Original source height.
   */
  public preprocess(
    source: HTMLImageElement | HTMLVideoElement | HTMLCanvasElement,
    modelWidth: number,
    modelHeight: number
  ): [tf.Tensor, { scale: number; dx: number; dy: number; origWidth: number; origHeight: number }] {
    // Convert source to tensor
    const imgTensor = tf.browser.fromPixels(source);
    const [origHeight, origWidth] = imgTensor.shape.slice(0, 2);

    // Compute the scaling factor (preserve aspect ratio)
    const scale = Math.min(modelWidth / origWidth, modelHeight / origHeight);
    const newWidth = Math.round(origWidth * scale);
    const newHeight = Math.round(origHeight * scale);

    // Resize image with the uniform scale.
    const resized = tf.image.resizeBilinear(imgTensor, [newHeight, newWidth]).div(255.0);

    // Compute padding (center the resized image in the model input)
    const dx = Math.floor((modelWidth - newWidth) / 2);
    const dy = Math.floor((modelHeight - newHeight) / 2);

    // Pad the resized image so that it becomes modelWidth x modelHeight
    const padded = tf.tidy(() => {
      return tf.pad(
        resized,
        [
          [dy, modelHeight - newHeight - dy],
          [dx, modelWidth - newWidth - dx],
          [0, 0],
        ]
      );
    });

    // Expand dims to add the batch dimension
    const input = padded.expandDims(0);

    // Dispose intermediate tensors if needed.
    tf.dispose([imgTensor, resized, padded]);

    return [input, {scale, dx, dy, origWidth, origHeight}];
  }


  /**
   * Renders detection boxes and labels on the provided canvas.
   *
   * @param canvasRef - The canvas element where detections will be drawn.
   * @param boxesData - Array of box coordinates (assumed to be in the original image coordinate system).
   * @param scoresData - Array of detection confidence scores.
   * @param classesData - Array of detected class indices.
   * @param ratios - Scaling ratios (typically [1, 1] if boxes are already in original image coordinates).
   */
  public renderBoxes(
    canvasRef: HTMLCanvasElement,
    boxesData: number[] | Float32Array,
    scoresData: number[] | Float32Array,
    classesData: number[] | Float32Array,
    ratios: [number, number],
  ): void {
    const ctx = canvasRef.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvasRef.width, canvasRef.height);
    ctx.font = "16px Arial";
    ctx.textBaseline = "top";

    const [xRatio, yRatio] = ratios;

    for (let i = 0; i < scoresData.length; i++) {
      if (scoresData[i] < this.config.scoreThreshold) continue;
      const classLabel = this.config.labels[classesData[i]];
      if (this.config.displayLabels && !this.config.displayLabels.has(classLabel)) continue;
      const color = this.config.colors[classesData[i] % this.config.colors.length];
      const scorePercentage = (scoresData[i] * 100).toFixed(1);
      const text = `${classLabel} - ${scorePercentage}%`;

      let [y1, x1, y2, x2] = boxesData.slice(i * 4, (i + 1) * 4);
      const boxWidth = x2 - x1;
      const boxHeight = y2 - y1;

      ctx.strokeStyle = color;
      ctx.lineWidth = this.config.boxLineWidth;
      ctx.strokeRect(x1, y1, boxWidth, boxHeight);
      const textWidth = ctx.measureText(text).width;
      const textHeight = 16;
      ctx.fillStyle = color;
      ctx.fillRect(x1, y1 - textHeight, textWidth + 4, textHeight);
      ctx.fillStyle = "#ffffff";
      ctx.fillText(text, x1 + 2, y1 - textHeight);
    }
  }
}

export default YOLO;
