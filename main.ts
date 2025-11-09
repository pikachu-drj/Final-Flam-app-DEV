import "./style.css";
import { SelectionManager } from "./ui-utils.js";
import { EvaluationManager } from "./evaluation-manager.js";

export interface Point {
  x: number;
  y: number;
}

export interface DetectedShape {
  type: "circle" | "triangle" | "rectangle" | "pentagon" | "star";
  confidence: number;
  boundingBox: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  center: Point;
  area: number;
}

export interface DetectionResult {
  shapes: DetectedShape[];
  processingTime: number;
  imageWidth: number;
  imageHeight: number;
}

export class ShapeDetector {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d")!;
  }

  /**
   * MAIN ALGORITHM TO IMPLEMENT
   * Method for detecting shapes in an image
   * @param imageData - ImageData from canvas
   * @returns Promise<DetectionResult> - Detection results
   *
   * Implemented: grayscale -> blur -> threshold -> components -> contour -> polygon approx -> classification
   */
  async detectShapes(imageData: ImageData): Promise<DetectionResult> {
    const startTime = performance.now();

    const w = imageData.width;
    const h = imageData.height;
    const rgba = imageData.data;

    // 1) Convert to grayscale
    const gray = new Uint8ClampedArray(w * h);
    for (let i = 0; i < w * h; i++) {
      const r = rgba[i * 4];
      const g = rgba[i * 4 + 1];
      const b = rgba[i * 4 + 2];
      // luminance
      gray[i] = (0.299 * r + 0.587 * g + 0.114 * b) | 0;
    }

    // 2) Simple blur (box blur) to reduce noise
    function boxBlur(src: Uint8ClampedArray, ww: number, hh: number, radius = 1) {
      const dst = new Uint8ClampedArray(ww * hh);
      const size = (2 * radius + 1) * (2 * radius + 1);
      for (let y = 0; y < hh; y++) {
        for (let x = 0; x < ww; x++) {
          let sum = 0;
          let count = 0;
          for (let ky = -radius; ky <= radius; ky++) {
            const yy = y + ky;
            if (yy < 0 || yy >= hh) continue;
            for (let kx = -radius; kx <= radius; kx++) {
              const xx = x + kx;
              if (xx < 0 || xx >= ww) continue;
              sum += src[yy * ww + xx];
              count++;
            }
          }
          dst[y * ww + x] = (sum / count) | 0;
        }
      }
      return dst;
    }

    const blurred = boxBlur(gray, w, h, 1);

    // 3) Global threshold (Otsu would be nicer; here we compute a quick histogram and pick a threshold)
    let threshold = 128;
    {
      const hist = new Array(256).fill(0);
      for (let i = 0; i < blurred.length; i++) hist[blurred[i]]++;
      // approximate Otsu (fast)
      let total = w * h;
      let sum = 0;
      for (let t = 0; t < 256; t++) sum += t * hist[t];
      let sumB = 0;
      let wB = 0;
      let maxVar = 0;
      let bestT = 0;
      for (let t = 0; t < 256; t++) {
        wB += hist[t];
        if (wB === 0) continue;
        const wF = total - wB;
        if (wF === 0) break;
        sumB += t * hist[t];
        const mB = sumB / wB;
        const mF = (sum - sumB) / wF;
        const varBetween = wB * wF * (mB - mF) * (mB - mF);
        if (varBetween > maxVar) {
          maxVar = varBetween;
          bestT = t;
        }
      }
      threshold = bestT || 128;
    }

    // 4) Binary image (foreground = shapes; assume shapes are darker or darker/colored -> invert if needed)
    const binary = new Uint8ClampedArray(w * h);
    for (let i = 0; i < w * h; i++) {
      binary[i] = blurred[i] < threshold ? 1 : 0;
    }

    // small morphological clean-up: remove isolated pixels (simple)
    function cleanSmallNoise(bin: Uint8ClampedArray, ww: number, hh: number) {
      const out = bin.slice();
      for (let y = 1; y < hh - 1; y++) {
        for (let x = 1; x < ww - 1; x++) {
          const i = y * ww + x;
          if (bin[i] === 1) {
            // count neighbors
            let nb = 0;
            for (let yy = -1; yy <= 1; yy++)
              for (let xx = -1; xx <= 1; xx++)
                if (bin[(y + yy) * ww + (x + xx)]) nb++;
            if (nb <= 2) out[i] = 0;
          }
        }
      }
      return out;
    }
    const cleaned = cleanSmallNoise(binary, w, h);

    // 5) Connected components (BFS) to extract blobs
    const labels = new Int32Array(w * h).fill(0);
    let currentLabel = 0;
    const components: { pixels: number[] }[] = [];

    const neighbors4 = [1, -1, w, -w];
    for (let i = 0; i < w * h; i++) {
      if (cleaned[i] === 1 && labels[i] === 0) {
        currentLabel++;
        const q: number[] = [i];
        labels[i] = currentLabel;
        const compPixels = [i];
        while (q.length) {
          const p = q.pop()!;
          const py = Math.floor(p / w);
          const px = p - py * w;
          // 4-neighbor BFS
          if (px + 1 < w) {
            const ni = p + 1;
            if (cleaned[ni] === 1 && labels[ni] === 0) {
              labels[ni] = currentLabel;
              q.push(ni);
              compPixels.push(ni);
            }
          }
          if (px - 1 >= 0) {
            const ni = p - 1;
            if (cleaned[ni] === 1 && labels[ni] === 0) {
              labels[ni] = currentLabel;
              q.push(ni);
              compPixels.push(ni);
            }
          }
          if (py + 1 < h) {
            const ni = p + w;
            if (cleaned[ni] === 1 && labels[ni] === 0) {
              labels[ni] = currentLabel;
              q.push(ni);
              compPixels.push(ni);
            }
          }
          if (py - 1 >= 0) {
            const ni = p - w;
            if (cleaned[ni] === 1 && labels[ni] === 0) {
              labels[ni] = currentLabel;
              q.push(ni);
              compPixels.push(ni);
            }
          }
        }
        components.push({ pixels: compPixels });
      }
    }

    // Filter out very small components
    const minAreaPx = Math.max(20, Math.round((w * h) * 0.0005)); // heuristic
    const filteredComps = components.filter((c) => c.pixels.length >= minAreaPx);

    // Helper: find boundary points by checking neighbors
    function extractBoundaryOrdered(labelIndex: number, compPixels: number[], labArr: Int32Array, ww: number, hh: number) {
      // Moore-Neighbor tracing (simple version)
      const set = new Set(compPixels);
      // find starting pixel: leftmost-topmost
      let start = compPixels[0];
      for (const p of compPixels) {
        const py = Math.floor(p / ww);
        const px = p - py * ww;
        const sPy = Math.floor(start / ww);
        const sPx = start - sPy * ww;
        if (py < sPy || (py === sPy && px < sPx)) start = p;
      }
      const contour: Point[] = [];
      let current = start;
      let prevDir = 6; // start search direction (arbitrary)
      const dirs = [
        [-1, 0], // left
        [-1, -1],
        [0, -1], // up
        [1, -1],
        [1, 0], // right
        [1, 1],
        [0, 1], // down
        [-1, 1],
      ];
      function inComp(x: number, y: number) {
        if (x < 0 || x >= ww || y < 0 || y >= hh) return false;
        return set.has(y * ww + x);
      }
      // start tracing
      let safety = 0;
      let done = false;
      let cx = start % ww;
      let cy = Math.floor(start / ww);
      contour.push({ x: cx, y: cy });
      while (!done && safety++ < 100000) {
        let found = false;
        // search neighbors starting from prevDir-3 mod 8 (per Moore)
        let startDir = (prevDir + 5) % 8;
        for (let k = 0; k < 8; k++) {
          const dir = (startDir + k) % 8;
          const nx = cx + dirs[dir][0];
          const ny = cy + dirs[dir][1];
          if (inComp(nx, ny)) {
            // move
            cx = nx;
            cy = ny;
            contour.push({ x: cx, y: cy });
            prevDir = dir;
            found = true;
            break;
          }
        }
        if (!found) {
          // isolated single point? break
          break;
        }
        if (cx === start % ww && cy === Math.floor(start / ww) && contour.length > 1) {
          done = true;
        }
      }
      // make unique (remove repeated tail)
      // remove duplicates that are adjacent
      const cleanedContour: Point[] = [];
      for (let i = 0; i < contour.length; i++) {
        const p = contour[i];
        const last = cleanedContour[cleanedContour.length - 1];
        if (!last || last.x !== p.x || last.y !== p.y) cleanedContour.push(p);
      }
      return cleanedContour;
    }

    // Ramer-Douglas-Peucker polygonal approximation
    function rdp(points: Point[], eps: number): Point[] {
      if (points.length < 3) return points.slice();
      const sq = (a: number) => a * a;
      function perpendicularDistance(pt: Point, a: Point, b: Point) {
        const A = pt.x - a.x;
        const B = pt.y - a.y;
        const C = b.x - a.x;
        const D = b.y - a.y;
        const dot = A * C + B * D;
        const lenSq = C * C + D * D;
        let param = -1;
        if (lenSq !== 0) param = dot / lenSq;
        let xx, yy;
        if (param < 0) {
          xx = a.x;
          yy = a.y;
        } else if (param > 1) {
          xx = b.x;
          yy = b.y;
        } else {
          xx = a.x + param * C;
          yy = a.y + param * D;
        }
        return Math.sqrt(sq(pt.x - xx) + sq(pt.y - yy));
      }

      let dmax = 0;
      let index = 0;
      for (let i = 1; i < points.length - 1; i++) {
        const d = perpendicularDistance(points[i], points[0], points[points.length - 1]);
        if (d > dmax) {
          index = i;
          dmax = d;
        }
      }
      if (dmax > eps) {
        const rec1 = rdp(points.slice(0, index + 1), eps);
        const rec2 = rdp(points.slice(index), eps);
        return rec1.slice(0, rec1.length - 1).concat(rec2);
      } else {
        return [points[0], points[points.length - 1]];
      }
    }

    // Compute perimeter from ordered contour
    function contourPerimeter(contour: Point[]) {
      let p = 0;
      for (let i = 0; i < contour.length; i++) {
        const a = contour[i];
        const b = contour[(i + 1) % contour.length];
        const dx = a.x - b.x;
        const dy = a.y - b.y;
        p += Math.hypot(dx, dy);
      }
      return p;
    }

    // compute centroid
    function centroidFromPixels(pixels: number[], ww: number) {
      let sx = 0;
      let sy = 0;
      for (const p of pixels) {
        const y = Math.floor(p / ww);
        const x = p - y * ww;
        sx += x;
        sy += y;
      }
      return { x: sx / pixels.length, y: sy / pixels.length };
    }

    // 6) For each component, extract boundary, approximate polygon, classify
    const shapes: DetectedShape[] = [];

    for (let li = 0; li < filteredComps.length; li++) {
      const comp = filteredComps[li];
      const px = comp.pixels;
      const area = px.length;
      // bounding box
      let minx = w, miny = h, maxx = 0, maxy = 0;
      for (const p of px) {
        const y = Math.floor(p / w);
        const x = p - y * w;
        if (x < minx) minx = x;
        if (x > maxx) maxx = x;
        if (y < miny) miny = y;
        if (y > maxy) maxy = y;
      }
      const bw = maxx - minx + 1;
      const bh = maxy - miny + 1;
      // get ordered contour (scaled limited to bounding box area for speed)
      const contour = extractBoundaryOrdered(li + 1, px, labels, w, h);
      if (contour.length < 6) {
        // too small to determine
        continue;
      }

      // polygon approximation: epsilon relative to bounding box size/perimeter
      const per = contourPerimeter(contour) || 1;
      const eps = Math.max(1, per * 0.02); // 2% of perimeter
      const poly = rdp(contour, eps);

      // compute circularity: 4πA / P²
      const circularity = (4 * Math.PI * area) / (per * per);

      // simple concavity heuristic: compare poly area to pixel area
      // classify by number of vertices first, then refine
      const vcount = poly.length;

      // centroid
      const centroid = centroidFromPixels(px, w);

      // confidence heuristics:
      let type: DetectedShape["type"] = "rectangle";
      let confidence = 0.5;

      // circle detection: high circularity (near 1)
      if (circularity > 0.70 && Math.abs(bw - bh) / Math.max(bw, bh) < 0.4) {
        type = "circle";
        // confidence increases with circularity and how compact bounding box is
        confidence = Math.min(0.99, 0.5 + (circularity - 0.7) * 2.5);
      } else {
        // polygon-based classification
        // sanitize vertex count: polygon might include duplicate endpoints => handle
        const uniquePoly = poly.filter((p, i, arr) => {
          const prev = arr[(i - 1 + arr.length) % arr.length];
          return !(p.x === prev.x && p.y === prev.y);
        });

        const n = uniquePoly.length;

        if (n <= 3) {
          type = "triangle";
          confidence = 0.8;
        } else if (n === 4) {
          // check rectangle by angle approx
          function angle(a: Point, b: Point, c: Point) {
            const ux = a.x - b.x;
            const uy = a.y - b.y;
            const vx = c.x - b.x;
            const vy = c.y - b.y;
            const dot = ux * vx + uy * vy;
            const lu = Math.hypot(ux, uy);
            const lv = Math.hypot(vx, vy);
            if (lu * lv === 0) return 0;
            const cos = dot / (lu * lv);
            return Math.acos(Math.max(-1, Math.min(1, cos)));
          }
          let rightAngles = 0;
          for (let i = 0; i < 4; i++) {
            const a = uniquePoly[(i + 3) % 4];
            const b = uniquePoly[i];
            const c = uniquePoly[(i + 1) % 4];
            const ang = (angle(a, b, c) * 180) / Math.PI;
            if (Math.abs(ang - 90) < 25) rightAngles++;
          }
          if (rightAngles >= 2) {
            type = "rectangle";
            confidence = 0.6 + (rightAngles / 4) * 0.4; // up to 1.0
          } else {
            // maybe quadrilateral; treat as rectangle with lower confidence
            type = "rectangle";
            confidence = 0.5;
          }
        } else if (n === 5) {
          type = "pentagon";
          confidence = 0.75;
        } else if (n > 5 && n <= 8) {
          // could be noisy polygon (circle-like) or star/complex shape
          // check concavity count (simple check: angle sign changes)
          function cross(a: Point, b: Point, c: Point) {
            return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
          }
          let sign = 0;
          let concaveCount = 0;
          for (let i = 0; i < uniquePoly.length; i++) {
            const a = uniquePoly[i];
            const b = uniquePoly[(i + 1) % uniquePoly.length];
            const c = uniquePoly[(i + 2) % uniquePoly.length];
            const cr = cross(a, b, c);
            const s = Math.sign(cr);
            if (s !== 0) {
              if (sign === 0) sign = s;
              else if (s !== sign) concaveCount++;
            }
          }
          if (concaveCount >= 2) {
            type = "star";
            confidence = 0.75;
          } else {
            // treat as circle-ish polygon
            if (circularity > 0.45) {
              type = "circle";
              confidence = Math.min(0.9, 0.5 + (circularity - 0.45) * 1.2);
            } else {
              type = "star";
              confidence = 0.5;
            }
          }
        } else {
          // many vertices -> likely star or circle
          if (circularity > 0.5) {
            type = "circle";
            confidence = Math.min(0.95, 0.5 + (circularity - 0.5) * 1.0);
          } else {
            // many vertices but low circularity -> star or complex; choose star
            type = "star";
            confidence = 0.55;
          }
        }
      }

      // clamp confidence
      if (confidence < 0) confidence = 0;
      if (confidence > 1) confidence = 1;

      shapes.push({
        type,
        confidence,
        boundingBox: {
          x: minx,
          y: miny,
          width: bw,
          height: bh,
        },
        center: { x: centroid.x, y: centroid.y },
        area,
      });
    }

    const processingTime = performance.now() - startTime;

    // Debug: draw detections on canvas overlay (optional)
    // NOTE: Do not permanently affect main canvas; create a copy and draw to help debugging
    try {
      // draw overlay in the canvas for quick visual feedback
      const debugCtx = this.ctx;
      debugCtx.save();
      debugCtx.strokeStyle = "red";
      debugCtx.lineWidth = Math.max(1, Math.round(Math.min(w, h) / 200));
      debugCtx.fillStyle = "rgba(255,0,0,0.15)";
      for (const s of shapes) {
        debugCtx.beginPath();
        debugCtx.rect(s.boundingBox.x, s.boundingBox.y, s.boundingBox.width, s.boundingBox.height);
        debugCtx.stroke();
        debugCtx.fillRect(s.boundingBox.x, s.boundingBox.y, 4, 4);
        // center marker
        debugCtx.beginPath();
        debugCtx.arc(s.center.x, s.center.y, 3, 0, Math.PI * 2);
        debugCtx.fill();
        debugCtx.fillText(`${s.type} ${(s.confidence * 100).toFixed(0)}%`, s.boundingBox.x, s.boundingBox.y - 4);
      }
      debugCtx.restore();
    } catch (e) {
      // ignore drawing errors on some contexts
    }

    return {
      shapes,
      processingTime,
      imageWidth: w,
      imageHeight: h,
    };
  }

  loadImage(file: File): Promise<ImageData> {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        this.canvas.width = img.width;
        this.canvas.height = img.height;
        this.ctx.drawImage(img, 0, 0);
        const imageData = this.ctx.getImageData(0, 0, img.width, img.height);
        resolve(imageData);
      };
      img.onerror = reject;
      img.src = URL.createObjectURL(file);
    });
  }
}

class ShapeDetectionApp {
  private detector: ShapeDetector;
  private imageInput: HTMLInputElement;
  private resultsDiv: HTMLDivElement;
  private testImagesDiv: HTMLDivElement;
  private evaluateButton: HTMLButtonElement;
  private evaluationResultsDiv: HTMLDivElement;
  private selectionManager: SelectionManager;
  private evaluationManager: EvaluationManager;

  constructor() {
    const canvas = document.getElementById(
      "originalCanvas"
    ) as HTMLCanvasElement;
    this.detector = new ShapeDetector(canvas);

    this.imageInput = document.getElementById("imageInput") as HTMLInputElement;
    this.resultsDiv = document.getElementById("results") as HTMLDivElement;
    this.testImagesDiv = document.getElementById(
      "testImages"
    ) as HTMLDivElement;
    this.evaluateButton = document.getElementById(
      "evaluateButton"
    ) as HTMLButtonElement;
    this.evaluationResultsDiv = document.getElementById(
      "evaluationResults"
    ) as HTMLDivElement;

    this.selectionManager = new SelectionManager();
    this.evaluationManager = new EvaluationManager(
      this.detector,
      this.evaluateButton,
      this.evaluationResultsDiv
    );

    this.setupEventListeners();
    this.loadTestImages().catch(console.error);
  }

  private setupEventListeners(): void {
    this.imageInput.addEventListener("change", async (event) => {
      const file = (event.target as HTMLInputElement).files?.[0];
      if (file) {
        await this.processImage(file);
      }
    });

    this.evaluateButton.addEventListener("click", async () => {
      const selectedImages = this.selectionManager.getSelectedImages();
      await this.evaluationManager.runSelectedEvaluation(selectedImages);
    });
  }

  private async processImage(file: File): Promise<void> {
    try {
      this.resultsDiv.innerHTML = "<p>Processing...</p>";

      const imageData = await this.detector.loadImage(file);
      const results = await this.detector.detectShapes(imageData);

      this.displayResults(results);
    } catch (error) {
      this.resultsDiv.innerHTML = `<p>Error: ${error}</p>`;
    }
  }

  private displayResults(results: DetectionResult): void {
    const { shapes, processingTime } = results;

    let html = `
      <p><strong>Processing Time:</strong> ${processingTime.toFixed(2)}ms</p>
      <p><strong>Shapes Found:</strong> ${shapes.length}</p>
    `;

    if (shapes.length > 0) {
      html += "<h4>Detected Shapes:</h4><ul>";
      shapes.forEach((shape) => {
        html += `
          <li>
            <strong>${
              shape.type.charAt(0).toUpperCase() + shape.type.slice(1)
            }</strong><br>
            Confidence: ${(shape.confidence * 100).toFixed(1)}%<br>
            Center: (${shape.center.x.toFixed(1)}, ${shape.center.y.toFixed(
          1
        )})<br>
            Area: ${shape.area.toFixed(1)}px²
          </li>
        `;
      });
      html += "</ul>";
    } else {
      html +=
        "<p>No shapes detected. Please implement the detection algorithm.</p>";
    }

    this.resultsDiv.innerHTML = html;
  }

  private async loadTestImages(): Promise<void> {
    try {
      const module = await import("./test-images-data.js");
      const testImages = module.testImages;
      const imageNames = module.getAllTestImageNames();

      let html =
        '<h4>Click to upload your own image or use test images for detection. Right-click test images to select/deselect for evaluation:</h4><div class="evaluation-controls"><button id="selectAllBtn">Select All</button><button id="deselectAllBtn">Deselect All</button><span class="selection-info">0 images selected</span></div><div class="test-images-grid">';

      // Add upload functionality as first grid item
      html += `
        <div class="test-image-item upload-item" onclick="triggerFileUpload()">
          <div class="upload-icon">📁</div>
          <div class="upload-text">Upload Image</div>  
          <div class="upload-subtext">Click to select file</div>
        </div>
      `;

      imageNames.forEach((imageName) => {
        const dataUrl = testImages[imageName as keyof typeof testImages];
        const displayName = imageName
          .replace(/[_-]/g, " ")
          .replace(/\.(svg|png)$/i, "");
        html += `
          <div class="test-image-item" data-image="${imageName}" 
               onclick="loadTestImage('${imageName}', '${dataUrl}')" 
               oncontextmenu="toggleImageSelection(event, '${imageName}')">
            <img src="${dataUrl}" alt="${imageName}">
            <div>${displayName}</div>
          </div>
        `;
      });

      html += "</div>";
      this.testImagesDiv.innerHTML = html;

      this.selectionManager.setupSelectionControls();

      (window as any).loadTestImage = async (name: string, dataUrl: string) => {
        try {
          const response = await fetch(dataUrl);
          const blob = await response.blob();
          // pick mime type based on extension
          const mime = /\.svg$/i.test(name) ? "image/svg+xml" : "image/png";
          const file = new File([blob], name, { type: mime });

          const imageData = await this.detector.loadImage(file);
          const results = await this.detector.detectShapes(imageData);
          this.displayResults(results);

          console.log(`Loaded test image: ${name}`);
        } catch (error) {
          console.error("Error loading test image:", error);
        }
      };

      (window as any).toggleImageSelection = (
        event: MouseEvent,
        imageName: string
      ) => {
        event.preventDefault();
        this.selectionManager.toggleImageSelection(imageName);
      };

      // Add upload functionality
      (window as any).triggerFileUpload = () => {
        this.imageInput.click();
      };
    } catch (error) {
      this.testImagesDiv.innerHTML = `
        <p>Test images not available. Run 'node convert-svg-to-png.js' to generate test image data.</p>
        <p>SVG files are available in the test-images/ directory.</p>
      `;
    }
  }
}

document.addEventListener("DOMContentLoaded", () => {
  new ShapeDetectionApp();
});
