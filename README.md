# Final Flam app DEV
shape detection for flam assignment
Shape detection — theory & explanation
Overview (what this does and why)

This detector is a small, classical image-processing pipeline designed to locate and classify simple 2D shapes (circle, triangle, rectangle/square, pentagon, star) in clean images.
It avoids heavy ML models and instead uses basic operations you can inspect and tweak: grayscale → blur → automatic thresholding → connected components → contour tracing → polygon simplification → rule-based classification. That makes it lightweight, fast in a browser, and easy to tune for specific datasets.

High-level pipeline (step by step)

Grayscale conversion
Convert the color image to a single luminance channel. This reduces data to process and is enough when shapes are distinguished by intensity from their background.

Noise reduction (box blur)
A small box blur smooths minor texture and sensor noise while keeping corners and main edges intact. This reduces tiny specks that would otherwise make many tiny components.

Automatic thresholding (Otsu-like)
Compute a histogram and pick an optimal global threshold. Pixels darker than the threshold are treated as foreground (shape) and the rest as background. This step assumes shapes are darker than the background — invert the comparison if the opposite is true.

Small noise removal
Remove isolated single pixels or tiny specks using a simple neighborhood check. This removes very small artifacts before component extraction.

Connected components
Group foreground pixels into blobs using a 4-neighbour flood fill (BFS/stack). Each blob is a candidate shape.

Pre-filter components
Discard blobs that are implausibly small or extremely elongated (likely text, lines, or noise). Keep small but valid shapes (like star tips) by tuning the minimum area.

Contour extraction (Moore neighbor tracing)
For each blob, trace the boundary in order to get a clean contour (sequence of boundary pixels). This is required for polygon approximation and curvature analysis.

Polygon approximation (Ramer–Douglas–Peucker)
Reduce the contour to a simpler polygon with fewer vertices, while preserving corner positions. This step gives a small set of vertices to be used for classification (number-of-vertices heuristic).

Geometry measures
Compute perimeter, pixel area, bounding box, fill ratio, circularity (4πA/P²), convex hull and hull area (for solidity), and concavity count. These are robust, interpretable features.

Rule-based classification

If circularity is high and bounding box near-square ⇒ circle.

If polygon has 3 vertices ⇒ triangle.

If 4 vertices with many roughly 90° angles or high solidity ⇒ rectangle/square (distinguish square by aspect ratio).

If 5 vertices ⇒ pentagon.

If many vertices but strong concavity (alternating angle sign) or low solidity ⇒ star.
Confidence scores are computed from these heuristics so you can threshold outputs.

Return results
For each detected shape return its type, confidence, bounding box, center and area along with processing time.

Why these choices?

No ML required: this is deterministic and fast, and easy to reason about. Good for well-constrained datasets (white background, dark shapes).

RDP polygonization preserves the essential corners so pentagons or stars remain identifiable, while still removing noise along long smooth curves.

Concavity / solidity checks are very effective at discriminating stars and other concave shapes from convex polygons.

Unable to detect no shapes because there is shapes in the picture two lines

pentagon and rectangloe was not able to detect simultaneously.

web console o/p
Testing selected: circle_simple.png
evaluation.ts:60 Testing selected: complex_scene.png
evaluation.ts:60 Testing selected: edge_cases.png
evaluation.ts:60 Testing selected: mixed_shapes_simple.png
evaluation.ts:60 Testing selected: no_shapes.png
evaluation.ts:60 Testing selected: noisy_background.png
evaluation.ts:60 Testing selected: pentagon_regular.png
evaluation.ts:60 Testing selected: rectangle_square.png
evaluation.ts:60 Testing selected: star_five_point.png
evaluation.ts:60 Testing selected: triangle_basic.png
evaluation.ts:148 Selected evaluation complete!
evaluation-manager.ts:41 Selected Evaluation Results: 
{totalScore: 725, maxScore: 1000, percentage: 72.5, grade: 'C', testResults: Array(10), …}
grade
: 
"C"
maxScore
: 
1000
percentage
: 
72.5
summary
: 
{averagePrecision: 0.6166666666666666, averageRecall: 0.7166666666666666, averageF1: 0.6166666666666666, averageIoU: 0.6403989852722465, totalProcessingTime: 143.40000009536743}
testResults
: 
(10) [{…}, {…}, {…}, {…}, {…}, {…}, {…}, {…}, {…}, {…}]
totalScore
: 
725
[[Prototype]]
: 
Object
