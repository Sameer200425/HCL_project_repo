/**
 * Image Quality Assessment Utilities (CPU / Canvas-based)
 * ========================================================
 * Lightweight image quality checks that run entirely in the browser.
 * Uses the Canvas API to read pixel data — no native WASM binary needed.
 *
 * For production, swap the internals with opencv.js for real Laplacian,
 * Canny, and perspective-transform support. The component API stays the same.
 */

export interface QualityReport {
  /** Overall pass / fail */
  passed: boolean;
  /** 0-100 sharpness score (higher = sharper) */
  sharpness: number;
  /** true when large overexposed patches detected */
  hasGlare: boolean;
  /** Fraction of edge pixels in the image (0-1) */
  edgeDensity: number;
  /** Human-readable issues list */
  issues: string[];
}

/* ------------------------------------------------------------------ */
/*  Helpers                                                           */
/* ------------------------------------------------------------------ */

/** Load an image source (File | Blob | data-URL) into an ImageData. */
export async function loadImageData(
  source: File | Blob | string,
  targetWidth = 640
): Promise<{ imageData: ImageData; canvas: HTMLCanvasElement }> {
  const img = new Image();
  const url =
    typeof source === 'string'
      ? source
      : URL.createObjectURL(source as Blob);

  await new Promise<void>((resolve, reject) => {
    img.onload = () => resolve();
    img.onerror = reject;
    img.src = url;
  });

  if (typeof source !== 'string') URL.revokeObjectURL(url);

  const scale = targetWidth / img.width;
  const w = targetWidth;
  const h = Math.round(img.height * scale);

  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d')!;
  ctx.drawImage(img, 0, 0, w, h);

  return { imageData: ctx.getImageData(0, 0, w, h), canvas };
}

/** Convert RGBA ImageData → single-channel grayscale Uint8Array. */
function toGrayscale(data: ImageData): Uint8Array {
  const gray = new Uint8Array(data.width * data.height);
  for (let i = 0; i < gray.length; i++) {
    const r = data.data[i * 4];
    const g = data.data[i * 4 + 1];
    const b = data.data[i * 4 + 2];
    gray[i] = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
  }
  return gray;
}

/* ------------------------------------------------------------------ */
/*  Blur detection  (variance of Laplacian approximation)             */
/* ------------------------------------------------------------------ */

function laplacianVariance(
  gray: Uint8Array,
  w: number,
  h: number
): number {
  // 3×3 Laplacian kernel: [0,1,0 / 1,-4,1 / 0,1,0]
  let sum = 0;
  let sumSq = 0;
  let count = 0;

  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const idx = y * w + x;
      const lap =
        gray[idx - w] +
        gray[idx - 1] +
        -4 * gray[idx] +
        gray[idx + 1] +
        gray[idx + w];
      sum += lap;
      sumSq += lap * lap;
      count++;
    }
  }

  const mean = sum / count;
  return sumSq / count - mean * mean; // variance
}

/* ------------------------------------------------------------------ */
/*  Glare detection  (overexposed patch ratio)                        */
/* ------------------------------------------------------------------ */

function detectGlare(data: ImageData, threshold = 250): boolean {
  let bright = 0;
  const total = data.width * data.height;

  for (let i = 0; i < total; i++) {
    const r = data.data[i * 4];
    const g = data.data[i * 4 + 1];
    const b = data.data[i * 4 + 2];
    if (r > threshold && g > threshold && b > threshold) bright++;
  }

  // If >15% of the image is blown-out white → glare
  return bright / total > 0.15;
}

/* ------------------------------------------------------------------ */
/*  Edge density  (simple Sobel approximation)                        */
/* ------------------------------------------------------------------ */

function edgeDensity(
  gray: Uint8Array,
  w: number,
  h: number,
  edgeThreshold = 60
): number {
  let edges = 0;
  let count = 0;

  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const idx = y * w + x;
      // Simplified Sobel Gx / Gy
      const gx =
        -gray[idx - w - 1] +
        gray[idx - w + 1] +
        -2 * gray[idx - 1] +
        2 * gray[idx + 1] +
        -gray[idx + w - 1] +
        gray[idx + w + 1];
      const gy =
        -gray[idx - w - 1] +
        -2 * gray[idx - w] +
        -gray[idx - w + 1] +
        gray[idx + w - 1] +
        2 * gray[idx + w] +
        gray[idx + w + 1];

      const mag = Math.sqrt(gx * gx + gy * gy);
      if (mag > edgeThreshold) edges++;
      count++;
    }
  }

  return edges / count;
}

/* ------------------------------------------------------------------ */
/*  Public API                                                        */
/* ------------------------------------------------------------------ */

const SHARPNESS_THRESHOLD = 120; // variance below this → blurry
const EDGE_DENSITY_MIN = 0.04;  // documents should have text edges

/**
 * Run all quality checks on an image.
 * Returns a QualityReport with pass/fail and individual scores.
 */
export async function assessImageQuality(
  source: File | Blob | string
): Promise<QualityReport> {
  const { imageData } = await loadImageData(source);
  const gray = toGrayscale(imageData);
  const w = imageData.width;
  const h = imageData.height;

  const sharpnessRaw = laplacianVariance(gray, w, h);
  // Normalise raw variance (typically 0–5000+) to a 0–100 score
  const sharpness = Math.min(100, Math.round((sharpnessRaw / 50) * 100) / 100 * 10);
  const hasGlare = detectGlare(imageData);
  const edgeDens = edgeDensity(gray, w, h);

  const issues: string[] = [];
  if (sharpnessRaw < SHARPNESS_THRESHOLD) issues.push('Image appears blurry — hold the camera steady.');
  if (hasGlare) issues.push('Glare detected — move away from direct light.');
  if (edgeDens < EDGE_DENSITY_MIN) issues.push('Low edge detail — ensure the full document is visible.');

  return {
    passed: issues.length === 0,
    sharpness: Math.round(sharpness),
    hasGlare,
    edgeDensity: Math.round(edgeDens * 1000) / 1000,
    issues,
  };
}

/**
 * Auto-crop the document from the image.
 * Returns a new canvas with the document region extracted.
 * Falls-back to full image if no clear boundary is found.
 */
export async function autoCropDocument(
  source: File | Blob | string,
  targetSize = 224
): Promise<Blob> {
  const { canvas } = await loadImageData(source, 640);
  // For a full perspective-warp we'd use opencv.js here.
  // This simplified version centre-crops to a square & resizes to targetSize.
  const size = Math.min(canvas.width, canvas.height);
  const sx = (canvas.width - size) / 2;
  const sy = (canvas.height - size) / 2;

  const out = document.createElement('canvas');
  out.width = targetSize;
  out.height = targetSize;
  const ctx = out.getContext('2d')!;
  ctx.drawImage(canvas, sx, sy, size, size, 0, 0, targetSize, targetSize);

  return new Promise<Blob>((resolve) => {
    out.toBlob((blob) => resolve(blob!), 'image/png');
  });
}
