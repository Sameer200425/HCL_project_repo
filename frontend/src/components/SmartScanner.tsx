'use client';

import { useRef, useState, useCallback, useEffect } from 'react';
import {
  Camera, CameraOff, RotateCcw, CheckCircle, AlertTriangle,
  Sun, Focus, Scan, Loader2,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import {
  assessImageQuality,
  autoCropDocument,
  type QualityReport,
} from '@/lib/image-quality';

interface SmartScannerProps {
  /** Called with the processed, quality-checked image ready for upload. */
  onCapture: (file: File) => void;
  /** Target output size (default 224 to match ViT input). */
  targetSize?: number;
}

export default function SmartScanner({ onCapture, targetSize = 224 }: SmartScannerProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const frameIdRef = useRef<number>(0);

  const [cameraActive, setCameraActive] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [quality, setQuality] = useState<QualityReport | null>(null);
  const [processing, setProcessing] = useState(false);
  const [captured, setCaptured] = useState<string | null>(null);
  const [facingMode, setFacingMode] = useState<'environment' | 'user'>('environment');

  /* ---------------------------------------------------------------- */
  /*  Camera lifecycle                                                */
  /* ---------------------------------------------------------------- */

  const startCamera = useCallback(async () => {
    setCameraError(null);
    try {
      // Check if mediaDevices is available
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('Camera API not available. Use HTTPS or localhost.');
      }
      
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode, width: { ideal: 1280 }, height: { ideal: 720 } },
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setCameraActive(true);
    } catch (err: any) {
      console.error('Camera error:', err);
      if (err.name === 'NotAllowedError') {
        setCameraError('Camera permission denied. Click the camera icon in browser address bar to allow.');
      } else if (err.name === 'NotFoundError') {
        setCameraError('No camera found. Please connect a camera.');
      } else if (err.name === 'NotReadableError') {
        setCameraError('Camera is in use by another application. Close other apps using camera.');
      } else {
        setCameraError(err.message || 'Camera access denied or unavailable.');
      }
    }
  }, [facingMode]);

  const stopCamera = useCallback(() => {
    cancelAnimationFrame(frameIdRef.current);
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    setCameraActive(false);
  }, []);

  useEffect(() => {
    return () => stopCamera(); // cleanup on unmount
  }, [stopCamera]);

  /* ---------------------------------------------------------------- */
  /*  Real-time quality feedback loop (~5 FPS to stay light)          */
  /* ---------------------------------------------------------------- */

  const runQualityLoop = useCallback(() => {
    if (!videoRef.current || !canvasRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d')!;

    let lastCheck = 0;
    const CHECK_INTERVAL_MS = 200; // ~5 checks / second

    const loop = async (timestamp: number) => {
      if (!streamRef.current) return;

      if (timestamp - lastCheck > CHECK_INTERVAL_MS) {
        lastCheck = timestamp;
        canvas.width = 320;
        canvas.height = Math.round((video.videoHeight / video.videoWidth) * 320);
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        const blob = await new Promise<Blob | null>((r) =>
          canvas.toBlob(r, 'image/jpeg', 0.6)
        );
        if (blob) {
          const report = await assessImageQuality(blob);
          setQuality(report);
        }
      }

      frameIdRef.current = requestAnimationFrame(loop);
    };

    frameIdRef.current = requestAnimationFrame(loop);
  }, []);

  useEffect(() => {
    if (cameraActive) runQualityLoop();
  }, [cameraActive, runQualityLoop]);

  /* ---------------------------------------------------------------- */
  /*  Capture & process                                               */
  /* ---------------------------------------------------------------- */

  const captureFrame = useCallback(async () => {
    if (!videoRef.current) return;
    setProcessing(true);

    const video = videoRef.current;
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = video.videoWidth;
    tempCanvas.height = video.videoHeight;
    const ctx = tempCanvas.getContext('2d')!;
    ctx.drawImage(video, 0, 0);

    const hiResBlob = await new Promise<Blob | null>((r) =>
      tempCanvas.toBlob(r, 'image/png')
    );

    if (!hiResBlob) {
      setProcessing(false);
      return;
    }

    // Auto-crop & resize to model target size
    const croppedBlob = await autoCropDocument(hiResBlob, targetSize);
    const file = new File([croppedBlob], 'smart_scan.png', { type: 'image/png' });

    // Preview
    const reader = new FileReader();
    reader.onload = (e) => setCaptured(e.target?.result as string);
    reader.readAsDataURL(croppedBlob);

    stopCamera();
    setProcessing(false);
    onCapture(file);
  }, [targetSize, stopCamera, onCapture]);

  // Auto-capture when quality passes for >1 second
  const autoCapTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (quality?.passed && cameraActive && !processing) {
      if (!autoCapTimerRef.current) {
        autoCapTimerRef.current = setTimeout(() => {
          captureFrame();
        }, 1200);
      }
    } else {
      if (autoCapTimerRef.current) {
        clearTimeout(autoCapTimerRef.current);
        autoCapTimerRef.current = null;
      }
    }
    return () => {
      if (autoCapTimerRef.current) clearTimeout(autoCapTimerRef.current);
    };
  }, [quality, cameraActive, processing, captureFrame]);

  /* ---------------------------------------------------------------- */
  /*  Flip camera                                                     */
  /* ---------------------------------------------------------------- */
  const flipCamera = () => {
    stopCamera();
    setFacingMode((f) => (f === 'environment' ? 'user' : 'environment'));
  };

  useEffect(() => {
    if (!cameraActive && facingMode) {
      // re-start after flip if was previously active
    }
  }, [facingMode, cameraActive]);

  const reset = () => {
    setCaptured(null);
    setQuality(null);
  };

  /* ---------------------------------------------------------------- */
  /*  Render                                                          */
  /* ---------------------------------------------------------------- */

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Scan className="h-5 w-5 text-blue-500" /> Smart Scanner
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Camera feed */}
        {!captured ? (
          <>
            <div className="relative rounded-lg overflow-hidden bg-black aspect-video">
              <video
                ref={videoRef}
                className="w-full h-full object-cover"
                playsInline
                muted
              />
              {/* Hidden canvas for frame grabs */}
              <canvas ref={canvasRef} className="hidden" />

              {!cameraActive && (
                <div className="absolute inset-0 flex items-center justify-center bg-slate-900/80 text-white">
                  <div className="text-center">
                    <Camera className="h-12 w-12 mx-auto mb-3 opacity-60" />
                    <p className="text-sm opacity-80">
                      {cameraError || 'Camera not active'}
                    </p>
                  </div>
                </div>
              )}

              {/* Quality overlay */}
              {cameraActive && quality && (
                <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/70 to-transparent p-3 space-y-1">
                  {quality.issues.length === 0 ? (
                    <div className="flex items-center gap-2 text-green-400 text-sm font-medium">
                      <CheckCircle className="h-4 w-4" /> Ready — hold still to capture
                    </div>
                  ) : (
                    quality.issues.map((issue, i) => (
                      <div key={i} className="flex items-center gap-2 text-amber-400 text-sm">
                        {issue.includes('blurry') && <Focus className="h-4 w-4 shrink-0" />}
                        {issue.includes('Glare') && <Sun className="h-4 w-4 shrink-0" />}
                        {issue.includes('edge') && <AlertTriangle className="h-4 w-4 shrink-0" />}
                        {issue}
                      </div>
                    ))
                  )}
                </div>
              )}

              {/* Scanning guide overlay */}
              {cameraActive && (
                <div
                  className={`absolute inset-4 border-2 rounded-lg pointer-events-none transition-colors ${
                    quality?.passed ? 'border-green-400' : 'border-white/40'
                  }`}
                />
              )}
            </div>

            {/* Quality meter */}
            {cameraActive && quality && (
              <div className="flex items-center gap-3 text-sm">
                <span className="text-muted-foreground w-20">Sharpness</span>
                <Progress
                  value={quality.sharpness}
                  className={`flex-1 h-2 ${quality.sharpness > 50 ? '[&>div]:bg-green-500' : '[&>div]:bg-amber-500'}`}
                />
                <span className="w-10 text-right font-mono">{quality.sharpness}%</span>
              </div>
            )}

            {/* Controls */}
            <div className="flex gap-2">
              {!cameraActive ? (
                <Button className="flex-1" onClick={startCamera}>
                  <Camera className="h-4 w-4 mr-2" /> Start Camera
                </Button>
              ) : (
                <>
                  <Button variant="outline" onClick={flipCamera}>
                    <RotateCcw className="h-4 w-4" />
                  </Button>
                  <Button
                    className="flex-1"
                    onClick={captureFrame}
                    disabled={processing}
                  >
                    {processing ? (
                      <><Loader2 className="h-4 w-4 mr-2 animate-spin" /> Processing…</>
                    ) : (
                      <><Camera className="h-4 w-4 mr-2" /> Capture Now</>
                    )}
                  </Button>
                  <Button variant="destructive" onClick={stopCamera}>
                    <CameraOff className="h-4 w-4" />
                  </Button>
                </>
              )}
            </div>

            <p className="text-xs text-muted-foreground text-center">
              Point camera at a bank document. The scanner checks blur, glare &amp;
              edges in real-time and auto-captures when quality passes.
            </p>
          </>
        ) : (
          /* Captured preview */
          <div className="space-y-3">
            <img
              src={captured}
              alt="Captured scan"
              className="w-full rounded-lg bg-slate-100"
            />
            <div className="flex items-center gap-2 text-green-600 text-sm font-medium">
              <CheckCircle className="h-4 w-4" /> Image captured &amp; processed ({targetSize}×{targetSize})
            </div>
            <Button variant="outline" className="w-full" onClick={reset}>
              <RotateCcw className="h-4 w-4 mr-2" /> Retake
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
