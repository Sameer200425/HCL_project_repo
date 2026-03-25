'use client';

import { useState, useCallback, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import {
  Upload,
  FileImage,
  X,
  Loader2,
  AlertTriangle,
  CheckCircle,
  LogOut,
  User,
  Camera,
  MessageSquare,
  Sparkles,
  Radar,
  ShieldCheck,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import Image from 'next/image';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import { Progress } from '@/components/ui/progress';
import { useAuthStore, usePredictionStore } from '@/lib/store';
import { useToast } from '@/hooks/use-toast';
import SmartScanner from '@/components/SmartScanner';
import { assessImageQuality, type QualityReport } from '@/lib/image-quality';

interface PredictionResult {
  predicted_class: string;
  confidence: number;
  risk_level: string;
  probabilities: Record<string, number>;
  explanation?: string;
}

const riskStyles: Record<string, string> = {
  critical: 'from-rose-500/20 to-red-700/20 border-rose-300/40 text-rose-100',
  high: 'from-orange-400/20 to-amber-700/20 border-orange-300/40 text-orange-100',
  medium: 'from-yellow-400/20 to-amber-500/20 border-yellow-300/40 text-yellow-100',
  low: 'from-emerald-400/20 to-teal-600/20 border-emerald-300/40 text-emerald-100',
};

export default function PredictPage() {
  const router = useRouter();
  const { user, token, logout, authDisabled } = useAuthStore();
  const { predict, isLoading } = usePredictionStore();
  const { toast } = useToast();

  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] = useState('vit');
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [inputMode, setInputMode] = useState<'upload' | 'camera'>('upload');
  const [qualityReport, setQualityReport] = useState<QualityReport | null>(null);
  const [recentResults, setRecentResults] = useState<
    { predicted_class: string; risk_level: string; timestamp: string }[]
  >([]);

  useEffect(() => {
    if (!token && !authDisabled) {
      router.push('/login');
    }
  }, [token, authDisabled, router]);

  const handleFile = useCallback((file: File) => {
    if (!file.type.startsWith('image/')) {
      toast({
        title: 'Invalid file type',
        description: 'Please upload an image file.',
        variant: 'destructive',
      });
      return;
    }

    setSelectedFile(file);
    setResult(null);
    setQualityReport(null);

    const reader = new FileReader();
    reader.onload = (e) => {
      setPreview(e.target?.result as string);
    };
    reader.readAsDataURL(file);

    assessImageQuality(file).then((report) => {
      setQualityReport(report);
      if (!report.passed) {
        toast({
          title: 'Image quality warning',
          description: report.issues[0] || 'Image quality may affect accuracy.',
          variant: 'destructive',
        });
      }
    });
  }, [toast]);

  const handleScanCapture = useCallback((file: File) => {
    setSelectedFile(file);
    setResult(null);
    setQualityReport({ passed: true, sharpness: 100, hasGlare: false, edgeDensity: 0.1, issues: [] });

    const reader = new FileReader();
    reader.onload = (e) => setPreview(e.target?.result as string);
    reader.readAsDataURL(file);

    toast({ title: 'Scan captured', description: 'Quality-checked image ready for analysis.' });
  }, [toast]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);

    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  }, [handleFile]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback(() => {
    setIsDragging(false);
  }, []);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;

    const predResult = await predict(selectedFile, selectedModel);

    if (predResult.success) {
      const data = predResult.data as PredictionResult;
      setResult(data);
      setRecentResults((prev) => {
        const entry = {
          predicted_class: data.predicted_class,
          risk_level: data.risk_level,
          timestamp: new Date().toLocaleTimeString(),
        };
        return [entry, ...prev].slice(0, 3);
      });
      toast({
        title: 'Analysis complete',
        description: `Document classified as: ${(predResult.data as PredictionResult).predicted_class}`,
      });
    } else {
      toast({
        title: 'Analysis failed',
        description: predResult.error || 'Please try again.',
        variant: 'destructive',
      });
    }
  };

  const clearFile = () => {
    setSelectedFile(null);
    setPreview(null);
    setResult(null);
  };

  const handleLogout = () => {
    logout();
    router.push('/');
  };

  const models = [
    { id: 'vit', name: 'Vision Transformer', description: 'Highest test accuracy · Recommended for most cases' },
    { id: 'vit_ssl', name: 'ViT + SSL', description: 'Robust to noisy scan texture and lighting' },
    { id: 'hybrid', name: 'Hybrid CNN-ViT', description: 'Balanced throughput and precision for production' },
    { id: 'cnn', name: 'CNN', description: 'Fastest response mode for bulk triage' },
  ];

  const normalizedRisk = result?.risk_level?.toLowerCase() ?? 'low';
  const riskLabel = normalizedRisk.charAt(0).toUpperCase() + normalizedRisk.slice(1);

  return (
    <div className="min-h-screen bg-[#0B1220] text-slate-100 relative overflow-hidden">
      <div className="pointer-events-none absolute -top-40 -left-40 h-[28rem] w-[28rem] rounded-full bg-cyan-500/15 blur-3xl" />
      <div className="pointer-events-none absolute -bottom-40 -right-40 h-[30rem] w-[30rem] rounded-full bg-emerald-500/10 blur-3xl" />

      <header className="relative z-10 border-b border-white/10 backdrop-blur-sm bg-[#0B1220]/80">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
          <Link href="/dashboard" className="flex items-center gap-3">
            <div className="h-9 w-9 rounded-lg bg-gradient-to-br from-cyan-400 to-emerald-400 flex items-center justify-center shadow-[0_0_18px_rgba(52,211,153,0.35)]">
              <Radar className="h-5 w-5 text-slate-950" />
            </div>
          </Link>

          <div className="flex items-center gap-3 text-sm">
            <div className="hidden md:flex items-center gap-2 px-3 py-1.5 rounded-full border border-emerald-300/30 bg-emerald-300/10 text-emerald-200">
              <ShieldCheck className="h-4 w-4" />
              Live Detection
            </div>
            <div className="flex items-center gap-2 ml-2 pl-3 border-l border-white/15">
              <User className="h-4 w-4 text-slate-300" />
              <span className="text-slate-200">{user?.username || 'User'}</span>
              <Button variant="ghost" size="sm" onClick={handleLogout} className="text-slate-200 hover:text-white hover:bg-white/10">
                <LogOut className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </div>
      </header>

      <main className="relative z-10 max-w-7xl mx-auto px-4 py-8">
        <div className="mb-8 flex flex-col gap-3">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-white/5 border border-white/10 text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-300 w-fit">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
            Live cheque &amp; statement fraud scan
          </div>
          <h1 className="text-3xl md:text-4xl font-extrabold tracking-tight">Cheque &amp; Statement Analysis</h1>
          <p className="text-slate-300 max-w-3xl">
            Upload or scan a banking document to run 4-class fraud analysis with confidence scores, risk band, and an AI explanation summary.
          </p>
          <div className="flex flex-wrap gap-2 text-[11px] text-slate-200">
            <span className="px-3 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/40 font-medium">
              99.2% test accuracy · Hybrid CNN + ViT
            </span>
            <span className="px-3 py-1 rounded-full bg-cyan-500/10 border border-cyan-500/40 font-medium">
              4 classes · Genuine, Fraud, Tampered, Forged
            </span>
            <span className="px-3 py-1 rounded-full bg-purple-500/10 border border-purple-500/40 font-medium flex items-center gap-1">
              <Sparkles className="h-3 w-3" /> Explainable results · Heatmaps &amp; summary
            </span>
          </div>
        </div>

        <div className="grid xl:grid-cols-12 gap-6">
          <section className="xl:col-span-7 space-y-6">
            <Card className="bg-white/[0.03] border-white/10 backdrop-blur-md rounded-2xl shadow-[0_12px_35px_rgba(0,0,0,0.28)]">
              <CardContent className="p-4 md:p-5 space-y-4">
                <div className="grid grid-cols-2 gap-2 rounded-xl border border-white/10 p-1 bg-slate-900/40">
                  <button
                    onClick={() => setInputMode('upload')}
                    className={`flex items-center justify-center gap-2 py-2.5 px-3 rounded-lg text-sm font-semibold transition-all ${
                      inputMode === 'upload'
                        ? 'bg-gradient-to-r from-cyan-500 to-blue-500 text-white shadow-[0_0_18px_rgba(56,189,248,0.4)]'
                        : 'text-slate-300 hover:bg-white/5'
                    }`}
                  >
                    <Upload className="h-4 w-4" /> File Upload
                  </button>
                  <button
                    onClick={() => setInputMode('camera')}
                    className={`flex items-center justify-center gap-2 py-2.5 px-3 rounded-lg text-sm font-semibold transition-all ${
                      inputMode === 'camera'
                        ? 'bg-gradient-to-r from-emerald-500 to-cyan-500 text-white shadow-[0_0_18px_rgba(16,185,129,0.45)]'
                        : 'text-slate-300 hover:bg-white/5'
                    }`}
                  >
                    <Camera className="h-4 w-4" /> Smart Scanner
                  </button>
                </div>

                <p className="text-xs text-slate-400 px-0.5">
                  Step 1: Choose how to provide the document. Step 2: Upload or capture it. Step 3: Select a model profile and run analysis.
                </p>

                {inputMode === 'camera' ? (
                  <SmartScanner onCapture={handleScanCapture} targetSize={224} />
                ) : (
                  <div className="space-y-4">
                    {!selectedFile ? (
                      <label
                        className={`block border-2 border-dashed rounded-2xl p-10 text-center transition-all cursor-pointer ${
                          isDragging
                            ? 'border-cyan-400 bg-cyan-400/10'
                            : 'border-slate-500/40 hover:border-cyan-300/70 hover:bg-white/5'
                        }`}
                        onDrop={handleDrop}
                        onDragOver={handleDragOver}
                        onDragLeave={handleDragLeave}
                      >
                        <input type="file" accept="image/*" className="hidden" onChange={handleFileSelect} />
                        <FileImage className="h-12 w-12 mx-auto mb-4 text-slate-300" />
                        <p className="text-lg font-semibold mb-1">Drop image to begin</p>
                        <p className="text-sm text-slate-300 mb-4">or click to browse your files</p>
                        <span className="inline-flex px-4 py-2 rounded-lg border border-cyan-300/40 bg-cyan-300/10 text-cyan-100 text-sm font-medium">Select Document</span>
                      </label>
                    ) : (
                      <div className="relative rounded-2xl overflow-hidden border border-white/10 bg-black/30">
                        {preview ? (
                          <Image
                            src={preview}
                            alt="Preview"
                            className="w-full h-72 object-contain bg-slate-950/70"
                            width={600}
                            height={288}
                            unoptimized
                          />
                        ) : null}
                        <Button
                          variant="destructive"
                          size="sm"
                          className="absolute top-3 right-3"
                          onClick={clearFile}
                        >
                          <X className="h-4 w-4" />
                        </Button>
                        <p className="px-3 py-2 text-xs text-slate-300 border-t border-white/10">{selectedFile.name}</p>
                      </div>
                    )}
                  </div>
                )}

                {qualityReport && selectedFile && (
                  <div
                    className={`flex items-center gap-2 px-4 py-3 rounded-xl text-sm border ${
                      qualityReport.passed
                        ? 'bg-emerald-400/10 border-emerald-300/30 text-emerald-100'
                        : 'bg-amber-400/10 border-amber-300/30 text-amber-100'
                    }`}
                  >
                    {qualityReport.passed ? (
                      <>
                        <CheckCircle className="h-4 w-4 shrink-0" />
                        <span>
                          Image quality ready for analysis ({qualityReport.sharpness}%).
                          <span className="ml-1 text-xs opacity-80 block sm:inline">
                            {qualityReport.sharpness >= 70 ? 'Sharp' : 'Slightly soft'} ·{' '}
                            {qualityReport.hasGlare ? 'Glare present' : 'Good lighting'} ·{' '}
                            {qualityReport.edgeDensity > 0.08 ? 'Clear edges' : 'Low edge detail'}
                          </span>
                        </span>
                      </>
                    ) : (
                      <>
                        <AlertTriangle className="h-4 w-4 shrink-0" /> {qualityReport.issues[0]}
                      </>
                    )}
                  </div>
                )}
              </CardContent>
            </Card>

            <Card className="bg-white/[0.03] border-white/10 backdrop-blur-md rounded-2xl shadow-[0_12px_35px_rgba(0,0,0,0.28)]">
              <CardHeader>
                <CardTitle className="text-slate-100">Model Arena</CardTitle>
                <CardDescription className="text-slate-300">
                  Choose a model profile for this cheque or statement. Vision Transformer is recommended for best overall performance.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid sm:grid-cols-2 gap-3">
                  {models.map((model) => (
                    <button
                      key={model.id}
                      onClick={() => setSelectedModel(model.id)}
                      className={`p-3.5 rounded-xl border text-left transition-all ${
                        selectedModel === model.id
                          ? 'border-cyan-300/70 bg-cyan-300/10 shadow-[0_0_18px_rgba(34,211,238,0.28)]'
                          : 'border-white/10 bg-slate-900/40 hover:border-cyan-300/40'
                      }`}
                    >
                      <p className="font-semibold text-sm text-slate-100">{model.name}</p>
                      <p className="text-xs text-slate-300 mt-0.5">{model.description}</p>
                    </button>
                  ))}
                </div>

                <Button
                  className="w-full h-12 mt-5 text-base font-semibold bg-gradient-to-r from-cyan-500 to-emerald-500 hover:from-cyan-400 hover:to-emerald-400 text-slate-950"
                  disabled={!selectedFile || isLoading}
                  onClick={handleAnalyze}
                >
                  {isLoading ? (
                    <><Loader2 className="mr-2 h-5 w-5 animate-spin" /> Analyzing...</>
                  ) : (
                    <><Upload className="mr-2 h-5 w-5" /> Analyze Document</>
                  )}
                </Button>
                <p className="mt-2 text-xs text-slate-400 text-center">
                  {!selectedFile ? 'No document selected yet.' : 'Ready to analyze the selected document.'}
                </p>
              </CardContent>
            </Card>
          </section>

          <section className="xl:col-span-5">
            <Card className="h-full bg-white/[0.03] border-white/10 backdrop-blur-md rounded-2xl shadow-[0_12px_35px_rgba(0,0,0,0.28)]">
              <CardHeader>
                <CardTitle className="text-slate-100">Analysis Results</CardTitle>
                <CardDescription className="text-slate-300">
                  {result ? 'Live prediction finished.' : 'Upload and analyze a document to view fraud insights.'}
                </CardDescription>
              </CardHeader>
              <CardContent>
                {isLoading ? (
                  <div className="h-[28rem] rounded-2xl border border-white/10 bg-slate-900/40 p-5 animate-pulse">
                    <div className="h-8 w-40 bg-slate-700/60 rounded-lg mb-4" />
                    <div className="h-4 w-24 bg-slate-700/40 rounded mb-6" />
                    <div className="space-y-3 mb-6">
                      <div className="h-3 w-full bg-slate-700/40 rounded" />
                      <div className="h-3 w-[85%] bg-slate-700/30 rounded" />
                      <div className="h-3 w-[70%] bg-slate-700/20 rounded" />
                    </div>
                    <div className="space-y-2 mb-6">
                      <div className="h-3 w-32 bg-slate-700/40 rounded" />
                      <div className="h-2 w-full bg-slate-700/30 rounded" />
                    </div>
                    <div className="space-y-2">
                      <div className="h-3 w-40 bg-slate-700/40 rounded" />
                      <div className="h-2 w-full bg-slate-700/30 rounded" />
                      <div className="h-2 w-[90%] bg-slate-700/20 rounded" />
                    </div>
                  </div>
                ) : !result ? (
                  <div className="h-[28rem] rounded-2xl border border-dashed border-white/15 bg-slate-900/30 flex items-center justify-center text-slate-300">
                    <div className="text-center px-8">
                      <FileImage className="h-16 w-16 mx-auto mb-4 opacity-50" />
                      <p className="font-medium">Result panel is waiting for input.</p>
                      <p className="text-sm mt-1">Run analysis to see risk score, confidence, and AI explanation.</p>
                    </div>
                  </div>
                ) : (
                  <div className="space-y-6">
                    <div className={`rounded-2xl border p-5 bg-gradient-to-br ${riskStyles[normalizedRisk] || riskStyles.low}`}>
                      <div className="flex items-center justify-between gap-3 mb-4">
                        <div>
                          <p className="text-xs uppercase tracking-wider opacity-80">Predicted Class</p>
                          <h3 className="text-2xl font-extrabold mt-1 capitalize">{result.predicted_class}</h3>
                        </div>
                        <div className="rounded-xl p-3 bg-black/20 border border-white/15">
                          {normalizedRisk === 'low' ? (
                            <CheckCircle className="h-8 w-8" />
                          ) : (
                            <AlertTriangle className="h-8 w-8" />
                          )}
                        </div>
                      </div>
                      <p className="text-sm uppercase tracking-wide opacity-90">{riskLabel} risk</p>
                    </div>

                    <div>
                      <Label className="text-slate-300">Confidence</Label>
                      <div className="flex items-center gap-3 mt-2">
                        <Progress value={result.confidence * 100} className="flex-1" />
                        <span className="font-bold text-lg text-slate-100">{(result.confidence * 100).toFixed(1)}%</span>
                      </div>
                    </div>

                    <div>
                      <Label className="text-slate-300">Class Probabilities</Label>
                      <div className="space-y-2 mt-2">
                        {Object.entries(result.probabilities).map(([cls, prob]) => (
                          <div key={cls} className="flex items-center gap-2">
                            <span className="w-24 text-sm text-slate-200 truncate">{cls}</span>
                            <Progress value={prob * 100} className="flex-1 h-2" />
                            <span className="text-sm w-14 text-right text-slate-200">{(prob * 100).toFixed(1)}%</span>
                          </div>
                        ))}
                      </div>
                    </div>

                    {result.explanation && (
                      <div>
                        <Label className="text-slate-300 flex items-center gap-2">
                          <MessageSquare className="h-4 w-4" />
                          AI Explanation
                        </Label>
                        <div className="mt-2 p-4 rounded-xl border border-white/15 bg-slate-900/50 text-sm text-slate-200 leading-relaxed">
                          {result.explanation}
                        </div>
                      </div>
                    )}

                    {recentResults.length > 0 && (
                      <div className="pt-3 mt-1 border-t border-white/10">
                        <p className="text-xs text-slate-400 mb-2">Last runs (this session)</p>
                        <div className="space-y-1.5">
                          {recentResults.map((r, idx) => {
                            const rl = r.risk_level.charAt(0).toUpperCase() + r.risk_level.slice(1);
                            return (
                              <div
                                key={idx}
                                className="flex items-center justify-between text-xs text-slate-300"
                              >
                                <span className="truncate max-w-[45%] capitalize">{r.predicted_class}</span>
                                <span className="text-slate-400">{rl}</span>
                                <span className="text-slate-500">{r.timestamp}</span>
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </CardContent>
            </Card>
          </section>
        </div>
      </main>
    </div>
  );
}
