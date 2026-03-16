'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import {
  Shield, BarChart3, Upload, LogOut, User,
  TrendingUp, Activity, Settings, Cpu, Zap, HardDrive,
  Timer, Check, X, Layers,
} from 'lucide-react';
import {
  BarChart, Bar, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from 'recharts';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import { useAuthStore } from '@/lib/store';
import { useToast } from '@/hooks/use-toast';
import { api } from '@/lib/api';

/* ------------------------------------------------------------------ */
/* Types & Static Data                                                 */
/* ------------------------------------------------------------------ */

interface ModelInfo {
  name: string;
  id: string;
  version: string;
  classes: string[];
  input_size: number[];
  parameters: number;
  loaded: boolean;
}

/** Benchmark data from project training results */
const BENCHMARK_DATA = [
  { name: 'CNN (ResNet50)',   accuracy: 100.0,  f1: 100.0,  inference: 23.48, sizeMB: 93.7,  params: '23.5M' },
  { name: 'Hybrid CNN+ViT',  accuracy: 97.35,  f1: 97.39,  inference: 25.70, sizeMB: 92.3,  params: '24.2M' },
  { name: 'ViT (scratch)',    accuracy: 53.64,  f1: 47.61,  inference: 1.41,  sizeMB: 2.5,   params: '0.66M' },
  { name: 'ViT + SSL (MAE)', accuracy: 49.67,  f1: 37.01,  inference: 1.36,  sizeMB: 2.5,   params: '0.66M' },
];

const RADAR_DATA = BENCHMARK_DATA.map(m => ({
  name: m.name,
  accuracy: m.accuracy,
  f1: m.f1,
  speed: Math.min(100, (1 / m.inference) * 100 * 5),   // Normalize: faster = higher
  efficiency: Math.min(100, (1 / m.sizeMB) * 100 * 10), // Smaller = higher
}));

const KD_DATA = [
  { metric: 'Accuracy',       teacher: 100.0, student: 87.5 },
  { metric: 'F1-Score',       teacher: 100.0, student: 85.2 },
  { metric: 'Inference (ms)', teacher: 23.48, student: 1.41 },
  { metric: 'Size (MB)',      teacher: 93.7,  student: 2.5 },
  { metric: 'Parameters',     teacher: 23.5,  student: 0.66 },
];

const KFOLD_DATA = [
  { fold: 'Fold 1', accuracy: 31.8 },
  { fold: 'Fold 2', accuracy: 25.0 },
  { fold: 'Fold 3', accuracy: 34.1 },
  { fold: 'Fold 4', accuracy: 35.2 },
  { fold: 'Fold 5', accuracy: 31.4 },
];

/* ------------------------------------------------------------------ */
/* Component                                                           */
/* ------------------------------------------------------------------ */

export default function ModelComparisonPage() {
  const router = useRouter();
  const { user, token, logout, loadUser, authDisabled } = useAuthStore();
  const { toast } = useToast();
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const init = async () => {
      if (!token && !authDisabled) { router.push('/login'); return; }
      await loadUser();
      try {
        const res = await api.getModels();
        if (res.data) setModels(res.data);
      } catch { /* API may not be running */ }
      setIsLoading(false);
    };
    init();
  }, [token, authDisabled, router, loadUser]);

  const handleLogout = () => {
    logout();
    toast({ title: 'Logged out', description: 'Session ended.' });
    router.push('/');
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-slate-50 flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600" />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-50">
      {/* Header */}
      <header className="bg-white border-b">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
          <Link href="/dashboard" className="flex items-center gap-2">
            
            <span className="text-xl font-bold"></span>
          </Link>
          <nav className="flex items-center gap-4">
            
            
            
            
            
            <div className="flex items-center gap-2 ml-4 pl-4 border-l">
              <User className="h-4 w-4 text-muted-foreground" />
              <span className="text-sm">{user?.username || 'User'}</span>
              <Button variant="ghost" size="sm" onClick={handleLogout}><LogOut className="h-4 w-4" /></Button>
            </div>
          </nav>
        </div>
      </header>

      {/* Main */}
      <main className="max-w-7xl mx-auto px-4 py-8">
        <div className="mb-8">
          <h1 className="text-3xl font-bold">Model Comparison</h1>
          <p className="text-muted-foreground">Compare architectures: CNN, ViT, Hybrid, ViT+SSL, and Knowledge Distillation</p>
        </div>

        <Tabs defaultValue="benchmark" className="space-y-6">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="benchmark">Benchmark</TabsTrigger>
            <TabsTrigger value="loaded">Loaded Models</TabsTrigger>
            <TabsTrigger value="distillation">Distillation</TabsTrigger>
            <TabsTrigger value="crossval">Cross-Validation</TabsTrigger>
          </TabsList>

          {/* Benchmark Tab */}
          <TabsContent value="benchmark">
            <div className="grid gap-6">
              {/* Comparison Table */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2"><Layers className="h-5 w-5" /> Architecture Benchmark</CardTitle>
                  <CardDescription>Side-by-side comparison of all trained model architectures</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="border-b">
                          <th className="text-left py-3 px-4 font-semibold">Model</th>
                          <th className="text-right py-3 px-4 font-semibold">Accuracy</th>
                          <th className="text-right py-3 px-4 font-semibold">F1-Score</th>
                          <th className="text-right py-3 px-4 font-semibold">Inference</th>
                          <th className="text-right py-3 px-4 font-semibold">Size</th>
                          <th className="text-right py-3 px-4 font-semibold">Params</th>
                        </tr>
                      </thead>
                      <tbody>
                        {BENCHMARK_DATA.map((m, i) => (
                          <tr key={m.name} className={`border-b ${i === 0 ? 'bg-green-50' : ''}`}>
                            <td className="py-3 px-4 font-medium">{m.name} {i === 0 && <span className="text-xs bg-green-500 text-white px-2 py-0.5 rounded ml-2">Best</span>}</td>
                            <td className="text-right py-3 px-4">{m.accuracy.toFixed(2)}%</td>
                            <td className="text-right py-3 px-4">{m.f1.toFixed(2)}%</td>
                            <td className="text-right py-3 px-4">{m.inference.toFixed(2)} ms</td>
                            <td className="text-right py-3 px-4">{m.sizeMB} MB</td>
                            <td className="text-right py-3 px-4">{m.params}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </CardContent>
              </Card>

              {/* Charts Row */}
              <div className="grid md:grid-cols-2 gap-6">
                <Card>
                  <CardHeader>
                    <CardTitle>Accuracy &amp; F1 Comparison</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <ResponsiveContainer width="100%" height={300}>
                      <BarChart data={BENCHMARK_DATA} layout="vertical">
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis type="number" domain={[0, 100]} fontSize={12} />
                        <YAxis type="category" dataKey="name" fontSize={11} width={120} />
                        <Tooltip />
                        <Legend />
                        <Bar dataKey="accuracy" fill="#3b82f6" name="Accuracy %" radius={[0, 4, 4, 0]} />
                        <Bar dataKey="f1" fill="#22c55e" name="F1 %" radius={[0, 4, 4, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle>Multi-Metric Radar</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <ResponsiveContainer width="100%" height={300}>
                      <RadarChart data={[
                        { metric: 'Accuracy', ...Object.fromEntries(RADAR_DATA.map(m => [m.name, m.accuracy])) },
                        { metric: 'F1', ...Object.fromEntries(RADAR_DATA.map(m => [m.name, m.f1])) },
                        { metric: 'Speed', ...Object.fromEntries(RADAR_DATA.map(m => [m.name, m.speed])) },
                        { metric: 'Efficiency', ...Object.fromEntries(RADAR_DATA.map(m => [m.name, m.efficiency])) },
                      ]}>
                        <PolarGrid />
                        <PolarAngleAxis dataKey="metric" fontSize={12} />
                        <PolarRadiusAxis angle={30} domain={[0, 100]} />
                        <Radar name="CNN" dataKey="CNN (ResNet50)" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.2} />
                        <Radar name="Hybrid" dataKey="Hybrid CNN+ViT" stroke="#22c55e" fill="#22c55e" fillOpacity={0.2} />
                        <Radar name="ViT" dataKey="ViT (scratch)" stroke="#f97316" fill="#f97316" fillOpacity={0.2} />
                        <Legend />
                        <Tooltip />
                      </RadarChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>
              </div>

              {/* Inference vs Size Scatter */}
              <div className="grid md:grid-cols-2 gap-6">
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2"><Timer className="h-5 w-5" /> Inference Time</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <ResponsiveContainer width="100%" height={250}>
                      <BarChart data={BENCHMARK_DATA}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="name" fontSize={11} />
                        <YAxis fontSize={12} />
                        <Tooltip />
                        <Bar dataKey="inference" fill="#a855f7" name="ms" radius={[4, 4, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2"><HardDrive className="h-5 w-5" /> Model Size</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <ResponsiveContainer width="100%" height={250}>
                      <BarChart data={BENCHMARK_DATA}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="name" fontSize={11} />
                        <YAxis fontSize={12} />
                        <Tooltip />
                        <Bar dataKey="sizeMB" fill="#f97316" name="MB" radius={[4, 4, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>
              </div>
            </div>
          </TabsContent>

          {/* Loaded Models Tab */}
          <TabsContent value="loaded">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2"><Cpu className="h-5 w-5" /> Currently Loaded Models</CardTitle>
                <CardDescription>Models available on the backend API for inference</CardDescription>
              </CardHeader>
              <CardContent>
                {models.length === 0 ? (
                  <div className="text-center py-12 text-muted-foreground">
                    <Cpu className="h-12 w-12 mx-auto mb-4 opacity-50" />
                    <p>No models loaded. Start the backend server first.</p>
                    <p className="text-xs mt-2">Run: <code className="bg-slate-100 px-2 py-1 rounded">uvicorn backend.main:app --port 8001</code></p>
                  </div>
                ) : (
                  <div className="grid md:grid-cols-2 gap-4">
                    {models.map(m => (
                      <div key={m.id} className="border rounded-lg p-4">
                        <div className="flex items-center justify-between mb-3">
                          <h3 className="font-semibold text-lg">{m.name}</h3>
                          {m.loaded ? (
                            <span className="flex items-center gap-1 text-xs bg-green-100 text-green-700 px-2 py-1 rounded-full"><Check className="h-3 w-3" /> Loaded</span>
                          ) : (
                            <span className="flex items-center gap-1 text-xs bg-red-100 text-red-700 px-2 py-1 rounded-full"><X className="h-3 w-3" /> Offline</span>
                          )}
                        </div>
                        <div className="grid grid-cols-2 gap-2 text-sm text-muted-foreground">
                          <div>Version: <span className="text-foreground">{m.version}</span></div>
                          <div>Input: <span className="text-foreground">{m.input_size.join('×')}</span></div>
                          <div>Params: <span className="text-foreground">{m.parameters.toLocaleString()}</span></div>
                          <div>Classes: <span className="text-foreground">{m.classes.length}</span></div>
                        </div>
                        <div className="mt-2 flex flex-wrap gap-1">
                          {m.classes.map(c => (
                            <span key={c} className="text-xs bg-slate-100 px-2 py-0.5 rounded">{c}</span>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          {/* Distillation Tab */}
          <TabsContent value="distillation">
            <div className="grid gap-6">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2"><Zap className="h-5 w-5" /> Knowledge Distillation</CardTitle>
                  <CardDescription>Teacher (CNN ResNet50) → Student (ViT) compression comparison</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="border-b">
                          <th className="text-left py-3 px-4 font-semibold">Metric</th>
                          <th className="text-right py-3 px-4 font-semibold">Teacher (CNN)</th>
                          <th className="text-right py-3 px-4 font-semibold">Student (ViT)</th>
                          <th className="text-right py-3 px-4 font-semibold">Ratio</th>
                        </tr>
                      </thead>
                      <tbody>
                        {KD_DATA.map(row => (
                          <tr key={row.metric} className="border-b">
                            <td className="py-3 px-4 font-medium">{row.metric}</td>
                            <td className="text-right py-3 px-4">{row.teacher}</td>
                            <td className="text-right py-3 px-4">{row.student}</td>
                            <td className="text-right py-3 px-4">{(row.student / row.teacher).toFixed(2)}×</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Compression Benefits</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid md:grid-cols-3 gap-4">
                    <div className="text-center p-4 bg-blue-50 rounded-lg">
                      <p className="text-3xl font-bold text-blue-600">37.5×</p>
                      <p className="text-sm text-muted-foreground">Size Reduction</p>
                      <p className="text-xs text-muted-foreground mt-1">93.7 MB → 2.5 MB</p>
                    </div>
                    <div className="text-center p-4 bg-green-50 rounded-lg">
                      <p className="text-3xl font-bold text-green-600">16.7×</p>
                      <p className="text-sm text-muted-foreground">Speed Improvement</p>
                      <p className="text-xs text-muted-foreground mt-1">23.48 ms → 1.41 ms</p>
                    </div>
                    <div className="text-center p-4 bg-orange-50 rounded-lg">
                      <p className="text-3xl font-bold text-orange-600">87.5%</p>
                      <p className="text-sm text-muted-foreground">Accuracy Retained</p>
                      <p className="text-xs text-muted-foreground mt-1">100% → 87.5% (with KD)</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* Cross-Validation Tab */}
          <TabsContent value="crossval">
            <div className="grid gap-6">
              <Card>
                <CardHeader>
                  <CardTitle>5-Fold Cross-Validation Results</CardTitle>
                  <CardDescription>Stratified K-Fold on training dataset</CardDescription>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={KFOLD_DATA}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="fold" fontSize={12} />
                      <YAxis domain={[0, 100]} fontSize={12} />
                      <Tooltip />
                      <Bar dataKey="accuracy" fill="#6366f1" name="Val Accuracy %" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Summary Statistics</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid md:grid-cols-4 gap-4">
                    <div className="text-center p-4 bg-indigo-50 rounded-lg">
                      <p className="text-2xl font-bold text-indigo-600">31.5%</p>
                      <p className="text-sm text-muted-foreground">Mean Accuracy</p>
                    </div>
                    <div className="text-center p-4 bg-indigo-50 rounded-lg">
                      <p className="text-2xl font-bold text-indigo-600">±3.5%</p>
                      <p className="text-sm text-muted-foreground">Std Deviation</p>
                    </div>
                    <div className="text-center p-4 bg-indigo-50 rounded-lg">
                      <p className="text-2xl font-bold text-indigo-600">5</p>
                      <p className="text-sm text-muted-foreground">Folds</p>
                    </div>
                    <div className="text-center p-4 bg-indigo-50 rounded-lg">
                      <p className="text-2xl font-bold text-indigo-600">10</p>
                      <p className="text-sm text-muted-foreground">Epochs / Fold</p>
                    </div>
                  </div>
                  <p className="text-sm text-muted-foreground mt-4">
                    Note: ViT trained from scratch on limited data achieves lower cross-validation accuracy. 
                    CNN with ImageNet pretraining achieves 100% on this dataset. Consider using the Hybrid model 
                    or collecting more training data for better ViT performance.
                  </p>
                </CardContent>
              </Card>
            </div>
          </TabsContent>
        </Tabs>
      </main>
    </div>
  );
}
