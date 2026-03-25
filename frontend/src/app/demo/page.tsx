'use client';

import { useState } from 'react';
import Link from 'next/link';
import { Play, Sparkles, Server, Zap, ChevronRight, Activity, ArrowRightLeft } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';

export default function DemoPage() {
  const [isRunning, setIsRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [demoState, setDemoState] = useState<'idle' | 'running' | 'completed'>('idle');

  const runDemo = () => {
    setIsRunning(true);
    setDemoState('running');
    setProgress(0);

    const interval = setInterval(() => {
      setProgress((prev) => {
        if (prev >= 100) {
          clearInterval(interval);
          setIsRunning(false);
          setDemoState('completed');
          return 100;
        }
        return prev + Math.floor(Math.random() * 15) + 5;
      });
    }, 400);
  };

  const resetDemo = () => {
    setDemoState('idle');
    setProgress(0);
  };

  return (
    <div className="min-h-screen bg-[#0B1220] flex flex-col relative overflow-hidden text-slate-200">
      <div className="pointer-events-none absolute top-40 -left-40 h-[30rem] w-[30rem] rounded-full bg-cyan-500/10 blur-[100px]" />
      <div className="pointer-events-none absolute top-40 -right-40 h-[30rem] w-[30rem] rounded-full bg-emerald-500/10 blur-[100px]" />

      <header className="border-b border-white/10 backdrop-blur-sm bg-[#0B1220]/80 z-10 sticky top-0">
        <div className="max-w-5xl mx-auto px-4 py-4 flex items-center justify-between">
          <Link href="/" className="flex items-center gap-2">
            <Sparkles className="h-6 w-6 text-emerald-400" />
            <span className="font-bold text-xl tracking-wider text-white">Sentinel Demo</span>
          </Link>
          <Link href="/dashboard">
            <Button variant="ghost" className="text-emerald-400 hover:text-emerald-300 hover:bg-emerald-400/10">
              Back to App <ChevronRight className="h-4 w-4 ml-1" />
            </Button>
          </Link>
        </div>
      </header>

      <main className="flex-1 max-w-5xl w-full mx-auto px-4 py-12 flex flex-col items-center justify-center z-10">
        <div className="text-center max-w-2xl mb-10">
          <h1 className="text-4xl sm:text-5xl font-extrabold text-white mb-4 bg-clip-text text-transparent bg-gradient-to-r from-emerald-400 to-cyan-400 pb-2">
            Live Interactive Demo
          </h1>
          <p className="text-lg text-slate-400">
            Simulate a high-throughput transaction environment. Watch as Sentinel Risk analyzes synthetic cheque payloads in real-time, identifying fraud instantly via ViT and CNN ensemble models.
          </p>
        </div>

        <Card className="w-full max-w-3xl bg-slate-900/50 backdrop-blur-xl border-slate-800 shadow-2xl">
          <CardHeader className="border-b border-slate-800 pb-6">
            <div className="flex items-center justify-between">
              <div>
                <CardTitle className="text-2xl text-white flex items-center gap-2">
                  <Server className="h-5 w-5 text-cyan-400" /> API Stream Simulator
                </CardTitle>
                <CardDescription className="text-slate-400 mt-1">Batch processing of 500 records</CardDescription>
              </div>
              {demoState === 'running' ? (
                <div className="px-3 py-1 rounded-full bg-emerald-500/20 border border-emerald-500/30 text-emerald-400 text-sm font-medium flex items-center gap-2 animate-pulse">
                  <Activity className="h-4 w-4" /> Processing
                </div>
              ) : demoState === 'completed' ? (
                <div className="px-3 py-1 rounded-full bg-cyan-500/20 border border-cyan-500/30 text-cyan-400 text-sm font-medium flex items-center gap-2">
                  <Zap className="h-4 w-4" /> Complete
                </div>
              ) : null}
            </div>
          </CardHeader>
          <CardContent className="p-8">
            <div className="space-y-8">
              {demoState === 'idle' && (
                <div className="flex flex-col items-center justify-center h-48 border-2 border-dashed border-slate-800 rounded-xl bg-slate-950/30">
                  <ArrowRightLeft className="h-10 w-10 text-slate-600 mb-3" />
                  <p className="text-slate-500 font-medium">Ready to start streaming</p>
                </div>
              )}

              {(demoState === 'running' || demoState === 'completed') && (
                <div className="space-y-6 bg-slate-950/50 p-6 rounded-xl border border-slate-800">
                  <div className="flex justify-between text-sm font-medium">
                    <span className="text-slate-300">Throughput Analysis</span>
                    <span className="text-emerald-400">{progress}%</span>
                  </div>
                  <Progress value={progress} className="h-2 bg-slate-800" />
                  
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-6">
                    <div className="p-4 bg-slate-900 rounded-lg border border-slate-800/60 transition-all duration-300">
                      <div className="text-xs text-slate-500 font-semibold mb-1">PROCESSED</div>
                      <div className="text-2xl font-bold text-white">{Math.floor((progress / 100) * 500)}</div>
                    </div>
                    <div className="p-4 bg-slate-900 rounded-lg border border-slate-800/60">
                      <div className="text-xs text-slate-500 font-semibold mb-1">LATENCY</div>
                      <div className="text-2xl font-bold text-white">{progress > 0 ? '14' : '0'} <span className="text-sm font-normal text-slate-500">ms</span></div>
                    </div>
                    <div className="p-4 bg-slate-900 rounded-lg border border-rose-900/30">
                      <div className="text-xs text-rose-500 font-semibold mb-1">FLAGGED</div>
                      <div className="text-2xl font-bold text-rose-400">{Math.floor((progress / 100) * 12)}</div>
                    </div>
                    <div className="p-4 bg-slate-900 rounded-lg border border-emerald-900/30">
                      <div className="text-xs text-emerald-500 font-semibold mb-1">ACCURACY</div>
                      <div className="text-2xl font-bold text-emerald-400">{progress > 0 ? '99.4' : '0.0'} <span className="text-sm font-normal text-emerald-500">%</span></div>
                    </div>
                  </div>
                </div>
              )}

              <div className="flex justify-center gap-4 pt-4">
                {demoState === 'idle' ? (
                  <Button size="lg" onClick={runDemo} className="bg-emerald-500 hover:bg-emerald-600 text-white min-w-[200px] h-12 text-lg">
                    <Play className="h-5 w-5 mr-2" /> Start Demo
                  </Button>
                ) : demoState === 'completed' ? (
                  <Button size="lg" onClick={resetDemo} variant="outline" className="border-slate-700 text-slate-300 hover:bg-slate-800 hover:text-white min-w-[200px] h-12 text-lg">
                    Reset Simulator
                  </Button>
                ) : (
                  <Button size="lg" disabled className="bg-emerald-500/50 text-white min-w-[200px] h-12 text-lg cursor-not-allowed">
                    Processing...
                  </Button>
                )}
              </div>
            </div>
          </CardContent>
        </Card>
      </main>
    </div>
  );
}