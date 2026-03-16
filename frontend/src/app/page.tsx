'use client';

import { useRouter } from 'next/navigation';
import { ShieldAlert, ScanLine, Activity, Fingerprint, ChevronRight, Lock } from 'lucide-react';

export default function HomePage() {
  const router = useRouter();

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200 overflow-hidden relative selection:bg-cyan-500/30">
      
      {/* Background Ambient Glows */}
      <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-cyan-600/20 rounded-full blur-[120px] mix-blend-screen pointer-events-none animate-pulse duration-10000"></div>
      <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-emerald-600/20 rounded-full blur-[120px] mix-blend-screen pointer-events-none animate-pulse duration-7000"></div>

      {/* Navbar */}
      <nav className="relative z-10 border-b border-white/5 bg-slate-950/50 backdrop-blur-md">
        <div className="max-w-7xl mx-auto px-6 h-20 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-500 to-emerald-500 flex items-center justify-center shadow-[0_0_20px_rgba(6,182,212,0.4)]">
              <ShieldAlert className="text-white w-6 h-6" />
            </div>
          </div>
          <button 
            onClick={() => router.push('/login')}
            className="text-sm font-medium text-slate-300 hover:text-white transition-colors flex items-center gap-2"
          >
            <Lock className="w-4 h-4" /> Secure Portal
          </button>
        </div>
      </nav>

      {/* Hero Section */}
      <main className="relative z-10 max-w-7xl mx-auto px-6 pt-20 pb-32 lg:pt-32">
        <div className="grid lg:grid-cols-2 gap-16 items-center">
          
          {/* Left Content */}
          <div className="flex flex-col items-start text-left space-y-8">
            <h1 className="text-5xl lg:text-7xl font-extrabold text-white leading-[1.1] tracking-tight">
              Intercept Fraud.<br />
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 via-teal-400 to-emerald-400">
                In Real-Time.
              </span>
            </h1>
            
            <p className="text-lg lg:text-xl text-slate-400 max-w-xl leading-relaxed">
              Deploy Vision Transformers to instantly authenticate signatures, detect structural tampering, and evaluate financial risk on cheques and statements with 99.2% accuracy.
            </p>

            <div className="flex flex-col sm:flex-row gap-4 w-full sm:w-auto pt-4">
              <button
                onClick={() => router.push('/predict')}
                className="group relative px-8 py-4 bg-white text-slate-950 font-bold text-lg rounded-2xl overflow-hidden transition-all hover:scale-105 shadow-[0_0_40px_rgba(255,255,255,0.1)]"
              >
                <div className="absolute inset-0 bg-gradient-to-r from-cyan-400 to-emerald-400 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
                <span className="relative flex items-center justify-center gap-2 group-hover:text-white transition-colors duration-300">
                  <ScanLine className="w-5 h-5" /> Launch Live Scanner
                </span>
              </button>
              
              <button
                onClick={() => router.push('/analytics')}
                className="px-8 py-4 rounded-2xl font-bold text-lg text-white border border-white/10 bg-white/5 hover:bg-white/10 transition-colors flex items-center justify-center gap-2 backdrop-blur-sm"
              >
                View Analytics <ChevronRight className="w-5 h-5" />
              </button>
            </div>
          </div>

          {/* Right Visual (Interactive Scanner Mockup) */}
          <div className="relative group perspective">
            <div className="absolute -inset-1 bg-gradient-to-r from-cyan-500 to-emerald-500 rounded-[2rem] blur opacity-20 group-hover:opacity-40 transition duration-1000 group-hover:duration-200"></div>
            
            <div className="relative bg-slate-900 border border-white/10 rounded-[2rem] p-8 shadow-2xl overflow-hidden transform transition-transform duration-500 hover:-translate-y-2">
              {/* Fake Cheque Graphic */}
              <div className="w-full h-48 bg-slate-800 rounded-xl border border-slate-700 relative overflow-hidden flex flex-col justify-between p-6">
                <div className="w-3/4 h-4 bg-slate-700 rounded mb-4"></div>
                <div className="flex justify-between items-center">
                  <div className="w-1/3 h-2 bg-slate-700 rounded"></div>
                  <div className="w-24 h-8 bg-emerald-500/20 border border-emerald-500/50 text-emerald-400 rounded flex items-center justify-center text-xs font-bold font-mono shadow-[0_0_10px_rgba(16,185,129,0.2)]">
                    $15,000.00
                  </div>
                </div>
                <div className="self-end mt-4">
                  {/* Fake Signature */}
                  <svg viewBox="0 0 100 40" className="w-24 h-10 opacity-50">
                    <path d="M10,20 Q30,5 40,25 T70,10 T90,30" fill="none" stroke="#38bdf8" strokeWidth="3" strokeLinecap="round" />
                  </svg>
                </div>
                
                {/* CSS Animated Laser Line */}
                <div className="absolute top-0 left-0 w-full h-[2px] bg-cyan-400 shadow-[0_0_15px_#22d3ee] animate-[scan_3s_ease-in-out_infinite]"></div>
              </div>

              {/* Status Modules */}
              <div className="grid grid-cols-2 gap-4 mt-6">
                <div className="bg-slate-950 rounded-xl p-4 border border-white/5">
                  <p className="text-slate-500 text-xs uppercase font-bold mb-1 tracking-wider">ViT Confidence</p>
                  <p className="text-2xl font-black text-white">98.5<span className="text-emerald-400 text-lg">%</span></p>
                </div>
                <div className="bg-slate-950 rounded-xl p-4 border border-white/5">
                  <p className="text-slate-500 text-xs uppercase font-bold mb-1 tracking-wider">ONNX Latency</p>
                  <p className="text-2xl font-black text-white">42<span className="text-cyan-400 text-lg">ms</span></p>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Feature Grid */}
        <div className="grid md:grid-cols-3 gap-6 mt-32 relative z-10">
          <FeatureCard 
            icon={<ShieldAlert className="text-emerald-400 w-8 h-8" />}
            title="Structural Forgery Check"
            desc="Detects pixel-level tampering in amounts, dates, and names using CNN-based feature extraction."
          />
          <FeatureCard 
            icon={<Fingerprint className="text-cyan-400 w-8 h-8" />}
            title="Signature Verification"
            desc="Trained on CEDAR dataset patterns to validate complex biographical curves and flag anomalies."
          />
          <FeatureCard 
            icon={<Activity className="text-purple-400 w-8 h-8" />}
            title="Live Edge Scoring"
            desc="Generates instant Low, Medium, High, or Critical risk profiles to automate banking funnels."
          />
        </div>
      </main>

      <style dangerouslySetInnerHTML={{__html: `
        @keyframes scan {
          0% { top: 0%; opacity: 0; }
          10% { opacity: 1; }
          90% { opacity: 1; }
          100% { top: 100%; opacity: 0; }
        }
      `}} />
    </div>
  );
}

function FeatureCard({ icon, title, desc }: { icon: any, title: string, desc: string }) {
  return (
    <div className="p-1 rounded-2xl bg-gradient-to-b from-white/5 to-transparent">
      <div className="bg-slate-950/80 backdrop-blur-xl h-full p-8 rounded-xl border border-white/5 flex flex-col items-start hover:bg-slate-900 transition-colors">
        <div className="p-3 bg-white/5 rounded-lg mb-6 border border-white/5">
          {icon}
        </div>
        <h3 className="text-xl font-bold text-white mb-3">{title}</h3>
        <p className="text-slate-400 leading-relaxed text-sm">
          {desc}
        </p>
      </div>
    </div>
  );
}
