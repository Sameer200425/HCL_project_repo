'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { Network, Activity, Radar, AlertTriangle, ShieldCheck, User, LogOut, Search, Maximize, ZoomIn, ZoomOut, Filter } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { useAuthStore } from '@/lib/store';
import { useToast } from '@/hooks/use-toast';

// Simple mock data for graph visualization using Recharts ScatterChart or just custom SVG
const mockNodes = [
  { id: 1, x: 200, y: 150, type: 'user', risk: 'low', name: 'User 1294', radius: 15 },
  { id: 2, x: 300, y: 200, type: 'account', risk: 'medium', name: 'Acc 4829', radius: 10 },
  { id: 3, x: 250, y: 300, type: 'user', risk: 'critical', name: 'User 8831 (Mule)', radius: 18 },
  { id: 4, x: 400, y: 150, type: 'transaction', risk: 'low', name: 'Txn 9912', radius: 8 },
  { id: 5, x: 150, y: 280, type: 'account', risk: 'high', name: 'Acc 1102', radius: 12 },
  { id: 6, x: 350, y: 320, type: 'user', risk: 'high', name: 'User 9921', radius: 15 },
  { id: 7, x: 450, y: 250, type: 'account', risk: 'critical', name: 'Acc 6631', radius: 14 },
];

const mockEdges = [
  { source: 1, target: 2 },
  { source: 2, target: 4 },
  { source: 1, target: 5 },
  { source: 3, target: 2 },
  { source: 3, target: 5 },
  { source: 3, target: 6 },
  { source: 6, target: 7 },
];

export default function GraphPage() {
  const router = useRouter();
  const { user, token, logout, authDisabled } = useAuthStore();
  const { toast } = useToast();
  
  const [zoom, setZoom] = useState(1);
  const [searchQuery, setSearchQuery] = useState('');
  
  useEffect(() => {
    if (!token && !authDisabled) {
      router.push('/login');
    }
  }, [token, authDisabled, router]);

  const handleLogout = () => {
    logout();
    router.push('/');
  };

  const getRiskColor = (risk: string) => {
    switch(risk) {
      case 'critical': return '#e11d48'; // rose-600
      case 'high': return '#f59e0b'; // amber-500
      case 'medium': return '#eab308'; // yellow-500
      case 'low': return '#10b981'; // emerald-500
      default: return '#94a3b8'; // slate-400
    }
  };

  return (
    <div className="min-h-screen bg-[#0B1220] text-slate-100 flex flex-col">
      <header className="border-b border-white/10 backdrop-blur-sm bg-[#0B1220]/80 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
          <Link href="/dashboard" className="flex items-center gap-3 group">
            <div className="h-9 w-9 rounded-lg bg-gradient-to-br from-cyan-400 to-emerald-400 flex items-center justify-center shadow-[0_0_18px_rgba(52,211,153,0.35)] group-hover:shadow-[0_0_24px_rgba(52,211,153,0.5)] transition-shadow">
              <Network className="h-5 w-5 text-slate-950" />
            </div>
            <span className="font-bold text-lg tracking-wide bg-clip-text text-transparent bg-gradient-to-r from-white to-slate-400 hidden sm:block">
              Graph Analysis
            </span>
          </Link>

          <div className="flex items-center gap-4 text-sm">
            <div className="hidden md:flex items-center gap-2 px-3 py-1.5 rounded-full border border-emerald-300/30 bg-emerald-300/10 text-emerald-200">
              <ShieldCheck className="h-4 w-4" />
              Neo4j Connected
            </div>
            <div className="flex items-center gap-2 ml-2 pl-4 border-l border-white/15">
              <User className="h-4 w-4 text-slate-300" />
              <span className="text-slate-200">{user?.username || 'User'}</span>
              <Button variant="ghost" size="sm" onClick={handleLogout} className="text-slate-200 hover:text-white hover:bg-white/10 ml-1">
                <LogOut className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </div>
      </header>

      <main className="flex-1 max-w-7xl w-full mx-auto p-4 flex flex-col lg:flex-row gap-6">
        {/* Sidebar Controls */}
        <aside className="w-full lg:w-80 flex flex-col gap-4">
          <Card className="bg-slate-900/40 border-slate-800/60 backdrop-blur-xl">
            <CardHeader className="pb-3">
              <CardTitle className="text-lg flex items-center gap-2">
                <Filter className="h-4 w-4 text-emerald-400" />
                Network Filter
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="relative">
                <Search className="absolute left-3 top-3 h-4 w-4 text-slate-500" />
                <Input 
                  placeholder="Search user ID or account..." 
                  className="pl-9 bg-slate-950/50 border-slate-800 text-slate-300 placeholder:text-slate-600 focus-visible:ring-emerald-500/50"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                />
              </div>
              <div className="space-y-2">
                <p className="text-xs font-semibold uppercase text-slate-500 tracking-wider">Risk Layers</p>
                <div className="flex items-center gap-2">
                  <input type="checkbox" id="risk-crit" className="rounded border-slate-700 bg-slate-900 text-rose-500" defaultChecked />
                  <label htmlFor="risk-crit" className="text-sm text-slate-300">Critical Entities</label>
                </div>
                <div className="flex items-center gap-2">
                  <input type="checkbox" id="risk-high" className="rounded border-slate-700 bg-slate-900 text-orange-500" defaultChecked />
                  <label htmlFor="risk-high" className="text-sm text-slate-300">High Risk</label>
                </div>
              </div>
              <Button className="w-full bg-emerald-500/20 hover:bg-emerald-500/30 text-emerald-300 border border-emerald-500/30">
                Run Louvain Detection
              </Button>
            </CardContent>
          </Card>

          <Card className="bg-slate-900/40 border-slate-800/60 backdrop-blur-xl flex-1">
            <CardHeader className="pb-3">
              <CardTitle className="text-lg">Graph Insights</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="p-3 rounded-lg bg-rose-500/10 border border-rose-500/20">
                  <div className="flex items-center gap-2 mb-1">
                    <Activity className="h-4 w-4 text-rose-400" />
                    <span className="font-medium text-rose-200">Mule Ring Detected</span>
                  </div>
                  <p className="text-xs text-rose-100/70">Cluster #84 showing circular transaction patterns linked to flagged User 8831.</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </aside>

        {/* Graph Display */}
        <Card className="flex-1 bg-[#050B14] border-slate-800/80 shadow-inner overflow-hidden relative">
          <div className="absolute top-4 right-4 z-10 flex flex-col gap-2 bg-slate-900/80 backdrop-blur border border-slate-800 p-1.5 rounded-lg">
            <Button variant="ghost" size="icon" onClick={() => setZoom(z => Math.min(z + 0.2, 3))} className="h-8 w-8 text-slate-400 hover:text-white">
              <ZoomIn className="h-4 w-4" />
            </Button>
            <Button variant="ghost" size="icon" onClick={() => setZoom(z => Math.max(z - 0.2, 0.5))} className="h-8 w-8 text-slate-400 hover:text-white">
              <ZoomOut className="h-4 w-4" />
            </Button>
            <Button variant="ghost" size="icon" onClick={() => setZoom(1)} className="h-8 w-8 text-slate-400 hover:text-white">
              <Maximize className="h-4 w-4" />
            </Button>
          </div>

          <div className="w-full h-[600px] sm:h-full min-h-[500px] flex items-center justify-center cursor-move overflow-hidden">
            <svg 
              width="100%" 
              height="100%" 
              viewBox="0 0 600 450"
            >
              <defs>
                <filter id="glow-critical" x="-20%" y="-20%" width="140%" height="140%">
                  <feGaussianBlur stdDeviation="4" result="blur" />
                  <feComposite in="SourceGraphic" in2="blur" operator="over" />
                </filter>
              </defs>

              <g transform={`scale(${zoom})`}>

              {/* Edges */}
              {mockEdges.map((edge, i) => {
                const source = mockNodes.find(n => n.id === edge.source);
                const target = mockNodes.find(n => n.id === edge.target);
                if (!source || !target) return null;
                return (
                  <line 
                    key={i} 
                    x1={source.x} y1={source.y} 
                    x2={target.x} y2={target.y} 
                    stroke="#334155" 
                    strokeWidth="2"
                    strokeOpacity="0.4"
                  />
                );
              })}

              {/* Nodes */}
              {mockNodes.map((node) => (
                <g key={node.id} transform={`translate(${node.x}, ${node.y})`}>
                  <circle 
                    r={node.radius + 4} 
                    fill={getRiskColor(node.risk)} 
                    opacity="0.15"
                    filter={node.risk === 'critical' ? 'url(#glow-critical)' : undefined}
                  >
                    {node.risk === 'critical' && (
                      <animate attributeName="r" values={`${node.radius + 4};${node.radius + 12};${node.radius + 4}`} dur="2s" repeatCount="indefinite" />
                    )}
                  </circle>
                  <circle 
                    r={node.radius} 
                    fill="#1e293b" 
                    stroke={getRiskColor(node.risk)} 
                    strokeWidth="3"
                  />
                  <text 
                    y={node.radius + 15} 
                    textAnchor="middle" 
                    fill="#cbd5e1" 
                    fontSize="10" 
                    fontWeight="500"
                  >
                    {node.name}
                  </text>
                </g>
              ))}
              </g>
            </svg>
          </div>
        </Card>
      </main>
    </div>
  );
}