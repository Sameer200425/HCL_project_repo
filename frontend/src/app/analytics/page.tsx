'use client';

import { useEffect, useState, useMemo } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import {
  Shield, BarChart3, Upload, History, LogOut, User,
  TrendingUp, AlertTriangle, CheckCircle, Activity,
  Network, Settings, PieChart as PieIcon, ArrowUpRight, ArrowDownRight,
} from 'lucide-react';
import {
  LineChart, Line, BarChart, Bar, PieChart, Pie, Cell, AreaChart, Area,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from 'recharts';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import { useAuthStore, usePredictionStore, type Prediction } from '@/lib/store';
import { useToast } from '@/hooks/use-toast';

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */

const RISK_COLORS: Record<string, string> = {
  low: '#22c55e', medium: '#eab308', high: '#f97316', critical: '#ef4444',
};

const CLASS_COLORS: Record<string, string> = {
  genuine: '#22c55e', fraud: '#ef4444', tampered: '#f97316', forged: '#8b5cf6',
};

function buildDailyTrend(predictions: Prediction[]) {
  const map = new Map<string, { total: number; fraud: number; genuine: number; avgConf: number; confCount: number }>();
  for (const p of predictions) {
    const day = new Date(p.created_at).toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
    const entry = map.get(day) || { total: 0, fraud: 0, genuine: 0, avgConf: 0, confCount: 0 };
    entry.total += 1;
    if (['fraud', 'tampered', 'forged'].includes(p.predicted_class)) entry.fraud += 1;
    else entry.genuine += 1;
    entry.avgConf += p.confidence;
    entry.confCount += 1;
    map.set(day, entry);
  }
  return Array.from(map.entries()).map(([date, v]) => ({
    date,
    total: v.total,
    fraud: v.fraud,
    genuine: v.genuine,
    avgConfidence: Number(((v.avgConf / v.confCount) * 100).toFixed(1)),
  }));
}

function buildClassBreakdown(predictions: Prediction[]) {
  const counts: Record<string, number> = {};
  for (const p of predictions) {
    counts[p.predicted_class] = (counts[p.predicted_class] || 0) + 1;
  }
  return Object.entries(counts).map(([name, value]) => ({ name, value }));
}

function buildModelPerformance(predictions: Prediction[]) {
  const map = new Map<string, { count: number; totalConf: number; totalTime: number; fraudCount: number }>();
  for (const p of predictions) {
    const entry = map.get(p.model_name) || { count: 0, totalConf: 0, totalTime: 0, fraudCount: 0 };
    entry.count += 1;
    entry.totalConf += p.confidence;
    entry.totalTime += p.inference_time_ms;
    if (['fraud', 'tampered', 'forged'].includes(p.predicted_class)) entry.fraudCount += 1;
    map.set(p.model_name, entry);
  }
  return Array.from(map.entries()).map(([model, v]) => ({
    model,
    predictions: v.count,
    avgConfidence: Number(((v.totalConf / v.count) * 100).toFixed(1)),
    avgInference: Number((v.totalTime / v.count).toFixed(1)),
    fraudRate: Number(((v.fraudCount / v.count) * 100).toFixed(1)),
  }));
}

function buildRiskTimeline(predictions: Prediction[]) {
  const map = new Map<string, Record<string, number>>();
  for (const p of predictions) {
    const day = new Date(p.created_at).toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
    const entry = map.get(day) || { low: 0, medium: 0, high: 0, critical: 0 };
    entry[p.risk_level] = (entry[p.risk_level] || 0) + 1;
    map.set(day, entry);
  }
  return Array.from(map.entries()).map(([date, v]) => ({ date, ...v }));
}

function buildHourlyDistribution(predictions: Prediction[]) {
  const hours = Array.from({ length: 24 }, (_, i) => ({ hour: `${i.toString().padStart(2, '0')}:00`, count: 0 }));
  for (const p of predictions) {
    const h = new Date(p.created_at).getHours();
    hours[h].count += 1;
  }
  return hours;
}

/* ------------------------------------------------------------------ */
/* Page Component                                                      */
/* ------------------------------------------------------------------ */

export default function AnalyticsPage() {
  const router = useRouter();
  const { user, token, logout, loadUser, authDisabled } = useAuthStore();
  const { predictions, fetchHistory, fetchStats } = usePredictionStore();
  const { toast } = useToast();
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const init = async () => {
      if (!token && !authDisabled) { router.push('/login'); return; }
      await loadUser();
      await fetchHistory();
      await fetchStats();
      setIsLoading(false);
    };
    init();
  }, [token, authDisabled, router, loadUser, fetchHistory, fetchStats]);

  const handleLogout = () => {
    logout();
    toast({ title: 'Logged out', description: 'You have been logged out.' });
    router.push('/');
  };

  /* Derived data */
  const trendData     = useMemo(() => buildDailyTrend(predictions), [predictions]);
  const classData     = useMemo(() => buildClassBreakdown(predictions), [predictions]);
  const modelPerf     = useMemo(() => buildModelPerformance(predictions), [predictions]);
  const riskTimeline  = useMemo(() => buildRiskTimeline(predictions), [predictions]);
  const hourlyDist    = useMemo(() => buildHourlyDistribution(predictions), [predictions]);

  const totalPredictions = predictions.length;
  const fraudCount = predictions.filter(p => ['fraud', 'tampered', 'forged'].includes(p.predicted_class)).length;
  const avgConfidence = totalPredictions > 0
    ? predictions.reduce((s, p) => s + p.confidence, 0) / totalPredictions
    : 0;
  const avgInference = totalPredictions > 0
    ? predictions.reduce((s, p) => s + p.inference_time_ms, 0) / totalPredictions
    : 0;
  const dashboardCardClass = 'bg-[#0f172a]/90 border-slate-800 text-slate-100 shadow-xl';
  const mutedTextClass = 'text-slate-400';

  if (isLoading) {
    return (
      <div className="min-h-screen bg-[#0B1220] flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-emerald-500" />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[#0B1220] text-slate-200 relative overflow-hidden">
      <div className="pointer-events-none absolute -top-40 -left-40 h-[28rem] w-[28rem] rounded-full bg-cyan-500/15 blur-3xl" />
      <div className="pointer-events-none absolute -bottom-40 -right-40 h-[30rem] w-[30rem] rounded-full bg-emerald-500/10 blur-3xl" />
      {/* Header */}
      <header className="bg-[#0f172a]/90 backdrop-blur-xl border-b border-slate-800 relative z-10">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
          <Link href="/dashboard" className="flex items-center gap-2">
            
            <span className="text-xl font-bold"></span>
          </Link>
          <nav className="flex items-center gap-4">
            
            
            
            
            
            <div className="flex items-center gap-2 ml-4 pl-4 border-l border-slate-800">
              <User className="h-4 w-4 text-slate-400" />
              <span className="text-sm text-slate-200">{user?.username || 'User'}</span>
              <Button variant="ghost" size="sm" onClick={handleLogout} className="text-slate-300 hover:text-white hover:bg-slate-800"><LogOut className="h-4 w-4" /></Button>
            </div>
          </nav>
        </div>
      </header>

      {/* Main */}
      <main className="max-w-7xl mx-auto px-4 py-8 relative z-10">
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-3xl font-bold text-white">Analytics</h1>
            <p className="text-slate-400">Detailed fraud detection analytics and insights</p>
          </div>
        </div>

        {/* KPI Row */}
        <div className="grid md:grid-cols-4 gap-6 mb-8">
          <Card className={dashboardCardClass}>
            <CardHeader className="pb-2">
              <CardDescription className={mutedTextClass}>Total Predictions</CardDescription>
              <CardTitle className="text-3xl">{totalPredictions}</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-xs text-slate-400 flex items-center gap-1">
                <ArrowUpRight className="h-3 w-3 text-green-500" /> All time
              </p>
            </CardContent>
          </Card>
          <Card className={dashboardCardClass}>
            <CardHeader className="pb-2">
              <CardDescription className={mutedTextClass}>Fraud Rate</CardDescription>
              <CardTitle className="text-3xl text-red-400">
                {totalPredictions > 0 ? ((fraudCount / totalPredictions) * 100).toFixed(1) : '0'}%
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-xs text-slate-400 flex items-center gap-1">
                <AlertTriangle className="h-3 w-3 text-red-500" /> {fraudCount} flagged
              </p>
            </CardContent>
          </Card>
          <Card className={dashboardCardClass}>
            <CardHeader className="pb-2">
              <CardDescription className={mutedTextClass}>Avg Confidence</CardDescription>
              <CardTitle className="text-3xl text-cyan-400">{(avgConfidence * 100).toFixed(1)}%</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-xs text-slate-400 flex items-center gap-1">
                <CheckCircle className="h-3 w-3 text-blue-500" /> Across all models
              </p>
            </CardContent>
          </Card>
          <Card className={dashboardCardClass}>
            <CardHeader className="pb-2">
              <CardDescription className={mutedTextClass}>Avg Inference</CardDescription>
              <CardTitle className="text-3xl text-emerald-400">{avgInference.toFixed(1)}ms</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-xs text-slate-400 flex items-center gap-1">
                <Activity className="h-3 w-3 text-purple-500" /> Response time
              </p>
            </CardContent>
          </Card>
        </div>

        {/* Tabbed Charts */}
        <Tabs defaultValue="trends" className="space-y-6">
          <TabsList className="grid w-full grid-cols-5 bg-slate-900/70 border border-slate-800">
            <TabsTrigger value="trends">Trends</TabsTrigger>
            <TabsTrigger value="classes">Classes</TabsTrigger>
            <TabsTrigger value="risk">Risk</TabsTrigger>
            <TabsTrigger value="models">Models</TabsTrigger>
            <TabsTrigger value="volume">Volume</TabsTrigger>
          </TabsList>

          {/* Trends Tab */}
          <TabsContent value="trends">
            <div className="grid md:grid-cols-1 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle>Daily Prediction Trends</CardTitle>
                  <CardDescription>Volume and fraud detection over time</CardDescription>
                </CardHeader>
                <CardContent>
                  {trendData.length > 0 ? (
                    <ResponsiveContainer width="100%" height={350}>
                      <AreaChart data={trendData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="date" fontSize={12} />
                        <YAxis allowDecimals={false} fontSize={12} />
                        <Tooltip />
                        <Legend />
                        <Area type="monotone" dataKey="genuine" stackId="1" fill="#22c55e" stroke="#22c55e" fillOpacity={0.3} name="Genuine" />
                        <Area type="monotone" dataKey="fraud" stackId="1" fill="#ef4444" stroke="#ef4444" fillOpacity={0.3} name="Fraud" />
                      </AreaChart>
                    </ResponsiveContainer>
                  ) : (
                    <div className="text-center py-12 text-muted-foreground">No prediction data yet</div>
                  )}
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Confidence Trend</CardTitle>
                  <CardDescription>Average prediction confidence per day</CardDescription>
                </CardHeader>
                <CardContent>
                  {trendData.length > 0 ? (
                    <ResponsiveContainer width="100%" height={250}>
                      <LineChart data={trendData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="date" fontSize={12} />
                        <YAxis domain={[0, 100]} fontSize={12} />
                        <Tooltip />
                        <Line type="monotone" dataKey="avgConfidence" stroke="#6366f1" strokeWidth={2} name="Avg Confidence %" dot={{ r: 3 }} />
                      </LineChart>
                    </ResponsiveContainer>
                  ) : (
                    <div className="text-center py-12 text-muted-foreground">No data yet</div>
                  )}
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* Classes Tab */}
          <TabsContent value="classes">
            <div className="grid md:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle>Class Distribution</CardTitle>
                  <CardDescription>Breakdown by predicted class</CardDescription>
                </CardHeader>
                <CardContent>
                  {classData.length > 0 ? (
                    <ResponsiveContainer width="100%" height={300}>
                      <PieChart>
                        <Pie data={classData} cx="50%" cy="50%" innerRadius={60} outerRadius={100} paddingAngle={4} dataKey="value"
                          label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}>
                          {classData.map((e) => (
                            <Cell key={e.name} fill={CLASS_COLORS[e.name] || '#94a3b8'} />
                          ))}
                        </Pie>
                        <Tooltip />
                      </PieChart>
                    </ResponsiveContainer>
                  ) : (
                    <div className="text-center py-12 text-muted-foreground">No data</div>
                  )}
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Class Counts</CardTitle>
                  <CardDescription>Total predictions per class</CardDescription>
                </CardHeader>
                <CardContent>
                  {classData.length > 0 ? (
                    <ResponsiveContainer width="100%" height={300}>
                      <BarChart data={classData} layout="vertical">
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis type="number" allowDecimals={false} fontSize={12} />
                        <YAxis type="category" dataKey="name" fontSize={12} width={80} />
                        <Tooltip />
                        <Bar dataKey="value" name="Count" radius={[0, 4, 4, 0]}>
                          {classData.map((e) => (
                            <Cell key={e.name} fill={CLASS_COLORS[e.name] || '#94a3b8'} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  ) : (
                    <div className="text-center py-12 text-muted-foreground">No data</div>
                  )}
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* Risk Tab */}
          <TabsContent value="risk">
            <div className="grid md:grid-cols-2 gap-6">
              <Card className="md:col-span-2">
                <CardHeader>
                  <CardTitle>Risk Level Timeline</CardTitle>
                  <CardDescription>Risk distribution over time</CardDescription>
                </CardHeader>
                <CardContent>
                  {riskTimeline.length > 0 ? (
                    <ResponsiveContainer width="100%" height={300}>
                      <BarChart data={riskTimeline}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="date" fontSize={12} />
                        <YAxis allowDecimals={false} fontSize={12} />
                        <Tooltip />
                        <Legend />
                        <Bar dataKey="low" stackId="a" fill={RISK_COLORS.low} name="Low" />
                        <Bar dataKey="medium" stackId="a" fill={RISK_COLORS.medium} name="Medium" />
                        <Bar dataKey="high" stackId="a" fill={RISK_COLORS.high} name="High" />
                        <Bar dataKey="critical" stackId="a" fill={RISK_COLORS.critical} name="Critical" />
                      </BarChart>
                    </ResponsiveContainer>
                  ) : (
                    <div className="text-center py-12 text-muted-foreground">No data</div>
                  )}
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Risk Summary</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  {['low', 'medium', 'high', 'critical'].map(level => {
                    const count = predictions.filter(p => p.risk_level === level).length;
                    const pct = totalPredictions > 0 ? (count / totalPredictions) * 100 : 0;
                    return (
                      <div key={level}>
                        <div className="flex justify-between text-sm mb-1">
                          <span className="capitalize">{level}</span>
                          <span>{count} ({pct.toFixed(1)}%)</span>
                        </div>
                        <Progress value={pct} className={`[&>div]:bg-${level === 'low' ? 'green' : level === 'medium' ? 'yellow' : level === 'high' ? 'orange' : 'red'}-500`} />
                      </div>
                    );
                  })}
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>High-Risk Alerts</CardTitle>
                  <CardDescription>Recent critical & high-risk predictions</CardDescription>
                </CardHeader>
                <CardContent>
                  {predictions.filter(p => ['high', 'critical'].includes(p.risk_level)).length === 0 ? (
                    <div className="text-center py-8 text-muted-foreground">
                      <CheckCircle className="h-10 w-10 mx-auto mb-2 text-green-400" />
                      No high-risk alerts
                    </div>
                  ) : (
                    <div className="space-y-2 max-h-[250px] overflow-y-auto">
                      {predictions
                        .filter(p => ['high', 'critical'].includes(p.risk_level))
                        .slice(0, 10)
                        .map(p => (
                          <div key={p.id} className="flex items-center justify-between p-3 bg-red-50 rounded-lg">
                            <div>
                              <p className="font-medium text-sm">{p.predicted_class}</p>
                              <p className="text-xs text-muted-foreground">{p.model_name}</p>
                            </div>
                            <span className={`text-xs px-2 py-1 rounded-full font-medium ${
                              p.risk_level === 'critical' ? 'bg-red-500 text-white' : 'bg-orange-500 text-white'
                            }`}>{p.risk_level}</span>
                          </div>
                        ))}
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* Models Tab */}
          <TabsContent value="models">
            <div className="grid md:grid-cols-2 gap-6">
              <Card className="md:col-span-2">
                <CardHeader>
                  <CardTitle>Model Performance Comparison</CardTitle>
                  <CardDescription>Average confidence and inference time per model</CardDescription>
                </CardHeader>
                <CardContent>
                  {modelPerf.length > 0 ? (
                    <ResponsiveContainer width="100%" height={300}>
                      <BarChart data={modelPerf}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="model" fontSize={12} />
                        <YAxis yAxisId="left" fontSize={12} />
                        <YAxis yAxisId="right" orientation="right" fontSize={12} />
                        <Tooltip />
                        <Legend />
                        <Bar yAxisId="left" dataKey="avgConfidence" fill="#3b82f6" radius={[4, 4, 0, 0]} name="Avg Confidence %" />
                        <Bar yAxisId="right" dataKey="avgInference" fill="#a855f7" radius={[4, 4, 0, 0]} name="Avg Inference (ms)" />
                      </BarChart>
                    </ResponsiveContainer>
                  ) : (
                    <div className="text-center py-12 text-muted-foreground">No data</div>
                  )}
                </CardContent>
              </Card>

              {modelPerf.length > 0 && (
                <Card className="md:col-span-2">
                  <CardHeader>
                    <CardTitle>Model Radar</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <ResponsiveContainer width="100%" height={300}>
                      <RadarChart data={modelPerf}>
                        <PolarGrid />
                        <PolarAngleAxis dataKey="model" fontSize={12} />
                        <PolarRadiusAxis angle={30} domain={[0, 100]} />
                        <Radar name="Confidence" dataKey="avgConfidence" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.3} />
                        <Radar name="Fraud Rate" dataKey="fraudRate" stroke="#ef4444" fill="#ef4444" fillOpacity={0.2} />
                        <Legend />
                        <Tooltip />
                      </RadarChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>
              )}
            </div>
          </TabsContent>

          {/* Volume Tab */}
          <TabsContent value="volume">
            <Card>
              <CardHeader>
                <CardTitle>Hourly Prediction Volume</CardTitle>
                <CardDescription>Distribution of predictions by hour of day</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={hourlyDist}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="hour" fontSize={11} />
                    <YAxis allowDecimals={false} fontSize={12} />
                    <Tooltip />
                    <Bar dataKey="count" fill="#6366f1" radius={[4, 4, 0, 0]} name="Predictions" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </main>
    </div>
  );
}
