'use client';

import { useEffect, useState, useMemo } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { 
  Shield, BarChart3, Upload, History, LogOut, User,
  TrendingUp, AlertTriangle, CheckCircle, FileSearch, Activity
} from 'lucide-react';
import {
  LineChart, Line, BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { useAuthStore, usePredictionStore, type Prediction } from '@/lib/store';
import { useToast } from '@/hooks/use-toast';

interface PredictionStats {
  total_predictions: number;
  fraud_detected: number;
  genuine_documents: number;
  avg_confidence: number;
  by_model: Record<string, number>;
  by_class?: Record<string, number>;
  by_risk_level: Record<string, number>;
}

const RISK_COLORS: Record<string, string> = {
  low: '#22c55e',
  medium: '#eab308',
  high: '#f97316',
  critical: '#ef4444',
};

const CLASS_COLORS: Record<string, string> = {
  genuine: '#22c55e',
  fraud: '#ef4444',
  tampered: '#f97316',
  forged: '#8b5cf6',
};

/** Group predictions by date and produce trend data for charts. */
function buildTrendData(predictions: Prediction[]) {
  const byDate = new Map<string, { total: number; fraud: number; genuine: number }>();
  const sorted = [...predictions].sort(
    (a, b) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime()
  );
  for (const p of sorted) {
    const day = new Date(p.created_at).toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
    const entry = byDate.get(day) || { total: 0, fraud: 0, genuine: 0 };
    entry.total += 1;
    if (['fraud', 'tampered', 'forged'].includes(p.predicted_class)) entry.fraud += 1;
    else entry.genuine += 1;
    byDate.set(day, entry);
  }
  return Array.from(byDate.entries()).map(([date, v]) => ({ date, ...v }));
}

export default function DashboardPage() {
  const router = useRouter();
  const { user, token, logout, loadUser, authDisabled } = useAuthStore();
  const { fetchHistory, fetchStats, predictions } = usePredictionStore();
  const { toast } = useToast();
  
  const [stats, setStats] = useState<PredictionStats | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  useEffect(() => {
    if (!mounted) return;
    
    const init = async () => {
      try {
        if (!token && !authDisabled) {
          router.push('/login');
          return;
        }
      
        await loadUser();
        await fetchHistory();
      
        // Fetch stats
        try {
          const result = await fetchStats();
          if (result.success) {
            setStats(result.data as PredictionStats);
          }
        } catch (error) {
          console.error('Failed to fetch stats:', error);
        }
      } catch (error) {
        console.error('Dashboard init error:', error);
      } finally {
        setIsLoading(false);
      }
    };
    
    init();
  }, [mounted, token, authDisabled, router, loadUser, fetchHistory, fetchStats]);

  const handleLogout = () => {
    logout();
    toast({
      title: "Logged out",
      description: "You have been logged out successfully.",
    });
    router.push('/');
  };

  // Derive chart data - must be called unconditionally (React hooks rules)
  const trendData = useMemo(() => buildTrendData(predictions), [predictions]);

  const classData = useMemo(() => {
    if (!stats?.by_class) return [];
    return Object.entries(stats.by_class).map(([name, value]) => ({ name, value }));
  }, [stats]);

  const modelData = useMemo(() => {
    if (!stats?.by_model) return [];
    return Object.entries(stats.by_model).map(([name, count]) => ({ name, count }));
  }, [stats]);

  const fraudRate = stats?.total_predictions 
    ? ((stats.fraud_detected / stats.total_predictions) * 100).toFixed(1) 
    : '0';

  if (!mounted || isLoading) {
    return (
      <div className="min-h-screen bg-slate-50 flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
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
              <Button variant="ghost" size="sm" onClick={handleLogout}>
                <LogOut className="h-4 w-4" />
              </Button>
            </div>
          </nav>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 py-8">
        <h1 className="text-3xl font-bold mb-8">Dashboard</h1>
        
        {/* Stats Cards */}
        <div className="grid md:grid-cols-4 gap-6 mb-8">
          <Card>
            <CardHeader className="pb-2">
              <CardDescription>Total Scans</CardDescription>
              <CardTitle className="text-3xl flex items-center gap-2">
                <FileSearch className="h-6 w-6 text-blue-500" />
                {stats?.total_predictions || 0}
              </CardTitle>
            </CardHeader>
          </Card>
          
          <Card>
            <CardHeader className="pb-2">
              <CardDescription>Fraud Detected</CardDescription>
              <CardTitle className="text-3xl flex items-center gap-2 text-red-600">
                <AlertTriangle className="h-6 w-6" />
                {stats?.fraud_detected || 0}
              </CardTitle>
            </CardHeader>
          </Card>
          
          <Card>
            <CardHeader className="pb-2">
              <CardDescription>Genuine Documents</CardDescription>
              <CardTitle className="text-3xl flex items-center gap-2 text-green-600">
                <CheckCircle className="h-6 w-6" />
                {stats?.genuine_documents || 0}
              </CardTitle>
            </CardHeader>
          </Card>
          
          <Card>
            <CardHeader className="pb-2">
              <CardDescription>Avg Confidence</CardDescription>
              <CardTitle className="text-3xl flex items-center gap-2">
                <TrendingUp className="h-6 w-6 text-purple-500" />
                {stats?.avg_confidence ? `${(stats.avg_confidence * 100).toFixed(0)}%` : 'N/A'}
              </CardTitle>
            </CardHeader>
          </Card>
        </div>

        {/* Interactive Charts */}
        {predictions.length > 0 && (
          <div className="grid md:grid-cols-2 gap-6 mb-8">
            {/* Fraud Trend Line Chart */}
            <Card className="md:col-span-2">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <TrendingUp className="h-5 w-5 text-blue-500" /> Prediction Trends
                </CardTitle>
                <CardDescription>Daily prediction volume &amp; fraud detections</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={280}>
                  <LineChart data={trendData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" fontSize={12} />
                    <YAxis allowDecimals={false} fontSize={12} />
                    <Tooltip />
                    <Legend />
                    <Line type="monotone" dataKey="total" stroke="#3b82f6" strokeWidth={2} name="Total" />
                    <Line type="monotone" dataKey="fraud" stroke="#ef4444" strokeWidth={2} name="Fraud" />
                    <Line type="monotone" dataKey="genuine" stroke="#22c55e" strokeWidth={2} name="Genuine" />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            {/* Class Distribution Pie Chart */}
            {classData.length > 0 && (
              <Card>
                <CardHeader>
                  <CardTitle>Class Distribution</CardTitle>
                  <CardDescription>Breakdown by predicted class</CardDescription>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={260}>
                    <PieChart>
                      <Pie
                        data={classData}
                        cx="50%"
                        cy="50%"
                        innerRadius={50}
                        outerRadius={90}
                        paddingAngle={4}
                        dataKey="value"
                        label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                      >
                        {classData.map((entry) => (
                          <Cell key={entry.name} fill={CLASS_COLORS[entry.name] || '#94a3b8'} />
                        ))}
                      </Pie>
                      <Tooltip />
                    </PieChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            )}

            {/* Model Usage Bar Chart */}
            {modelData.length > 0 && (
              <Card>
                <CardHeader>
                  <CardTitle>Model Usage</CardTitle>
                  <CardDescription>Predictions per model</CardDescription>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={260}>
                    <BarChart data={modelData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="name" fontSize={12} />
                      <YAxis allowDecimals={false} fontSize={12} />
                      <Tooltip />
                      <Bar dataKey="count" fill="#6366f1" radius={[4, 4, 0, 0]} name="Predictions" />
                    </BarChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            )}
          </div>
        )}

        {/* Risk Distribution & Quick Actions */}
        <div className="grid md:grid-cols-2 gap-6 mb-8">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <BarChart3 className="h-5 w-5" /> Risk Distribution
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span>Low Risk</span>
                  <span className="text-green-600">{stats?.by_risk_level?.low || 0}</span>
                </div>
                <Progress 
                  value={stats?.total_predictions ? ((stats.by_risk_level?.low || 0) / stats.total_predictions) * 100 : 0} 
                  className="bg-green-100 [&>div]:bg-green-500"
                />
              </div>
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span>Medium Risk</span>
                  <span className="text-yellow-600">{stats?.by_risk_level?.medium || 0}</span>
                </div>
                <Progress 
                  value={stats?.total_predictions ? ((stats.by_risk_level?.medium || 0) / stats.total_predictions) * 100 : 0} 
                  className="bg-yellow-100 [&>div]:bg-yellow-500"
                />
              </div>
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span>High Risk</span>
                  <span className="text-orange-600">{stats?.by_risk_level?.high || 0}</span>
                </div>
                <Progress 
                  value={stats?.total_predictions ? ((stats.by_risk_level?.high || 0) / stats.total_predictions) * 100 : 0} 
                  className="bg-orange-100 [&>div]:bg-orange-500"
                />
              </div>
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span>Critical</span>
                  <span className="text-red-600">{stats?.by_risk_level?.critical || 0}</span>
                </div>
                <Progress 
                  value={stats?.total_predictions ? ((stats.by_risk_level?.critical || 0) / stats.total_predictions) * 100 : 0} 
                  className="bg-red-100 [&>div]:bg-red-500"
                />
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Quick Actions</CardTitle>
              <CardDescription>Get started with document analysis</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <Link href="/predict" className="block">
                <Button className="w-full justify-start h-16 text-lg" size="lg">
                  <Upload className="h-6 w-6 mr-3" />
                  Analyze New Document
                </Button>
              </Link>
              <Link href="/history" className="block">
                <Button variant="outline" className="w-full justify-start h-16 text-lg" size="lg">
                  <History className="h-6 w-6 mr-3" />
                  View Scan History
                </Button>
              </Link>
            </CardContent>
          </Card>
        </div>

        {/* Recent Predictions */}
        <Card>
          <CardHeader>
            <CardTitle>Recent Predictions</CardTitle>
            <CardDescription>Your latest document analyses</CardDescription>
          </CardHeader>
          <CardContent>
            {predictions.length === 0 ? (
              <div className="text-center py-8 text-muted-foreground">
                <FileSearch className="h-12 w-12 mx-auto mb-4 opacity-50" />
                <p>No predictions yet. Upload a document to get started.</p>
                <Link href="/predict">
                  <Button className="mt-4">
                    <Upload className="h-4 w-4 mr-2" /> Analyze Document
                  </Button>
                </Link>
              </div>
            ) : (
              <div className="space-y-3">
                {predictions.slice(0, 5).map((pred) => (
                  <div 
                    key={pred.id} 
                    className="flex items-center justify-between p-4 bg-slate-50 rounded-lg"
                  >
                    <div className="flex items-center gap-4">
                      <div className={`p-2 rounded-full ${
                        pred.risk_level === 'critical' ? 'bg-red-100 text-red-600' :
                        pred.risk_level === 'high' ? 'bg-orange-100 text-orange-600' :
                        pred.risk_level === 'medium' ? 'bg-yellow-100 text-yellow-600' :
                        'bg-green-100 text-green-600'
                      }`}>
                        {pred.risk_level === 'low' ? (
                          <CheckCircle className="h-5 w-5" />
                        ) : (
                          <AlertTriangle className="h-5 w-5" />
                        )}
                      </div>
                      <div>
                        <p className="font-medium">{pred.predicted_class}</p>
                        <p className="text-sm text-muted-foreground">{pred.model_name}</p>
                      </div>
                    </div>
                    <div className="text-right">
                      <p className="font-medium">{(pred.confidence * 100).toFixed(1)}%</p>
                      <p className="text-sm text-muted-foreground capitalize">{pred.risk_level} risk</p>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </main>
    </div>
  );
}
