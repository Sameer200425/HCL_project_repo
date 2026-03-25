'use client';

import { useEffect, useState, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import {
  Shield, BarChart3, Upload, History, LogOut, User,
  Activity, AlertTriangle, CheckCircle, XCircle,
  Settings, RefreshCw, Bell, TrendingUp, TrendingDown,
  Clock, Zap, Target, AlertCircle, FileText,
} from 'lucide-react';
import {
  LineChart, Line, BarChart, Bar, AreaChart, Area,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from 'recharts';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { useAuthStore } from '@/lib/store';
import { useToast } from '@/hooks/use-toast';
import { api } from '@/lib/api';

/* ------------------------------------------------------------------ */
/* Types                                                               */
/* ------------------------------------------------------------------ */

interface HealthStatus {
  status: 'HEALTHY' | 'DEGRADED' | 'CRITICAL';
  issues: string[];
  confidence_stats: {
    mean?: number;
    std?: number;
    min?: number;
    max?: number;
    low_confidence_rate?: number;
    critical_rate?: number;
    total_predictions?: number;
    error?: string;
  };
  drift_status: {
    drift_detected: boolean;
    recent_alerts: Alert[];
    samples_collected: number;
  };
  alerts: Alert[];
  timestamp: string;
}

interface Alert {
  type: string;
  message: string;
  timestamp: string;
  severity: 'WARNING' | 'CRITICAL' | 'INFO';
}

interface Metrics {
  total_predictions: number;
  mean_confidence: number | null;
  std_confidence: number | null;
  min_confidence: number | null;
  max_confidence: number | null;
  low_confidence_rate: number | null;
  critical_rate: number | null;
}

interface PredictionLog {
  timestamp: string;
  model_name: string;
  predicted_class: string;
  confidence: number;
  inference_time_ms: number;
}

/* ------------------------------------------------------------------ */
/* Component                                                           */
/* ------------------------------------------------------------------ */

export default function MonitoringPage() {
  const router = useRouter();
  const { user, token, logout } = useAuthStore();
  const { toast } = useToast();

  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [logs, setLogs] = useState<PredictionLog[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  const fetchMonitoringData = useCallback(async () => {
    try {
      const [healthRes, metricsRes, logsRes] = await Promise.all([
        api.get<HealthStatus>('/api/monitoring/health'),
        api.get<Metrics>('/api/monitoring/metrics'),
        api.get<{ logs: PredictionLog[]; total_count: number }>('/api/monitoring/logs?hours=24&limit=50'),
      ]);
      
      setHealth(healthRes);
      setMetrics(metricsRes);
      setLogs(logsRes.logs);
    } catch (error) {
      console.error('Failed to fetch monitoring data:', error);
      toast({
        title: 'Error',
        description: 'Failed to fetch monitoring data',
        variant: 'destructive',
      });
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, [toast]);

  useEffect(() => {
    fetchMonitoringData();
    // Auto-refresh every 30 seconds
    const interval = setInterval(fetchMonitoringData, 30000);
    return () => clearInterval(interval);
  }, [fetchMonitoringData]);

  const handleRefresh = () => {
    setRefreshing(true);
    fetchMonitoringData();
  };

  const handleClearAlerts = async () => {
    try {
      await api.post('/api/monitoring/clear-alerts', {});
      toast({ title: 'Success', description: 'All alerts cleared' });
      fetchMonitoringData();
    } catch {
      toast({ title: 'Error', description: 'Failed to clear alerts', variant: 'destructive' });
    }
  };

  const handleLogout = () => {
    logout();
    router.push('/login');
  };

  // Build chart data from logs
  const buildConfidenceTimeline = () => {
    return logs.map((log, i) => ({
      index: i,
      confidence: log.confidence * 100,
      time: new Date(log.timestamp).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }),
    })).reverse();
  };

  const buildInferenceTimeline = () => {
    return logs.map((log, i) => ({
      index: i,
      inference_ms: log.inference_time_ms,
      time: new Date(log.timestamp).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }),
    })).reverse();
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'HEALTHY': return 'text-green-500';
      case 'DEGRADED': return 'text-yellow-500';
      case 'CRITICAL': return 'text-red-500';
      default: return 'text-gray-500';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'HEALTHY': return <CheckCircle className="w-8 h-8 text-green-500" />;
      case 'DEGRADED': return <AlertTriangle className="w-8 h-8 text-yellow-500" />;
      case 'CRITICAL': return <XCircle className="w-8 h-8 text-red-500" />;
      default: return <Activity className="w-8 h-8 text-gray-500" />;
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'CRITICAL': return 'bg-red-500';
      case 'WARNING': return 'bg-yellow-500';
      default: return 'bg-blue-500';
    }
  };

  return (
    <div className="min-h-screen bg-[#0B1220] text-slate-200 relative overflow-hidden">
      <div className="pointer-events-none absolute -top-40 -left-40 h-[28rem] w-[28rem] rounded-full bg-cyan-500/15 blur-3xl" />
      <div className="pointer-events-none absolute -bottom-40 -right-40 h-[30rem] w-[30rem] rounded-full bg-emerald-500/10 blur-3xl" />
      {/* Header */}
      <header className="border-b border-slate-800 bg-[#0f172a]/90 backdrop-blur-xl relative z-10">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-gradient-to-br from-cyan-400 to-emerald-400 rounded-lg shadow-[0_0_18px_rgba(52,211,153,0.35)]">
                <Activity className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-white">MLOps Monitor</h1>
                <p className="text-sm text-slate-400">Real-time model health tracking</p>
              </div>
            </div>
            <nav className="flex items-center gap-2">
              <Link href="/dashboard">
                <Button variant="ghost" size="sm" className="text-slate-300 hover:text-white">
                  <Shield className="w-4 h-4 mr-2" />Dashboard
                </Button>
              </Link>
              <Link href="/predict">
                <Button variant="ghost" size="sm" className="text-slate-300 hover:text-white">
                  <Upload className="w-4 h-4 mr-2" />Predict
                </Button>
              </Link>
              <Link href="/analytics">
                <Button variant="ghost" size="sm" className="text-slate-300 hover:text-white">
                  <BarChart3 className="w-4 h-4 mr-2" />Analytics
                </Button>
              </Link>
              <Button variant="ghost" size="sm" onClick={handleLogout} className="text-slate-300 hover:text-white">
                <LogOut className="w-4 h-4 mr-2" />Logout
              </Button>
            </nav>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-8 relative z-10">
        {/* Actions & Status */}
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-4">
            {health && (
              <div className="flex items-center gap-2">
                {getStatusIcon(health.status)}
                <span className={`text-2xl font-bold ${getStatusColor(health.status)}`}>
                  {health.status}
                </span>
              </div>
            )}
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={handleRefresh}
              disabled={refreshing}
              className="border-slate-600 text-slate-300"
            >
              <RefreshCw className={`w-4 h-4 mr-2 ${refreshing ? 'animate-spin' : ''}`} />
              Refresh
            </Button>
            {health && health.alerts.length > 0 && (
              <Button
                variant="outline"
                size="sm"
                onClick={handleClearAlerts}
                className="border-yellow-600 text-yellow-400"
              >
                <Bell className="w-4 h-4 mr-2" />
                Clear Alerts ({health.alerts.length})
              </Button>
            )}
          </div>
        </div>

        {loading ? (
          <div className="flex items-center justify-center py-20">
            <RefreshCw className="w-8 h-8 text-emerald-500 animate-spin" />
          </div>
        ) : (
          <>
            {/* Overview Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
              {/* Total Predictions */}
              <Card className="bg-slate-800/50 border-slate-700">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm text-slate-400 flex items-center gap-2">
                    <Target className="w-4 h-4" /> Total Predictions
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-3xl font-bold text-white">
                    {metrics?.total_predictions || 0}
                  </div>
                  <p className="text-sm text-slate-500 mt-1">Last 24 hours</p>
                </CardContent>
              </Card>

              {/* Mean Confidence */}
              <Card className="bg-slate-800/50 border-slate-700">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm text-slate-400 flex items-center gap-2">
                    <TrendingUp className="w-4 h-4" /> Mean Confidence
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-3xl font-bold text-white">
                    {metrics?.mean_confidence != null
                      ? `${(metrics.mean_confidence * 100).toFixed(1)}%`
                      : 'N/A'}
                  </div>
                  <Progress
                    value={(metrics?.mean_confidence || 0) * 100}
                    className="mt-2 h-2"
                  />
                </CardContent>
              </Card>

              {/* Low Confidence Rate */}
              <Card className="bg-slate-800/50 border-slate-700">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm text-slate-400 flex items-center gap-2">
                    <AlertCircle className="w-4 h-4" /> Low Confidence Rate
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className={`text-3xl font-bold ${
                    (metrics?.low_confidence_rate || 0) > 0.2 ? 'text-red-400' : 'text-green-400'
                  }`}>
                    {metrics?.low_confidence_rate != null
                      ? `${(metrics.low_confidence_rate * 100).toFixed(1)}%`
                      : 'N/A'}
                  </div>
                  <p className="text-sm text-slate-500 mt-1">
                    {(metrics?.low_confidence_rate || 0) > 0.2 ? 'Above threshold' : 'Normal'}
                  </p>
                </CardContent>
              </Card>

              {/* Drift Status */}
              <Card className="bg-slate-800/50 border-slate-700">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm text-slate-400 flex items-center gap-2">
                    <Activity className="w-4 h-4" /> Drift Detection
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className={`text-3xl font-bold ${
                    health?.drift_status.drift_detected ? 'text-red-400' : 'text-green-400'
                  }`}>
                    {health?.drift_status.drift_detected ? 'DETECTED' : 'STABLE'}
                  </div>
                  <p className="text-sm text-slate-500 mt-1">
                    {health?.drift_status.samples_collected || 0} samples
                  </p>
                </CardContent>
              </Card>
            </div>

            {/* Tabs for different views */}
            <Tabs defaultValue="metrics" className="space-y-4">
              <TabsList className="bg-slate-800 border border-slate-700">
                <TabsTrigger value="metrics" className="data-[state=active]:bg-slate-700">
                  <BarChart3 className="w-4 h-4 mr-2" />Metrics
                </TabsTrigger>
                <TabsTrigger value="alerts" className="data-[state=active]:bg-slate-700">
                  <Bell className="w-4 h-4 mr-2" />Alerts
                  {health && health.alerts.length > 0 && (
                    <Badge variant="destructive" className="ml-2">
                      {health.alerts.length}
                    </Badge>
                  )}
                </TabsTrigger>
                <TabsTrigger value="logs" className="data-[state=active]:bg-slate-700">
                  <FileText className="w-4 h-4 mr-2" />Logs
                </TabsTrigger>
              </TabsList>

              {/* Metrics Tab */}
              <TabsContent value="metrics">
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  {/* Confidence Timeline */}
                  <Card className="bg-slate-800/50 border-slate-700">
                    <CardHeader>
                      <CardTitle className="text-white">Confidence Timeline</CardTitle>
                      <CardDescription className="text-slate-400">
                        Prediction confidence over recent predictions
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      <ResponsiveContainer width="100%" height={300}>
                        <AreaChart data={buildConfidenceTimeline()}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                          <XAxis dataKey="time" stroke="#9ca3af" fontSize={12} />
                          <YAxis stroke="#9ca3af" domain={[0, 100]} />
                          <Tooltip
                            contentStyle={{
                              backgroundColor: '#1e293b',
                              border: '1px solid #475569',
                              borderRadius: '8px',
                            }}
                          />
                          <Area
                            type="monotone"
                            dataKey="confidence"
                            stroke="#3b82f6"
                            fill="#3b82f680"
                            name="Confidence %"
                          />
                        </AreaChart>
                      </ResponsiveContainer>
                    </CardContent>
                  </Card>

                  {/* Inference Time Timeline */}
                  <Card className="bg-slate-800/50 border-slate-700">
                    <CardHeader>
                      <CardTitle className="text-white">Inference Time</CardTitle>
                      <CardDescription className="text-slate-400">
                        Model inference latency (ms)
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      <ResponsiveContainer width="100%" height={300}>
                        <LineChart data={buildInferenceTimeline()}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                          <XAxis dataKey="time" stroke="#9ca3af" fontSize={12} />
                          <YAxis stroke="#9ca3af" />
                          <Tooltip
                            contentStyle={{
                              backgroundColor: '#1e293b',
                              border: '1px solid #475569',
                              borderRadius: '8px',
                            }}
                          />
                          <Line
                            type="monotone"
                            dataKey="inference_ms"
                            stroke="#10b981"
                            strokeWidth={2}
                            dot={false}
                            name="Inference (ms)"
                          />
                        </LineChart>
                      </ResponsiveContainer>
                    </CardContent>
                  </Card>

                  {/* Statistics */}
                  <Card className="bg-slate-800/50 border-slate-700 lg:col-span-2">
                    <CardHeader>
                      <CardTitle className="text-white">Confidence Statistics</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                        <div className="text-center p-4 bg-slate-700/50 rounded-lg">
                          <div className="text-2xl font-bold text-white">
                            {metrics?.min_confidence != null
                              ? `${(metrics.min_confidence * 100).toFixed(1)}%`
                              : 'N/A'}
                          </div>
                          <div className="text-sm text-slate-400">Min</div>
                        </div>
                        <div className="text-center p-4 bg-slate-700/50 rounded-lg">
                          <div className="text-2xl font-bold text-white">
                            {metrics?.max_confidence != null
                              ? `${(metrics.max_confidence * 100).toFixed(1)}%`
                              : 'N/A'}
                          </div>
                          <div className="text-sm text-slate-400">Max</div>
                        </div>
                        <div className="text-center p-4 bg-slate-700/50 rounded-lg">
                          <div className="text-2xl font-bold text-white">
                            {metrics?.mean_confidence != null
                              ? `${(metrics.mean_confidence * 100).toFixed(1)}%`
                              : 'N/A'}
                          </div>
                          <div className="text-sm text-slate-400">Mean</div>
                        </div>
                        <div className="text-center p-4 bg-slate-700/50 rounded-lg">
                          <div className="text-2xl font-bold text-white">
                            {metrics?.std_confidence != null
                              ? `${(metrics.std_confidence * 100).toFixed(1)}%`
                              : 'N/A'}
                          </div>
                          <div className="text-sm text-slate-400">Std Dev</div>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </div>
              </TabsContent>

              {/* Alerts Tab */}
              <TabsContent value="alerts">
                <Card className="bg-slate-800/50 border-slate-700">
                  <CardHeader>
                    <CardTitle className="text-white">Active Alerts</CardTitle>
                    <CardDescription className="text-slate-400">
                      Model health and performance issues
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    {health?.alerts && health.alerts.length > 0 ? (
                      <div className="space-y-4">
                        {health.alerts.map((alert, i) => (
                          <div
                            key={i}
                            className="flex items-start gap-4 p-4 bg-slate-700/50 rounded-lg"
                          >
                            <div className={`p-2 rounded-full ${getSeverityColor(alert.severity)}`}>
                              <AlertTriangle className="w-4 h-4 text-white" />
                            </div>
                            <div className="flex-1">
                              <div className="flex items-center gap-2">
                                <Badge variant={alert.severity === 'CRITICAL' ? 'destructive' : 'secondary'}>
                                  {alert.severity}
                                </Badge>
                                <span className="text-sm text-slate-400">{alert.type}</span>
                              </div>
                              <p className="text-white mt-1">{alert.message}</p>
                              <p className="text-sm text-slate-500 mt-1">
                                {new Date(alert.timestamp).toLocaleString()}
                              </p>
                            </div>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <div className="text-center py-12">
                        <CheckCircle className="w-16 h-16 text-green-500 mx-auto mb-4" />
                        <p className="text-xl text-white">No Active Alerts</p>
                        <p className="text-slate-400 mt-2">All systems operating normally</p>
                      </div>
                    )}
                  </CardContent>
                </Card>

                {/* Issues */}
                {health?.issues && health.issues.length > 0 && (
                  <Card className="bg-slate-800/50 border-slate-700 mt-6">
                    <CardHeader>
                      <CardTitle className="text-white">Detected Issues</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <ul className="space-y-2">
                        {health.issues.map((issue, i) => (
                          <li key={i} className="flex items-center gap-3 text-yellow-400">
                            <AlertCircle className="w-5 h-5" />
                            {issue}
                          </li>
                        ))}
                      </ul>
                    </CardContent>
                  </Card>
                )}
              </TabsContent>

              {/* Logs Tab */}
              <TabsContent value="logs">
                <Card className="bg-slate-800/50 border-slate-700">
                  <CardHeader>
                    <CardTitle className="text-white">Recent Prediction Logs</CardTitle>
                    <CardDescription className="text-slate-400">
                      Last {logs.length} predictions
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="overflow-x-auto">
                      <table className="w-full">
                        <thead>
                          <tr className="border-b border-slate-700">
                            <th className="text-left py-3 px-4 text-slate-400 font-medium">Time</th>
                            <th className="text-left py-3 px-4 text-slate-400 font-medium">Model</th>
                            <th className="text-left py-3 px-4 text-slate-400 font-medium">Class</th>
                            <th className="text-left py-3 px-4 text-slate-400 font-medium">Confidence</th>
                            <th className="text-left py-3 px-4 text-slate-400 font-medium">Latency</th>
                          </tr>
                        </thead>
                        <tbody>
                          {logs.length > 0 ? (
                            logs.map((log, i) => (
                              <tr key={i} className="border-b border-slate-700/50 hover:bg-slate-700/30">
                                <td className="py-3 px-4 text-slate-300">
                                  {new Date(log.timestamp).toLocaleString()}
                                </td>
                                <td className="py-3 px-4">
                                  <Badge variant="outline" className="text-blue-400 border-blue-400">
                                    {log.model_name}
                                  </Badge>
                                </td>
                                <td className="py-3 px-4">
                                  <Badge
                                    className={
                                      log.predicted_class === 'genuine'
                                        ? 'bg-green-500'
                                        : 'bg-red-500'
                                    }
                                  >
                                    {log.predicted_class}
                                  </Badge>
                                </td>
                                <td className="py-3 px-4 text-white">
                                  {(log.confidence * 100).toFixed(1)}%
                                </td>
                                <td className="py-3 px-4 text-slate-300">
                                  {log.inference_time_ms.toFixed(1)} ms
                                </td>
                              </tr>
                            ))
                          ) : (
                            <tr>
                              <td colSpan={5} className="py-8 text-center text-slate-500">
                                No prediction logs available
                              </td>
                            </tr>
                          )}
                        </tbody>
                      </table>
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>
            </Tabs>
          </>
        )}
      </main>
    </div>
  );
}
