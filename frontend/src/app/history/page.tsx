'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { 
  Shield, Upload, History, LogOut, User, Trash2,
  AlertTriangle, CheckCircle, ChevronLeft, ChevronRight,
  FileSearch
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { useAuthStore, usePredictionStore, type Prediction } from '@/lib/store';
import { useToast } from '@/hooks/use-toast';
import { api } from '@/lib/api';

export default function HistoryPage() {
  const router = useRouter();
  const { user, token, logout, authDisabled } = useAuthStore();
  const { predictions, fetchHistory, isLoading } = usePredictionStore();
  const { toast } = useToast();
  
  const [page, setPage] = useState(1);
  const [riskFilter, setRiskFilter] = useState<string | null>(null);
  const itemsPerPage = 10;

  useEffect(() => {
    if (!token && !authDisabled) {
      router.push('/login');
      return;
    }
    
    fetchHistory();
  }, [token, authDisabled, router, fetchHistory]);

  const handleLogout = () => {
    logout();
    router.push('/');
  };

  const handleDelete = async (id: number) => {
    if (!confirm('Are you sure you want to delete this prediction?')) return;
    
    try {
      await api.deletePrediction(id);
      await fetchHistory();
      toast({
        title: "Deleted",
        description: "Prediction has been removed.",
      });
    } catch (error) {
      toast({
        title: "Delete failed",
        description: "Could not delete the prediction.",
        variant: "destructive",
      });
    }
  };

  const filteredPredictions = riskFilter
    ? predictions.filter(p => p.risk_level === riskFilter)
    : predictions;

  const totalPages = Math.ceil(filteredPredictions.length / itemsPerPage);
  const paginatedPredictions = filteredPredictions.slice(
    (page - 1) * itemsPerPage,
    page * itemsPerPage
  );

  const riskFilters = ['all', 'low', 'medium', 'high', 'critical'];
  const dashboardCardClass = 'bg-[#0f172a]/90 border-slate-800 text-slate-100 shadow-xl';
  const mutedTextClass = 'text-slate-400';

  const formatDate = (dateStr: string) => {
    return new Date(dateStr).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

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
              <Button variant="ghost" size="sm" onClick={handleLogout} className="text-slate-300 hover:text-white hover:bg-slate-800">
                <LogOut className="h-4 w-4" />
              </Button>
            </div>
          </nav>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 py-8 relative z-10">
        <div className="flex items-center justify-between mb-8">
          <h1 className="text-3xl font-bold text-white">Prediction History</h1>
          <p className="text-slate-400">
            {filteredPredictions.length} total predictions
          </p>
        </div>

        {/* Filters */}
        <Card className={`mb-6 ${dashboardCardClass}`}>
          <CardContent className="pt-4">
            <div className="flex items-center gap-4">
              <span className="text-sm font-medium text-slate-200">Filter by risk:</span>
              <div className="flex gap-2">
                {riskFilters.map((risk) => (
                  <Button
                    key={risk}
                    variant={
                      (risk === 'all' && !riskFilter) || riskFilter === risk 
                        ? 'default' 
                        : 'outline'
                    }
                    size="sm"
                    onClick={() => {
                      setRiskFilter(risk === 'all' ? null : risk);
                      setPage(1);
                    }}
                    className="capitalize"
                  >
                    {risk}
                  </Button>
                ))}
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Predictions List */}
        <Card className={dashboardCardClass}>
          <CardHeader>
            <CardTitle>Scan Results</CardTitle>
            <CardDescription className={mutedTextClass}>
              Your document analysis history
            </CardDescription>
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <div className="flex items-center justify-center py-12">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-emerald-500"></div>
              </div>
            ) : paginatedPredictions.length === 0 ? (
              <div className="text-center py-12 text-slate-400">
                <FileSearch className="h-12 w-12 mx-auto mb-4 opacity-50" />
                <p>No predictions found.</p>
                <Link href="/predict">
                  <Button className="mt-4 bg-emerald-500 hover:bg-emerald-600">
                    <Upload className="h-4 w-4 mr-2" /> Analyze Document
                  </Button>
                </Link>
              </div>
            ) : (
              <>
                <div className="space-y-3">
                  {paginatedPredictions.map((pred) => (
                    <div 
                      key={pred.id} 
                      className="flex items-center justify-between p-4 bg-slate-900/60 border border-slate-800 rounded-lg hover:bg-slate-900 transition-colors"
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
                          <p className="text-sm text-slate-400">
                            {pred.model_name} • {formatDate(pred.created_at)}
                          </p>
                        </div>
                      </div>
                      <div className="flex items-center gap-4">
                        <div className="text-right">
                          <p className="font-medium">{(pred.confidence * 100).toFixed(1)}%</p>
                          <p className="text-sm text-slate-400 capitalize">
                            {pred.risk_level} risk
                          </p>
                        </div>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => handleDelete(pred.id)}
                          className="text-red-400 hover:text-red-300 hover:bg-red-900/30"
                        >
                          <Trash2 className="h-4 w-4" />
                        </Button>
                      </div>
                    </div>
                  ))}
                </div>

                {/* Pagination */}
                {totalPages > 1 && (
                  <div className="flex items-center justify-center gap-2 mt-6">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setPage(p => Math.max(1, p - 1))}
                      disabled={page === 1}
                      className="border-slate-700 text-slate-200 hover:bg-slate-800"
                    >
                      <ChevronLeft className="h-4 w-4" />
                    </Button>
                    <span className="text-sm text-slate-300">
                      Page {page} of {totalPages}
                    </span>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setPage(p => Math.min(totalPages, p + 1))}
                      disabled={page === totalPages}
                      className="border-slate-700 text-slate-200 hover:bg-slate-800"
                    >
                      <ChevronRight className="h-4 w-4" />
                    </Button>
                  </div>
                )}
              </>
            )}
          </CardContent>
        </Card>
      </main>
    </div>
  );
}
