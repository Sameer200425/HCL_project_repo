'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import {
  Shield, BarChart3, Upload, LogOut, User as UserIcon,
  TrendingUp, Activity, Settings as SettingsIcon,
  Moon, Sun, Bell, Lock, Globe, Palette, Save, RefreshCw,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { useAuthStore } from '@/lib/store';
import { useToast } from '@/hooks/use-toast';

/* ------------------------------------------------------------------ */
/* Simple Toggle                                                       */
/* ------------------------------------------------------------------ */

function Toggle({ checked, onChange, label }: { checked: boolean; onChange: (v: boolean) => void; label: string }) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-sm">{label}</span>
      <button
        type="button"
        onClick={() => onChange(!checked)}
        className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${checked ? 'bg-blue-600' : 'bg-gray-300'}`}
      >
        <span className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${checked ? 'translate-x-6' : 'translate-x-1'}`} />
      </button>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Page                                                                */
/* ------------------------------------------------------------------ */

export default function SettingsPage() {
  const router = useRouter();
  const { user, token, logout, loadUser, authDisabled } = useAuthStore();
  const { toast } = useToast();
  const [isLoading, setIsLoading] = useState(true);

  // Profile
  const [fullName, setFullName] = useState('');
  const [email, setEmail] = useState('');

  // Preferences
  const [darkMode, setDarkMode] = useState(false);
  const [notifications, setNotifications] = useState(true);
  const [autoAnalyze, setAutoAnalyze] = useState(false);
  const [defaultModel, setDefaultModel] = useState('cnn');
  const [confidenceThreshold, setConfidenceThreshold] = useState('0.7');

  // API
  const [apiUrl, setApiUrl] = useState('http://localhost:8001');

  useEffect(() => {
    const init = async () => {
      if (!token && !authDisabled) { router.push('/login'); return; }
      await loadUser();
      setIsLoading(false);
    };
    init();
  }, [token, authDisabled, router, loadUser]);

  useEffect(() => {
    if (user) {
      setFullName(user.full_name || '');
      setEmail(user.email || '');
    }
  }, [user]);

  const handleLogout = () => {
    logout();
    toast({ title: 'Logged out', description: 'Session ended.' });
    router.push('/');
  };

  const handleSaveProfile = () => {
    toast({ title: 'Profile saved', description: 'Your profile has been updated.' });
  };

  const handleSavePreferences = () => {
    if (typeof window !== 'undefined') {
      localStorage.setItem('fg_preferences', JSON.stringify({
        darkMode, notifications, autoAnalyze, defaultModel, confidenceThreshold,
      }));
    }
    toast({ title: 'Preferences saved', description: 'Your preferences have been updated.' });
  };

  const handleSaveApi = () => {
    if (typeof window !== 'undefined') {
      localStorage.setItem('fg_api_url', apiUrl);
    }
    toast({ title: 'API settings saved', description: `Backend URL set to ${apiUrl}` });
  };

  const handleClearHistory = () => {
    if (typeof window !== 'undefined') {
      localStorage.removeItem('prediction-storage');
    }
    toast({ title: 'History cleared', description: 'All local prediction history has been removed.' });
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
              <UserIcon className="h-4 w-4 text-muted-foreground" />
              <span className="text-sm">{user?.username || 'User'}</span>
              <Button variant="ghost" size="sm" onClick={handleLogout}><LogOut className="h-4 w-4" /></Button>
            </div>
          </nav>
        </div>
      </header>

      {/* Main */}
      <main className="max-w-4xl mx-auto px-4 py-8">
        <div className="mb-8">
          <h1 className="text-3xl font-bold">Settings</h1>
          <p className="text-muted-foreground">Manage your account, preferences, and application configuration</p>
        </div>

        <Tabs defaultValue="profile" className="space-y-6">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="profile">Profile</TabsTrigger>
            <TabsTrigger value="preferences">Preferences</TabsTrigger>
            <TabsTrigger value="api">API &amp; Backend</TabsTrigger>
            <TabsTrigger value="danger">Data &amp; Privacy</TabsTrigger>
          </TabsList>

          {/* Profile Tab */}
          <TabsContent value="profile">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2"><UserIcon className="h-5 w-5" /> Account Profile</CardTitle>
                <CardDescription>Your personal information and account details</CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="grid md:grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label>Username</Label>
                    <Input value={user?.username || ''} disabled className="bg-slate-50" />
                    <p className="text-xs text-muted-foreground">Username cannot be changed</p>
                  </div>
                  <div className="space-y-2">
                    <Label>Role</Label>
                    <Input value={user?.role || 'user'} disabled className="bg-slate-50" />
                  </div>
                </div>
                <div className="space-y-2">
                  <Label>Full Name</Label>
                  <Input value={fullName} onChange={e => setFullName(e.target.value)} placeholder="Your full name" />
                </div>
                <div className="space-y-2">
                  <Label>Email</Label>
                  <Input value={email} onChange={e => setEmail(e.target.value)} type="email" placeholder="you@example.com" />
                </div>
                <Button onClick={handleSaveProfile} className="gap-2"><Save className="h-4 w-4" /> Save Profile</Button>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Preferences Tab */}
          <TabsContent value="preferences">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2"><Palette className="h-5 w-5" /> Application Preferences</CardTitle>
                <CardDescription>Customize the application behavior</CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="space-y-4">
                  <h3 className="font-semibold text-sm uppercase text-muted-foreground">Appearance</h3>
                  <Toggle checked={darkMode} onChange={setDarkMode} label="Dark Mode" />
                </div>

                <div className="space-y-4 border-t pt-4">
                  <h3 className="font-semibold text-sm uppercase text-muted-foreground">Notifications</h3>
                  <Toggle checked={notifications} onChange={setNotifications} label="Enable desktop notifications" />
                </div>

                <div className="space-y-4 border-t pt-4">
                  <h3 className="font-semibold text-sm uppercase text-muted-foreground">Analysis</h3>
                  <Toggle checked={autoAnalyze} onChange={setAutoAnalyze} label="Auto-analyze on upload" />

                  <div className="space-y-2">
                    <Label>Default Model</Label>
                    <select
                      value={defaultModel}
                      onChange={e => setDefaultModel(e.target.value)}
                      className="w-full border rounded-md px-3 py-2 text-sm bg-background"
                    >
                      <option value="cnn">CNN (ResNet50)</option>
                      <option value="vit">ViT (from scratch)</option>
                      <option value="hybrid">Hybrid CNN+ViT</option>
                      <option value="vit_ssl">ViT + SSL (MAE)</option>
                    </select>
                  </div>

                  <div className="space-y-2">
                    <Label>Confidence Threshold</Label>
                    <Input
                      type="number"
                      min="0" max="1" step="0.05"
                      value={confidenceThreshold}
                      onChange={e => setConfidenceThreshold(e.target.value)}
                    />
                    <p className="text-xs text-muted-foreground">Predictions below this threshold will be flagged for review</p>
                  </div>
                </div>

                <Button onClick={handleSavePreferences} className="gap-2"><Save className="h-4 w-4" /> Save Preferences</Button>
              </CardContent>
            </Card>
          </TabsContent>

          {/* API Tab */}
          <TabsContent value="api">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2"><Globe className="h-5 w-5" /> API Configuration</CardTitle>
                <CardDescription>Backend server and connectivity settings</CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="space-y-2">
                  <Label>Backend API URL</Label>
                  <Input value={apiUrl} onChange={e => setApiUrl(e.target.value)} placeholder="http://localhost:8001" />
                  <p className="text-xs text-muted-foreground">The FastAPI backend server address</p>
                </div>

                <div className="p-4 bg-slate-50 rounded-lg space-y-2">
                  <h4 className="font-semibold text-sm">Connection Info</h4>
                  <div className="grid grid-cols-2 gap-2 text-sm">
                    <span className="text-muted-foreground">Status:</span>
                    <span className="text-green-600 font-medium">Connected</span>
                    <span className="text-muted-foreground">Auth Token:</span>
                    <span className="font-mono text-xs">{token ? `${token.slice(0, 20)}...` : 'None'}</span>
                    <span className="text-muted-foreground">User Role:</span>
                    <span>{user?.role || 'N/A'}</span>
                  </div>
                </div>

                <Button onClick={handleSaveApi} className="gap-2"><Save className="h-4 w-4" /> Save API Settings</Button>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Danger Zone */}
          <TabsContent value="danger">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2"><Lock className="h-5 w-5" /> Data &amp; Privacy</CardTitle>
                <CardDescription>Manage your data, clear history, and privacy settings</CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="p-4 border rounded-lg space-y-3">
                  <h4 className="font-semibold">Local Data</h4>
                  <p className="text-sm text-muted-foreground">
                    Clear all locally stored prediction history and cached data. This does not affect server-side data.
                  </p>
                  <Button variant="outline" onClick={handleClearHistory} className="gap-2">
                    <RefreshCw className="h-4 w-4" /> Clear Local History
                  </Button>
                </div>

                <div className="p-4 border border-red-200 rounded-lg space-y-3 bg-red-50">
                  <h4 className="font-semibold text-red-600">Danger Zone</h4>
                  <p className="text-sm text-muted-foreground">
                    Log out and revoke your session token. You will need to log in again.
                  </p>
                  <Button variant="destructive" onClick={handleLogout} className="gap-2">
                    <LogOut className="h-4 w-4" /> Log Out &amp; Revoke Token
                  </Button>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </main>
    </div>
  );
}
