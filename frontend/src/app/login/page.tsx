'use client';

import { useState, FormEvent, useEffect, useRef } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import Link from 'next/link';
import { Radar, Mail, Lock, Loader2, ArrowRight } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { useAuthStore } from '@/lib/store';
import { useToast } from '@/hooks/use-toast';
import { useRedirectIfAuthenticated } from '@/hooks/use-redirect-if-authenticated';
import { mapAuthErrorMessage, normalizeIdentifier, validateLoginInput } from '@/lib/auth-contract';

export default function LoginPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { login, isLoading } = useAuthStore();
  const { toast } = useToast();
  useRedirectIfAuthenticated('/dashboard');

  const [identifier, setIdentifier] = useState('');
  const [password, setPassword] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const identifierRef = useRef<HTMLInputElement>(null);
  const passwordRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const queryEmail = searchParams.get('email');
    if (queryEmail) setIdentifier(queryEmail);
  }, [searchParams]);

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (isSubmitting || isLoading) return;

    const validation = validateLoginInput(identifier, password);
    if (!validation.isValid) {
      if (validation.field === 'identifier') identifierRef.current?.focus();
      if (validation.field === 'password') passwordRef.current?.focus();
      toast({
        title: 'Invalid input',
        description: validation.message || 'Please check your input and try again.',
        variant: 'destructive',
      });
      return;
    }

    setIsSubmitting(true);
    const result = await login(normalizeIdentifier(identifier), password);
    setIsSubmitting(false);

    if (result.success) {
      toast({
        title: 'Login successful',
        description: 'You are now authenticated.',
      });
      router.push('/dashboard');
    } else {
      toast({
        title: 'Login failed',
        description: mapAuthErrorMessage(result.error),
        variant: 'destructive',
      });
    }
  };

  return (
    <div className="min-h-screen bg-[#0B1220] flex items-center justify-center p-4 relative overflow-hidden">
      <div className="pointer-events-none absolute -top-40 -left-40 h-[28rem] w-[28rem] rounded-full bg-cyan-500/15 blur-3xl" />
      <div className="pointer-events-none absolute -bottom-40 -right-40 h-[30rem] w-[30rem] rounded-full bg-emerald-500/10 blur-3xl" />

      <Card className="w-full max-w-md bg-[#0f172a]/90 backdrop-blur-xl border-slate-800 shadow-2xl relative z-10">
        <CardHeader className="space-y-3">
          <div className="flex justify-center mb-2">
            <div className="h-12 w-12 rounded-xl bg-gradient-to-br from-cyan-400 to-emerald-400 flex items-center justify-center shadow-[0_0_24px_rgba(52,211,153,0.35)]">
              <Radar className="h-6 w-6 text-slate-950" />
            </div>
          </div>
          <CardTitle className="text-2xl font-bold text-center text-white">Authentication Login</CardTitle>
          <CardDescription className="text-center text-slate-400">
            Sign in with your email and password
          </CardDescription>
        </CardHeader>
        <form onSubmit={handleSubmit}>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="email">Email</Label>
              <div className="relative">
                <Mail className="absolute left-3 top-3 h-4 w-4 text-slate-500" />
                <Input
                  ref={identifierRef}
                  id="email"
                  type="text"
                  placeholder="name@example.com"
                  className="pl-9 bg-slate-900/50 border-slate-800 text-slate-200 placeholder:text-slate-500 focus-visible:ring-emerald-500"
                  value={identifier}
                  onChange={(e) => setIdentifier(e.target.value)}
                  onBlur={(e) => setIdentifier(normalizeIdentifier(e.target.value).toLowerCase())}
                  disabled={isLoading || isSubmitting}
                  autoComplete="email"
                />
              </div>
            </div>
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label htmlFor="password">Password</Label>
                <Link href="/forgot-password" className="text-xs text-emerald-400 hover:text-emerald-300 transition-colors">
                  Forgot password?
                </Link>
              </div>
              <div className="relative">
                <Lock className="absolute left-3 top-3 h-4 w-4 text-slate-500" />
                <Input
                  ref={passwordRef}
                  id="password"
                  type="password"
                  className="pl-9 bg-slate-900/50 border-slate-800 text-slate-200 placeholder:text-slate-500 focus-visible:ring-emerald-500"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  disabled={isLoading || isSubmitting}
                  autoComplete="current-password"
                />
              </div>
            </div>
          </CardContent>
          <CardFooter className="flex flex-col space-y-4">
            <Button
              type="submit"
              className="w-full bg-emerald-500 hover:bg-emerald-600 text-white font-medium"
              disabled={isLoading || isSubmitting}
            >
              {isLoading || isSubmitting ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Signing in...
                </>
              ) : (
                <>
                  Sign in
                  <ArrowRight className="ml-2 h-4 w-4" />
                </>
              )}
            </Button>
            <div className="text-center text-sm text-slate-400">
              Don&apos;t have an account?{' '}
              <Link href="/register" className="text-emerald-400 hover:text-emerald-300 font-medium transition-colors">
                Sign up
              </Link>
            </div>
            <div className="text-center text-xs text-slate-500 mt-4">
              Use your registered email and password.
            </div>
          </CardFooter>
        </form>
      </Card>
    </div>
  );
}
