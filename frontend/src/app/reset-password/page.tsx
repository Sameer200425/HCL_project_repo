'use client';

import { FormEvent, useMemo, useRef, useState } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import Link from 'next/link';
import { Loader2, ArrowRight, Lock, Radar } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { useToast } from '@/hooks/use-toast';
import { api } from '@/lib/api';
import { mapAuthErrorMessage, validateRegisterInput } from '@/lib/auth-contract';

export default function ResetPasswordPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { toast } = useToast();

  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);

  const passwordRef = useRef<HTMLInputElement>(null);
  const confirmPasswordRef = useRef<HTMLInputElement>(null);

  const token = useMemo(() => searchParams.get('token') || '', [searchParams]);

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (isSubmitting) return;

    if (!token) {
      toast({
        title: 'Invalid reset link',
        description: 'The reset link is missing a token. Request a new link.',
        variant: 'destructive',
      });
      return;
    }

    const validation = validateRegisterInput({
      email: 'reset@example.com',
      username: 'reset_user',
      password,
      confirmPassword,
    });

    if (!validation.isValid) {
      if (validation.field === 'password') passwordRef.current?.focus();
      if (validation.field === 'confirmPassword') confirmPasswordRef.current?.focus();
      toast({
        title: 'Invalid password',
        description: validation.message || 'Please check your password and try again.',
        variant: 'destructive',
      });
      return;
    }

    setIsSubmitting(true);
    const result = await api.resetPassword(token, password);
    setIsSubmitting(false);

    if (result.error) {
      toast({
        title: 'Reset failed',
        description: mapAuthErrorMessage(result.error),
        variant: 'destructive',
      });
      return;
    }

    toast({
      title: 'Password reset complete',
      description: 'Your password has been updated. Please sign in.',
    });
    router.push('/login');
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
          <CardTitle className="text-2xl font-bold text-center text-white">Set a new password</CardTitle>
          <CardDescription className="text-center text-slate-400">
            Use a strong password that is unique to this account.
          </CardDescription>
        </CardHeader>
        <form onSubmit={handleSubmit}>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="password">New password</Label>
              <div className="relative">
                <Lock className="absolute left-3 top-3 h-4 w-4 text-slate-500" />
                <Input
                  ref={passwordRef}
                  id="password"
                  type="password"
                  className="pl-9 bg-slate-900/50 border-slate-800 text-slate-200 placeholder:text-slate-500 focus-visible:ring-emerald-500"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  disabled={isSubmitting}
                  autoComplete="new-password"
                />
              </div>
            </div>
            <div className="space-y-2">
              <Label htmlFor="confirmPassword">Confirm password</Label>
              <div className="relative">
                <Lock className="absolute left-3 top-3 h-4 w-4 text-slate-500" />
                <Input
                  ref={confirmPasswordRef}
                  id="confirmPassword"
                  type="password"
                  className="pl-9 bg-slate-900/50 border-slate-800 text-slate-200 placeholder:text-slate-500 focus-visible:ring-emerald-500"
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  disabled={isSubmitting}
                  autoComplete="new-password"
                />
              </div>
            </div>
          </CardContent>
          <CardFooter className="flex flex-col space-y-4">
            <Button
              type="submit"
              className="w-full bg-emerald-500 hover:bg-emerald-600 text-white font-medium"
              disabled={isSubmitting}
            >
              {isSubmitting ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Resetting password...
                </>
              ) : (
                <>
                  Update password
                  <ArrowRight className="ml-2 h-4 w-4" />
                </>
              )}
            </Button>
            <div className="text-center text-sm text-slate-400">
              Return to{' '}
              <Link href="/login" className="text-emerald-400 hover:text-emerald-300 font-medium transition-colors">
                Sign in
              </Link>
            </div>
          </CardFooter>
        </form>
      </Card>
    </div>
  );
}
