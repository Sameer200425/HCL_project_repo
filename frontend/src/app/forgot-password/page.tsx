'use client';

import { FormEvent, useRef, useState } from 'react';
import Link from 'next/link';
import { Mail, Loader2, ArrowRight, Radar } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { useToast } from '@/hooks/use-toast';
import { api } from '@/lib/api';
import { mapAuthErrorMessage, normalizeEmail } from '@/lib/auth-contract';

export default function ForgotPasswordPage() {
  const { toast } = useToast();
  const [email, setEmail] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const emailRef = useRef<HTMLInputElement>(null);

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (isSubmitting) return;

    const normalizedEmail = normalizeEmail(email);
    if (!normalizedEmail) {
      emailRef.current?.focus();
      toast({
        title: 'Email required',
        description: 'Enter your email to reset your password.',
        variant: 'destructive',
      });
      return;
    }

    setIsSubmitting(true);
    const result = await api.forgotPassword(normalizedEmail);
    setIsSubmitting(false);

    if (result.error) {
      toast({
        title: 'Request failed',
        description: mapAuthErrorMessage(result.error),
        variant: 'destructive',
      });
      return;
    }

    const resetToken = result.data?.reset_token;
    toast({
      title: 'Reset link sent',
      description: 'If this email exists, a reset link has been issued.',
    });

    if (resetToken) {
      window.location.href = `/reset-password?token=${encodeURIComponent(resetToken)}`;
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
          <CardTitle className="text-2xl font-bold text-center text-white">Reset your password</CardTitle>
          <CardDescription className="text-center text-slate-400">
            Enter your account email and we will send reset instructions.
          </CardDescription>
        </CardHeader>
        <form onSubmit={handleSubmit}>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="email">Email</Label>
              <div className="relative">
                <Mail className="absolute left-3 top-3 h-4 w-4 text-slate-500" />
                <Input
                  ref={emailRef}
                  id="email"
                  type="email"
                  placeholder="name@example.com"
                  className="pl-9 bg-slate-900/50 border-slate-800 text-slate-200 placeholder:text-slate-500 focus-visible:ring-emerald-500"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  disabled={isSubmitting}
                  autoComplete="email"
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
                  Sending link...
                </>
              ) : (
                <>
                  Send reset link
                  <ArrowRight className="ml-2 h-4 w-4" />
                </>
              )}
            </Button>
            <div className="text-center text-sm text-slate-400">
              Back to{' '}
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
