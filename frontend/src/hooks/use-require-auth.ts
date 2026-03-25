import { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { useAuthStore } from '@/lib/store';

/**
 * useRequireAuth - React hook to protect pages/routes.
 * Redirects to /login if not authenticated and auth is enabled.
 * Usage: Call at the top of any protected page component.
 */
export function useRequireAuth() {
  const router = useRouter();
  const { token, isAuthenticated, authDisabled, loadUser } = useAuthStore();

  useEffect(() => {
    let alive = true;
    if (authDisabled) return;

    const verify = async () => {
      await loadUser();
      if (!alive) return;
      const state = useAuthStore.getState();
      if (!state.token || !state.isAuthenticated) {
        router.replace('/login');
      }
    };

    if (!token || !isAuthenticated) {
      router.replace('/login');
      return;
    }

    verify();
    return () => {
      alive = false;
    };
  }, [token, isAuthenticated, authDisabled, router, loadUser]);
}
