import { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { useAuthStore } from '@/lib/store';

export function useRedirectIfAuthenticated(target: string = '/dashboard') {
  const router = useRouter();
  const { authDisabled, isAuthenticated, loadUser } = useAuthStore();

  useEffect(() => {
    let alive = true;

    const run = async () => {
      if (authDisabled) return;
      await loadUser();
      if (!alive) return;
      if (useAuthStore.getState().isAuthenticated) {
        router.replace(target);
      }
    };

    run();
    return () => {
      alive = false;
    };
  }, [authDisabled, isAuthenticated, loadUser, router, target]);
}