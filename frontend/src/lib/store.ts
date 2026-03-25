/**
 * Global State Management with Zustand
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { api } from './api';

const AUTH_DISABLED = process.env.NEXT_PUBLIC_AUTH_DISABLED !== 'false';
const PUBLIC_TOKEN = 'public-access';
const COOKIE_SESSION_HINT = 'cookie-session';

const PUBLIC_USER = {
  id: -1,
  email: 'public@example.local',
  username: 'public_user',
  full_name: 'Public Analyst',
  role: 'admin',
} as const;

interface User {
  id: number;
  email: string;
  username: string;
  full_name: string | null;
  role: string;
}

interface Prediction {
  id: number;
  filename: string;
  model_name: string;
  predicted_class: string;
  confidence: number;
  probabilities: Record<string, number>;
  risk_level: string;
  inference_time_ms: number;
  explanation?: string;
  created_at: string;
}

interface AuthState {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  authDisabled: boolean;
  
  login: (identifier: string, password: string) => Promise<{ success: boolean; error?: string }>;
  register: (email: string, username: string, password: string, fullName?: string) => Promise<{ success: boolean; error?: string }>;
  logout: () => void;
  loadUser: () => Promise<void>;
}

interface PredictionState {
  predictions: Prediction[];
  currentPrediction: Prediction | null;
  isLoading: boolean;
  
  setPredictions: (predictions: Prediction[]) => void;
  setCurrentPrediction: (prediction: Prediction | null) => void;
  addPrediction: (prediction: Prediction) => void;
  clearPredictions: () => void;
  fetchHistory: () => Promise<{ success: boolean; error?: string }>;
  fetchStats: () => Promise<{ success: boolean; data?: unknown; error?: string }>;
  predict: (file: File, model: string) => Promise<{ success: boolean; data?: unknown; error?: string }>;
}

export type { Prediction };

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      user: AUTH_DISABLED ? { ...PUBLIC_USER } : null,
      token: AUTH_DISABLED ? PUBLIC_TOKEN : null,
      isAuthenticated: AUTH_DISABLED,
      isLoading: false,
      authDisabled: AUTH_DISABLED,

      login: async (identifier: string, password: string) => {
        if (AUTH_DISABLED) {
          set({ user: { ...PUBLIC_USER }, token: PUBLIC_TOKEN, isAuthenticated: true });
          return { success: true };
        }
        set({ isLoading: true });
        try {
          const result = await api.login(identifier, password);

          if (result.error) {
            set({ isLoading: false });
            return { success: false, error: result.error };
          }

          const token = result.data?.access_token || COOKIE_SESSION_HINT;
          api.setToken(token);
          set({ token });

          // Load user info and confirm auth state before reporting success
          await get().loadUser();
          const { isAuthenticated } = get();
          if (!isAuthenticated) {
            api.setToken(null);
            set({ token: null, user: null, isAuthenticated: false, isLoading: false });
            return { success: false, error: 'Session validation failed. Please try again.' };
          }

          set({ isLoading: false });
          return { success: true };
        } catch {
          api.setToken(null);
          set({ token: null, user: null, isAuthenticated: false, isLoading: false });
          return { success: false, error: 'Unable to sign in right now. Please try again.' };
        }
      },

      register: async (email: string, username: string, password: string, fullName?: string) => {
        if (AUTH_DISABLED) {
          return { success: true };
        }
        set({ isLoading: true });
        try {
          const result = await api.register(email, username, password, fullName);

          if (result.error) {
            set({ isLoading: false });
            return { success: false, error: result.error };
          }

          set({ isLoading: false });
          return { success: true };
        } catch {
          set({ isLoading: false });
          return { success: false, error: 'Unable to register right now. Please try again.' };
        }
      },

      logout: () => {
        if (AUTH_DISABLED) {
          set({ user: { ...PUBLIC_USER }, token: PUBLIC_TOKEN, isAuthenticated: true });
          return;
        }
        void api.logout();
        api.setToken(null);
        set({ user: null, token: null, isAuthenticated: false });
      },

      loadUser: async () => {
        if (AUTH_DISABLED) {
          set({ user: { ...PUBLIC_USER }, isAuthenticated: true });
          return;
        }
        set({ isLoading: true });
        const token = get().token;
        api.setToken(token);

        try {
          const result = await api.getMe();

          if (result.data) {
            const currentToken = get().token;
            set({
              user: result.data,
              token: currentToken || COOKIE_SESSION_HINT,
              isAuthenticated: true,
              isLoading: false,
            });
            return;
          }
        } catch {
          // Fall through to unauthenticated state
        }

        api.setToken(null);
        set({
          user: null,
          token: null,
          isAuthenticated: false,
          isLoading: false,
        });
      },
    }),
    {
      name: 'auth-storage',
      partialize: (state) => ({ token: state.token }),
    }
  )
);

export const usePredictionStore = create<PredictionState>((set, get) => ({
  predictions: [],
  currentPrediction: null,
  isLoading: false,

  setPredictions: (predictions) => set({ predictions }),
  setCurrentPrediction: (prediction) => set({ currentPrediction: prediction }),
  addPrediction: (prediction) => set((state) => ({ 
    predictions: [prediction, ...state.predictions] 
  })),
  clearPredictions: () => set({ predictions: [], currentPrediction: null }),

  fetchHistory: async () => {
    set({ isLoading: true });
    const result = await api.getPredictionHistory();
    
    if (result.error) {
      set({ isLoading: false });
      return { success: false, error: result.error };
    }

    set({ predictions: result.data?.predictions || [], isLoading: false });
    return { success: true };
  },

  fetchStats: async () => {
    const result = await api.getPredictionStats();
    
    if (result.error) {
      return { success: false, error: result.error };
    }

    return { success: true, data: result.data };
  },

  predict: async (file: File, model: string) => {
    set({ isLoading: true });
    const result = await api.predictSingle(file, model);
    
    if (result.error) {
      set({ isLoading: false });
      return { success: false, error: result.error };
    }

    const prediction = result.data!;
    set((state) => ({ 
      predictions: [prediction, ...state.predictions],
      currentPrediction: prediction,
      isLoading: false 
    }));
    return { success: true, data: prediction };
  },
}));
