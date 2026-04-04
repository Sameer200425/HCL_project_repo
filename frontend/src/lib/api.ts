/**
 * API Client for Backend Communication
 */

const DIRECT_API_URL = (process.env.NEXT_PUBLIC_API_URL || '').trim();
const API_TIMEOUT_MS = Number(process.env.NEXT_PUBLIC_API_TIMEOUT_MS || 12000);
const API_RETRY_COUNT = Number(process.env.NEXT_PUBLIC_API_RETRY_COUNT || 1);

function normalizeBase(base: string): string {
  return base.endsWith('/') ? base.slice(0, -1) : base;
}

const API_BASE_CANDIDATES = [
  DIRECT_API_URL ? normalizeBase(DIRECT_API_URL) : '',
  '', // Use Next.js rewrite proxy by default for stable local/dev routing.
  'http://localhost:8001',
  'http://localhost:8000',
].filter((value, index, arr) => arr.indexOf(value) === index);
const AUTH_DISABLED = process.env.NEXT_PUBLIC_AUTH_DISABLED !== 'false';
const PUBLIC_TOKEN = 'public-access';
const COOKIE_SESSION_HINT = 'cookie-session';
const USE_COOKIE_AUTH = !AUTH_DISABLED;

interface ApiResponse<T> {
  data?: T;
  error?: string;
}

function classifyNetworkError(error: unknown): string {
  const message = error instanceof Error ? error.message.toLowerCase() : '';
  if (message.includes('aborted') || message.includes('timeout')) {
    return 'Request timed out. Please try again.';
  }
  return 'Network error';
}

class ApiClient {
  private token: string | null = null;

  setToken(token: string | null) {
    if (AUTH_DISABLED) {
      this.token = token && token !== PUBLIC_TOKEN ? token : null;
    } else {
      this.token = token;
    }

    if (typeof window === 'undefined') return;

    if (this.token) {
      // Keep bearer token fallback even in cookie mode to avoid false auth failures
      // when browser cookie policies block cross-site dev requests.
      localStorage.setItem('token', this.token);
    } else {
      localStorage.removeItem('token');
    }
  }

  getToken(): string | null {
    if (AUTH_DISABLED) {
      return null;
    }
    if (!this.token && typeof window !== 'undefined') {
      this.token = localStorage.getItem('token');
    }
    if (USE_COOKIE_AUTH && this.token === COOKIE_SESSION_HINT) {
      return null;
    }
    return this.token;
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<ApiResponse<T>> {
    const token = this.getToken();
    const method = (options.method || 'GET').toUpperCase();
    const retryable = method === 'GET';

    const headers: Record<string, string> = {
      ...(options.headers as Record<string, string>),
    };

    if (token) {
      headers['Authorization'] = `Bearer ${token}`;
    }

    if (!(options.body instanceof FormData)) {
      headers['Content-Type'] = 'application/json';
    }

    for (let i = 0; i < API_BASE_CANDIDATES.length; i += 1) {
      const base = API_BASE_CANDIDATES[i];
      const url = `${base}${endpoint}`;
      for (let attempt = 0; attempt <= API_RETRY_COUNT; attempt += 1) {
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), API_TIMEOUT_MS);
        try {
          const response = await fetch(url, {
            ...options,
            headers,
            credentials: 'include',
            signal: controller.signal,
          });

          clearTimeout(timeout);

          if (!response.ok) {
            const error = await response.json().catch(() => ({ detail: 'Request failed' }));
            const detail = error.detail || 'Request failed';
            if (response.status === 401) {
              return { error: detail || 'Invalid credentials' };
            }
            return { error: detail };
          }

          const data = await response.json();
          return { data };
        } catch (error) {
          clearTimeout(timeout);
          const shouldRetry = retryable && attempt < API_RETRY_COUNT;
          if (shouldRetry) {
            await new Promise((resolve) => setTimeout(resolve, 350 * (attempt + 1)));
            continue;
          }
          if (i === API_BASE_CANDIDATES.length - 1) {
            const classified = classifyNetworkError(error);
            console.error('API request failed', { url, endpoint, method, error });
            return { error: classified };
          }
          break;
        }
      }
    }

    return { error: 'Network error' };
  }

  // Auth endpoints
  async register(email: string, username: string, password: string, fullName?: string) {
    return this.request<{ id: number; email: string; username: string }>('/api/auth/register', {
      method: 'POST',
      body: JSON.stringify({ email, username, password, full_name: fullName }),
    });
  }

  async login(identifier: string, password: string) {
    return this.request<{
      access_token: string;
      refresh_token: string;
      token_type: string;
      expires_in: number;
    }>('/api/auth/login/json', {
      method: 'POST',
      body: JSON.stringify({ email: identifier, password }),
    });
  }

  async logout() {
    return this.request<{ message: string }>('/api/auth/logout', {
      method: 'POST',
    });
  }

  async forgotPassword(email: string) {
    return this.request<{ message: string; reset_token?: string }>('/api/auth/forgot-password', {
      method: 'POST',
      body: JSON.stringify({ email }),
    });
  }

  async resetPassword(token: string, password: string) {
    return this.request<{ message: string }>('/api/auth/reset-password', {
      method: 'POST',
      body: JSON.stringify({ token, password }),
    });
  }

  async getMe() {
    return this.request<{
      id: number;
      email: string;
      username: string;
      full_name: string | null;
      role: string;
      is_active: boolean;
      created_at: string;
    }>('/api/auth/me');
  }

  // Prediction endpoints
  async predictSingle(file: File, model: string = 'cnn') {
    const formData = new FormData();
    formData.append('file', file);

    return this.request<{
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
    }>(`/api/predict/single?model=${model}`, {
      method: 'POST',
      body: formData,
    });
  }

  async predictBatch(files: File[], model: string = 'cnn') {
    const formData = new FormData();
    files.forEach((file) => formData.append('files', file));

    return this.request<Array<{
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
    }>>(`/api/predict/batch?model=${model}`, {
      method: 'POST',
      body: formData,
    });
  }

  async getPredictionHistory(page: number = 1, pageSize: number = 20) {
    return this.request<{
      predictions: Array<{
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
      }>;
      total: number;
      page: number;
      page_size: number;
    }>(`/api/predict/history?page=${page}&page_size=${pageSize}`);
  }

  async getPredictionStats() {
    return this.request<{
      total_predictions: number;
      fraud_detected: number;
      genuine_documents: number;
      by_class: Record<string, number>;
      by_risk_level: Record<string, number>;
      by_model: Record<string, number>;
      avg_confidence: number;
      avg_inference_time_ms: number;
    }>('/api/predict/stats');
  }

  async deletePrediction(id: number) {
    return this.request<{ message: string }>(`/api/predict/${id}`, {
      method: 'DELETE',
    });
  }

  // Models
  async getModels() {
    return this.request<Array<{
      name: string;
      id: string;
      version: string;
      classes: string[];
      input_size: number[];
      parameters: number;
      loaded: boolean;
    }>>('/api/models');
  }

  // Health
  async health() {
    return this.request<{
      status: string;
      database: string;
      models_loaded: string[];
      gpu_available: boolean;
      version: string;
    }>('/health');
  }

  // Generic GET request
  async get<T>(endpoint: string): Promise<T> {
    const result = await this.request<T>(endpoint);
    if (result.error) {
      throw new Error(result.error);
    }
    return result.data as T;
  }

  // Generic POST request
  async post<T>(endpoint: string, body: unknown): Promise<T> {
    const result = await this.request<T>(endpoint, {
      method: 'POST',
      body: JSON.stringify(body),
    });
    if (result.error) {
      throw new Error(result.error);
    }
    return result.data as T;
  }
}

export const api = new ApiClient();
