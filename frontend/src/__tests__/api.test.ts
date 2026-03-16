/**
 * Tests for API client
 */

// Mock fetch globally
const mockFetch = jest.fn();
global.fetch = mockFetch;

// Mock localStorage
const mockStorage: Record<string, string> = {};
Object.defineProperty(window, 'localStorage', {
  value: {
    getItem: (key: string) => mockStorage[key] || null,
    setItem: (key: string, val: string) => { mockStorage[key] = val; },
    removeItem: (key: string) => { delete mockStorage[key]; },
    clear: () => { Object.keys(mockStorage).forEach(k => delete mockStorage[k]); },
  },
});

import { api } from '@/lib/api';

beforeEach(() => {
  mockFetch.mockReset();
  Object.keys(mockStorage).forEach(k => delete mockStorage[k]);
  api.setToken(null);
});

describe('ApiClient', () => {
  describe('setToken / getToken', () => {
    it('stores and retrieves token', () => {
      api.setToken('test-token');
      expect(api.getToken()).toBe('test-token');
      expect(mockStorage['token']).toBe('test-token');
    });

    it('clears token on null', () => {
      api.setToken('abc');
      api.setToken(null);
      expect(api.getToken()).toBeNull();
      expect(mockStorage['token']).toBeUndefined();
    });
  });

  describe('register', () => {
    it('sends registration request', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ id: 1, email: 'a@b.com', username: 'user' }),
      });

      const result = await api.register('a@b.com', 'user', 'pass123');
      expect(result.data).toEqual({ id: 1, email: 'a@b.com', username: 'user' });

      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/auth/register'),
        expect.objectContaining({ method: 'POST' })
      );
    });
  });

  describe('login', () => {
    it('sends login request', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          access_token: 'tok',
          refresh_token: 'ref',
          token_type: 'bearer',
          expires_in: 86400,
        }),
      });

      const result = await api.login('a@b.com', 'pass');
      expect(result.data?.access_token).toBe('tok');
    });

    it('returns error on failure', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        json: async () => ({ detail: 'Invalid credentials' }),
      });

      const result = await api.login('a@b.com', 'wrong');
      expect(result.error).toBe('Invalid credentials');
    });
  });

  describe('predictSingle', () => {
    it('sends FormData with file', async () => {
      api.setToken('tok');
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          id: 1,
          predicted_class: 'genuine',
          confidence: 0.95,
          risk_level: 'LOW',
        }),
      });

      const file = new File(['fake'], 'doc.png', { type: 'image/png' });
      const result = await api.predictSingle(file, 'cnn');
      expect(result.data?.predicted_class).toBe('genuine');

      const call = mockFetch.mock.calls[0];
      expect(call[0]).toContain('/api/predict/single?model=cnn');
      expect(call[1].body).toBeInstanceOf(FormData);
    });
  });

  describe('getPredictionStats', () => {
    it('fetches stats', async () => {
      api.setToken('tok');
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          total_predictions: 42,
          fraud_detected: 5,
          genuine_documents: 37,
          by_class: { genuine: 37, fraud: 5 },
          by_risk_level: { low: 37, critical: 5 },
          by_model: { cnn: 42 },
          avg_confidence: 0.92,
          avg_inference_time_ms: 20.1,
        }),
      });

      const result = await api.getPredictionStats();
      expect(result.data?.total_predictions).toBe(42);
      expect(result.data?.fraud_detected).toBe(5);
    });
  });

  describe('network error handling', () => {
    it('returns network error on fetch failure', async () => {
      mockFetch.mockRejectedValueOnce(new Error('Network fail'));
      const result = await api.login('a@b.com', 'pass');
      expect(result.error).toBe('Network error');
    });
  });

  describe('health', () => {
    it('calls health endpoint', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ status: 'healthy', gpu_available: false }),
      });

      const result = await api.health();
      expect(result.data?.status).toBe('healthy');
    });
  });
});
