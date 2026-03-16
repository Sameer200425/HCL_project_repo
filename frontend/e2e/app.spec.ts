import { test, expect } from '@playwright/test';

/**
 * E2E Tests — App Landing & Auth Flow
 * ============================================
 * Run:   npx playwright test
 * UI:    npx playwright test --ui
 */

test.describe('Landing Page', () => {
  test('shows hero heading and CTA buttons', async ({ page }) => {
    await page.goto('/');
    await expect(page.getByRole('heading', { level: 1 })).toBeVisible();
    await expect(page.getByRole('link', { name: /get started/i })).toBeVisible();
  });

  test('navigates to login page', async ({ page }) => {
    await page.goto('/');
    await page.getByRole('link', { name: /login|sign in/i }).first().click();
    await expect(page).toHaveURL(/\/login/);
  });
});

test.describe('Registration & Login', () => {
  const uniqueEmail = `e2e_${Date.now()}@test.com`;
  const password = 'TestP@ss123';
  const username = `e2e_${Date.now()}`;

  test('registers a new user', async ({ page }) => {
    await page.goto('/register');

    await page.getByLabel(/email/i).fill(uniqueEmail);
    await page.getByLabel(/username/i).fill(username);
    await page.getByLabel(/^password$/i).fill(password);

    await page.getByRole('button', { name: /register|sign up|create/i }).click();

    // Should redirect to login or show success
    await expect(page).toHaveURL(/\/(login|dashboard)/, { timeout: 10_000 });
  });

  test('logs in with valid credentials', async ({ page }) => {
    await page.goto('/login');

    await page.getByLabel(/email/i).fill(uniqueEmail);
    await page.getByLabel(/password/i).fill(password);
    await page.getByRole('button', { name: /login|sign in/i }).click();

    // Should redirect to dashboard
    await expect(page).toHaveURL(/\/dashboard/, { timeout: 10_000 });
    await expect(page.getByText(/dashboard/i)).toBeVisible();
  });

  test('rejects invalid credentials', async ({ page }) => {
    await page.goto('/login');

    await page.getByLabel(/email/i).fill('wrong@test.com');
    await page.getByLabel(/password/i).fill('WrongPass');
    await page.getByRole('button', { name: /login|sign in/i }).click();

    // Should show error message
    await expect(page.getByText(/error|invalid|incorrect|failed/i)).toBeVisible({ timeout: 5_000 });
  });
});

test.describe('Dashboard (authenticated)', () => {
  const email = `dash_${Date.now()}@test.com`;
  const password = 'DashP@ss123';
  const username = `dash_${Date.now()}`;

  test.beforeAll(async ({ request }) => {
    // Pre-register via API
    const apiBase = process.env.API_URL || 'http://localhost:8000';
    await request.post(`${apiBase}/api/auth/register`, {
      data: { email, username, password },
    });
  });

  test.beforeEach(async ({ page }) => {
    // Login via UI
    await page.goto('/login');
    await page.getByLabel(/email/i).fill(email);
    await page.getByLabel(/password/i).fill(password);
    await page.getByRole('button', { name: /login|sign in/i }).click();
    await expect(page).toHaveURL(/\/dashboard/, { timeout: 10_000 });
  });

  test('displays stat cards', async ({ page }) => {
    await expect(page.getByText(/total scans/i)).toBeVisible();
    await expect(page.getByText(/fraud detected/i)).toBeVisible();
    await expect(page.getByText(/genuine documents/i)).toBeVisible();
  });

  test('navigate to predict page', async ({ page }) => {
    await page.getByRole('link', { name: /analyze/i }).first().click();
    await expect(page).toHaveURL(/\/predict/);
  });
});

test.describe('Demo Page', () => {
  test('loads and shows demo scenarios', async ({ page }) => {
    await page.goto('/demo');
    await expect(page.getByText(/demo/i)).toBeVisible();
  });
});
