export type AuthFormField = 'identifier' | 'email' | 'username' | 'password' | 'confirmPassword';

export interface AuthValidationResult {
  isValid: boolean;
  field?: AuthFormField;
  message?: string;
}

export interface RegisterInput {
  email: string;
  username: string;
  password: string;
  confirmPassword: string;
}

export function normalizeIdentifier(value: string): string {
  return value.trim();
}

export function normalizeEmail(value: string): string {
  return value.trim().toLowerCase();
}

export function normalizeUsername(value: string): string {
  return value.trim();
}

export function validateLoginInput(identifier: string, password: string): AuthValidationResult {
  const cleanIdentifier = normalizeIdentifier(identifier);
  if (!cleanIdentifier) {
    return { isValid: false, field: 'identifier', message: 'Enter your email address.' };
  }
  if (!password) {
    return { isValid: false, field: 'password', message: 'Enter your password.' };
  }
  return { isValid: true };
}

export function validateRegisterInput(input: RegisterInput): AuthValidationResult {
  const email = normalizeEmail(input.email);
  const username = normalizeUsername(input.username);
  const password = input.password;
  const confirmPassword = input.confirmPassword;

  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  const usernameRegex = /^[a-zA-Z0-9_]{3,30}$/;
  const passwordRegex = /^(?=.*[A-Za-z])(?=.*\d).{8,}$/;

  if (!email) {
    return { isValid: false, field: 'email', message: 'Email is required.' };
  }
  if (!emailRegex.test(email)) {
    return { isValid: false, field: 'email', message: 'Enter a valid email address.' };
  }
  if (!username) {
    return { isValid: false, field: 'username', message: 'Username is required.' };
  }
  if (!usernameRegex.test(username)) {
    return {
      isValid: false,
      field: 'username',
      message: 'Username must be 3-30 chars and use letters, numbers, or underscore.',
    };
  }
  if (!passwordRegex.test(password)) {
    return {
      isValid: false,
      field: 'password',
      message: 'Password must be at least 8 characters and include a letter and a number.',
    };
  }
  if (password !== confirmPassword) {
    return { isValid: false, field: 'confirmPassword', message: 'Passwords do not match.' };
  }

  return { isValid: true };
}

export function mapAuthErrorMessage(rawError?: string): string {
  const message = (rawError || '').trim();
  if (!message) return 'Authentication failed. Please try again.';

  const lower = message.toLowerCase();
  if (lower.includes('rate limit') || lower.includes('too many')) {
    return 'Too many attempts. Please wait a minute and try again.';
  }
  if (lower.includes('inactive')) {
    return 'Your account is inactive. Please contact support.';
  }
  if (lower.includes('incorrect') || lower.includes('invalid credentials')) {
    return 'Invalid email/username or password.';
  }
  if (lower.includes('already registered') || lower.includes('already taken') || lower.includes('already exists')) {
    return 'An account already exists for these details.';
  }
  if (lower.includes('network')) {
    return 'Network error. Check your connection and try again.';
  }
  return message;
}

export function isDuplicateAccountError(rawError?: string): boolean {
  const lower = (rawError || '').toLowerCase();
  return lower.includes('already registered') || lower.includes('already taken') || lower.includes('already exists');
}