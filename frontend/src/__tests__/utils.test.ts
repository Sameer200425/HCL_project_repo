/**
 * Tests for utility functions
 */
import { cn, formatDate, getRiskColor, getClassColor } from '@/lib/utils';

describe('cn (className merge)', () => {
  it('merges class names', () => {
    expect(cn('foo', 'bar')).toBe('foo bar');
  });

  it('handles conditional classes', () => {
    expect(cn('base', false && 'hidden', 'extra')).toBe('base extra');
  });

  it('deduplicates tailwind classes', () => {
    const result = cn('p-4', 'p-2');
    expect(result).toBe('p-2');
  });
});

describe('formatDate', () => {
  it('formats a date string', () => {
    const result = formatDate('2026-03-01T12:00:00Z');
    expect(result).toContain('2026');
    expect(result).toContain('Mar');
  });
});

describe('getRiskColor', () => {
  it('returns correct color for each risk level', () => {
    expect(getRiskColor('CRITICAL')).toContain('red');
    expect(getRiskColor('HIGH')).toContain('orange');
    expect(getRiskColor('MEDIUM')).toContain('yellow');
    expect(getRiskColor('LOW')).toContain('green');
  });

  it('returns gray for unknown levels', () => {
    expect(getRiskColor('unknown')).toContain('gray');
  });

  it('is case-insensitive', () => {
    expect(getRiskColor('low')).toContain('green');
    expect(getRiskColor('Critical')).toContain('red');
  });
});

describe('getClassColor', () => {
  it('returns green for genuine', () => {
    expect(getClassColor('genuine')).toContain('green');
  });

  it('returns red for fraud', () => {
    expect(getClassColor('fraud')).toContain('red');
  });

  it('returns orange for tampered', () => {
    expect(getClassColor('tampered')).toContain('orange');
  });
});
