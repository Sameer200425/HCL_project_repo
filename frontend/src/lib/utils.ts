import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function formatDate(dateString: string): string {
  return new Date(dateString).toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

export function getRiskColor(riskLevel: string): string {
  switch (riskLevel.toUpperCase()) {
    case 'CRITICAL':
      return 'bg-red-500 text-white';
    case 'HIGH':
      return 'bg-orange-500 text-white';
    case 'MEDIUM':
      return 'bg-yellow-500 text-black';
    case 'LOW':
      return 'bg-green-500 text-white';
    default:
      return 'bg-gray-500 text-white';
  }
}

export function getClassColor(className: string): string {
  switch (className.toLowerCase()) {
    case 'genuine':
      return 'text-green-500';
    case 'fraud':
      return 'text-red-500';
    case 'tampered':
      return 'text-orange-500';
    case 'forged':
      return 'text-purple-500';
    default:
      return 'text-gray-500';
  }
}

export function formatPercentage(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

export function formatMs(ms: number): string {
  return `${ms.toFixed(1)}ms`;
}
