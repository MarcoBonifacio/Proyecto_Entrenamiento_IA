import { apiClient } from '../../../services/apiConfig';
import type { ClassifyApiResponse, ClassificationResponse } from '../types';

/** Convierte cualquier valor a probabilidad 0-1 finita (acepta 0-100 por seguridad). */
const toProbability = (value: unknown): number => {
  const n = typeof value === 'number' ? value : Number(value);
  if (!Number.isFinite(n) || n < 0) return 0;
  return n > 1 ? Math.min(n / 100, 1) : n;
};

export const normalizeClassification = (raw: ClassifyApiResponse): ClassificationResponse => {
  const breakdown = Object.entries(raw.todas_las_predicciones ?? {})
    .map(([label, p]) => ({ label, percent: toProbability(p) * 100 }))
    .sort((a, b) => b.percent - a.percent);

  return {
    prediction: raw.categoria?.trim() || 'Desconocido',
    confidence: toProbability(raw.confianza),
    emoji: raw.emoji ?? '',
    disposal_recommendation: raw.recomendacion || undefined,
    breakdown,
  };
};

export const classifyImageFile = async (file: File): Promise<ClassificationResponse> => {
  const formData = new FormData();
  formData.append('file', file);

  const response = await apiClient.post<ClassifyApiResponse>('/classify', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });

  return normalizeClassification(response.data);
};
