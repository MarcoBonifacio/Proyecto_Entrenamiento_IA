import { useState, useRef, useCallback } from 'react';
import { useMutation } from '@tanstack/react-query';
import { classifyImageFile } from '../api/classifyApi';
import type { ClassificationResponse } from '../types';

const ANALYSIS_STAGES = [
  'Normalizando tensor de imagen y canales RGB (224×224 px)...',
  'Extrayendo mapas de características con ResNet-50...',
  'Calculando probabilidades y protocolo de disposición...',
];

export const useClassifyImage = () => {
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [statusMessage, setStatusMessage] = useState('Listo para analizar');
  const [finalResult, setFinalResult] = useState<ClassificationResponse | null>(null);
  const timersRef = useRef<number[]>([]);

  const clearTimers = () => {
    timersRef.current.forEach((id) => window.clearTimeout(id));
    timersRef.current = [];
  };

  const mutation = useMutation({
    mutationFn: async (file: File) => {
      clearTimers();
      setIsAnalyzing(true);
      setFinalResult(null);
      setProgress(12);
      setStatusMessage(ANALYSIS_STAGES[0]);

      // Etapa 1 -> Etapa 2 (avance pausado)
      const t1 = window.setTimeout(() => {
        setProgress(38);
        setStatusMessage(ANALYSIS_STAGES[1]);
      }, 950);

      // Etapa 2 -> Etapa 3
      const t2 = window.setTimeout(() => {
        setProgress(64);
        setStatusMessage(ANALYSIS_STAGES[2]);
      }, 1950);

      // Aceleración previa al cierre
      const t3 = window.setTimeout(() => {
        setProgress(86);
      }, 2650);

      timersRef.current.push(t1, t2, t3);

      // Ejecuta la API real en paralelo con un tiempo mínimo de visualización (3.1s)
      const [apiResponse] = await Promise.all([
        classifyImageFile(file),
        new Promise((resolve) => {
          const tEnd = window.setTimeout(resolve, 3100);
          timersRef.current.push(tEnd);
        }),
      ]);

      // Aceleración final al 100%
      setProgress(100);
      setStatusMessage('Análisis completado');

      await new Promise((resolve) => {
        const tReveal = window.setTimeout(resolve, 300);
        timersRef.current.push(tReveal);
      });

      setFinalResult(apiResponse);
      setIsAnalyzing(false);
      return apiResponse;
    },
    onError: () => {
      clearTimers();
      setIsAnalyzing(false);
      setProgress(0);
      setStatusMessage('Error en el análisis');
    },
  });

  const resetAll = useCallback(() => {
    clearTimers();
    setIsAnalyzing(false);
    setProgress(0);
    setStatusMessage('Listo para analizar');
    setFinalResult(null);
    mutation.reset();
  }, [mutation]);

  return {
    classify: mutation.mutate,
    result: finalResult,
    isLoading: isAnalyzing,
    isError: mutation.isError,
    progress,
    statusMessage,
    reset: resetAll,
  };
};
