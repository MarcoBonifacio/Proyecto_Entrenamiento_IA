import React from 'react';
import type { ClassificationResponse } from '../types';
import { CheckCircle2, AlertCircle, Cpu, Sparkles } from 'lucide-react';
import { TypewriterText } from './TypewriterText';

interface ResultCardProps {
  result: ClassificationResponse | null;
  isLoading: boolean;
  isError?: boolean;
  progress: number;
  statusMessage: string;
}

export const ResultCard: React.FC<ResultCardProps> = ({
  result,
  isLoading,
  isError,
  progress,
  statusMessage,
}) => {
  if (isError) {
    return (
      <div className="bg-red-50/75 backdrop-blur-md rounded-2xl p-6 border border-red-200/80 text-red-950 space-y-2 shadow-sm">
        <div className="flex items-center gap-2 font-semibold">
          <AlertCircle className="w-5 h-5 text-red-600" />
          <span>Sin conexión con el servidor de inferencia</span>
        </div>
        <p className="text-sm text-red-900/80">
          Verifica que FastAPI esté activo en <code className="font-mono">http://localhost:8000</code>.
        </p>
      </div>
    );
  }

  const recommendationText = result
    ? result.disposal_recommendation ||
      `Depositar el residuo en el contenedor correspondiente a ${result.prediction}.`
    : '';

  return (
    <div className="flex flex-col gap-4 h-full justify-between">
      {/* Bloque 1: Estado del pipeline y barra de progreso */}
      <div className="bg-white/45 backdrop-blur-md rounded-2xl p-5 border border-white/60 shadow-sm">
        <div className="flex items-center justify-between gap-2 mb-3">
          <div className="flex items-center gap-2 text-sm font-semibold text-[#0A3323]">
            <Cpu className={`w-4 h-4 text-[#839958] ${isLoading ? 'animate-spin' : ''}`} />
            <span>{isLoading ? statusMessage : result ? 'Inferencia completada' : 'Motor ResNet-50 en espera'}</span>
          </div>
          {(isLoading || result) && (
            <span className="text-xs font-bold text-[#0A3323]/75">{progress}%</span>
          )}
        </div>

        {/* Barra de carga con aceleración suave al final */}
        <div className="w-full bg-[#0A3323]/10 h-2.5 rounded-full overflow-hidden">
          <div
            className="bg-[#839958] h-full rounded-full"
            style={{
              width: `${isLoading || result ? progress : 0}%`,
              transition:
                progress >= 85
                  ? 'width 280ms cubic-bezier(0.22, 1, 0.36, 1)'
                  : 'width 850ms ease-out',
            }}
          />
        </div>
      </div>

      {/* Bloque 2: Material Detectado */}
      <div className="bg-white/45 backdrop-blur-md rounded-2xl p-5 border border-white/60 shadow-sm">
        {result && !isLoading ? (
          <div>
            <span className="text-xs font-bold tracking-wider text-[#839958] uppercase">
              Material Detectado
            </span>
            <h3 className="text-3xl font-extrabold text-[#0A3323] capitalize mt-0.5">
              {result.prediction}
            </h3>
            <p className="text-sm font-medium text-[#0A3323]/80 mt-1">
              Índice de confianza: <span className="font-bold">{(result.confidence * 100).toFixed(2)}%</span>
            </p>
          </div>
        ) : (
          <div className="py-2">
            <span className="text-xs font-bold tracking-wider text-[#0A3323]/50 uppercase">
              Material Detectado
            </span>
            <p className="text-lg font-semibold text-[#0A3323]/40 mt-1">
              {isLoading ? 'Procesando vectores de imagen...' : 'Esperando captura o archivo'}
            </p>
          </div>
        )}
      </div>

      {/* Bloque 3: Desecho Recomendado (texto palabra por palabra) */}
      <div className="bg-white/45 backdrop-blur-md rounded-2xl p-5 border border-white/60 shadow-sm flex-1 flex flex-col justify-between min-h-[145px]">
        <div>
          <div className="flex items-center gap-2 text-[#0A3323] font-bold text-base mb-2.5">
            {result && !isLoading ? (
              <CheckCircle2 className="w-5 h-5 text-[#839958]" />
            ) : (
              <Sparkles className="w-5 h-5 text-[#0A3323]/40" />
            )}
            <span>Desecho Recomendado</span>
          </div>

          {result && !isLoading ? (
            <p className="text-sm text-[#0A3323]/90 leading-relaxed font-medium">
              <TypewriterText text={recommendationText} speedMs={60} />
            </p>
          ) : (
            <p className="text-sm text-[#0A3323]/50 leading-relaxed">
              {isLoading
                ? 'Generando protocolo de separación y reciclaje...'
                : 'Las instrucciones de disposición ecológica aparecerán aquí al finalizar el análisis.'}
            </p>
          )}
        </div>
      </div>
    </div>
  );
};
