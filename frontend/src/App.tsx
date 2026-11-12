import { useState, useEffect, useCallback } from 'react';
import { ResultCard } from './features/classification/components/ResultCard';
import { useClassifyImage } from './features/classification/hooks/useClassifyImage';
import { Leaf, Upload, Camera, RotateCcw } from 'lucide-react';

export default function App() {
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  const {
    classify,
    result,
    isLoading,
    isError,
    progress,
    statusMessage,
    reset,
  } = useClassifyImage();

  const processFile = useCallback(
    (file: File) => {
      if (!file.type.startsWith('image/')) return;
      setPreviewUrl((prevUrl) => {
        if (prevUrl) URL.revokeObjectURL(prevUrl);
        return URL.createObjectURL(file);
      });
      classify(file);
    },
    [classify]
  );

  // Soporte global para pegar imágenes desde el portapapeles (Ctrl + V)
  useEffect(() => {
    const handlePaste = (e: ClipboardEvent) => {
      const items = e.clipboardData?.items;
      if (!items) return;

      for (let i = 0; i < items.length; i++) {
        if (items[i].type.startsWith('image/')) {
          const file = items[i].getAsFile();
          if (file) {
            e.preventDefault();
            processFile(file);
            break;
          }
        }
      }
    };

    window.addEventListener('paste', handlePaste);
    return () => window.removeEventListener('paste', handlePaste);
  }, [processFile]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      processFile(e.target.files[0]);
    }
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processFile(e.dataTransfer.files[0]);
    }
  };

  const handleClear = () => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(null);
    reset();
  };

  return (
    <main
      className="relative min-h-screen p-4 md:p-8 flex flex-col items-center justify-center bg-cover bg-center bg-no-repeat"
      style={{
        backgroundImage: "url('/bg-eco.jpg')",
      }}
    >
      {/* Sutil velo ambiental para equilibrar luces y sombras del fondo */}
      <div className="absolute inset-0 bg-[#0A3323]/15 pointer-events-none" />

      {/* Gran Contenedor Único Glassmorphism */}
      <div className="relative z-10 max-w-5xl w-full bg-white/40 backdrop-blur-xl rounded-3xl p-6 md:p-8 border border-white/65 shadow-2xl">
        {/* Encabezado Superior Integrado */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-6 mb-6 border-b border-white/50">
          <div className="flex items-center gap-2.5">
            <div className="bg-[#0A3323] p-2 rounded-xl text-[#F7F4D5] shadow-sm">
              <Leaf className="w-6 h-6" />
            </div>
            <div>
              <h1 className="text-2xl md:text-3xl font-extrabold text-[#0A3323] tracking-tight leading-none">
                EcoClasifica
              </h1>
              <p className="text-xs md:text-sm text-[#0A3323]/80 mt-1">
                Visión Artificial y Transfer Learning (ResNet-50) para clasificación de residuos
              </p>
            </div>
          </div>

          {previewUrl && (
            <button
              onClick={handleClear}
              className="self-start sm:self-auto text-xs font-semibold flex items-center gap-1.5 text-[#0A3323] bg-white/60 hover:bg-white/85 px-3 py-2 rounded-xl border border-white/70 transition-all cursor-pointer shadow-2xs"
            >
              <RotateCcw className="w-3.5 h-3.5" /> Nueva consulta
            </button>
          )}
        </div>

        {/* Cuerpo Interior: Dos Columnas dentro del mismo cuadro */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 items-stretch">
          {/* Columna Izquierda: Carga y Vista Previa */}
          <div
            onDragOver={(e) => {
              e.preventDefault();
              setIsDragging(true);
            }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={handleDrop}
            className={`bg-white/45 backdrop-blur-md rounded-2xl p-5 border transition-all flex flex-col justify-between ${
              isDragging
                ? 'border-[#0A3323] ring-2 ring-[#0A3323]/30 bg-white/65'
                : 'border-white/60'
            }`}
          >
            <div>
              <h2 className="text-lg font-bold text-[#0A3323] mb-4">
                Sube o toma una foto de tu residuo
              </h2>

              <div className="grid grid-cols-2 gap-3 mb-4">
                <label>
                  <input
                    type="file"
                    accept="image/*"
                    onChange={handleFileChange}
                    className="hidden"
                  />
                  <span className="w-full flex items-center justify-center gap-2 bg-[#0A3323] text-white py-3 px-4 rounded-xl font-semibold text-sm cursor-pointer hover:bg-[#072519] transition-all shadow-sm">
                    <Upload className="w-4 h-4" /> Subir Imagen
                  </span>
                </label>

                <label>
                  <input
                    type="file"
                    accept="image/*"
                    capture="environment"
                    onChange={handleFileChange}
                    className="hidden"
                  />
                  <span className="w-full flex items-center justify-center gap-2 bg-white/75 text-[#0A3323] border border-white/80 py-3 px-4 rounded-xl font-semibold text-sm cursor-pointer hover:bg-white transition-all shadow-2xs">
                    <Camera className="w-4 h-4" /> Usar Cámara
                  </span>
                </label>
              </div>
            </div>

            {/* Área de vista previa o placeholder centrado */}
            <div className="flex-1 min-h-[240px] rounded-xl border border-white/60 bg-white/30 flex items-center justify-center p-3 overflow-hidden">
                  {previewUrl ? (
                    <div className="relative flex items-center justify-center w-full h-full">
                      <img
                        src={previewUrl}
                        alt="Vista previa del residuo"
                        className={`max-h-60 max-w-full object-contain rounded-lg shadow-sm transition-all duration-300 ${
                          isLoading ? 'brightness-90 scale-[0.99]' : ''
                        }`}
                      />
                    {isLoading && (
                      <div className="absolute inset-x-3 h-14 pointer-events-none animate-scan-beam flex flex-col justify-end">
                        {/* Estela luminosa difuminada detrás del haz */}
                        <div className="w-full h-12 bg-gradient-to-t from-[#839958]/40 via-[#839958]/12 to-transparent blur-[2px]" />
                        {/* Núcleo del haz óptico con resplandor */}
                        <div className="w-full h-[2px] bg-gradient-to-r from-transparent via-[#d9f99d] to-transparent shadow-[0_0_16px_3px_rgba(131,153,88,0.9)]" />
                      </div>
                    )}
                    </div>
                  ) : (
                <div className="text-center text-[#0A3323]/65 space-y-2 p-4">
                  <Camera className="w-10 h-10 mx-auto opacity-50" />
                  <p className="text-sm font-medium">
                    Sube, arrastra o pega una foto (<kbd className="px-1.5 py-0.5 text-xs bg-white/70 rounded border border-[#0A3323]/20 font-mono">Ctrl + V</kbd>)
                  </p>
                  <p className="text-xs opacity-75">Soporta JPG, PNG y WEBP</p>
                </div>
              )}
            </div>
          </div>

          {/* Columna Derecha: Progreso, Material y Recomendación */}
          <ResultCard
            result={result}
            isLoading={isLoading}
            isError={isError}
            progress={progress}
            statusMessage={statusMessage}
          />
        </div>
      </div>
    </main>
  );
}
