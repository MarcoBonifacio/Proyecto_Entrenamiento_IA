import { useState, useEffect, useCallback, useRef } from 'react';
import { ResultCard } from './features/classification/components/ResultCard';
import { useClassifyImage } from './features/classification/hooks/useClassifyImage';
import { apiClient } from './services/apiConfig';
import {
  Leaf,
  Upload,
  Camera,
  RotateCcw,
  X,
  SwitchCamera,
  Loader2,
  CheckCircle2,
  Scan,
} from 'lucide-react';

export default function App() {
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  // Estado de disponibilidad del motor IA en el backend
  const [serverReady, setServerReady] = useState(false);

  // Estados para cámara en vivo
  const [isCameraOpen, setIsCameraOpen] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [facingMode, setFacingMode] = useState<'environment' | 'user'>('environment');

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const {
    classify,
    result,
    isLoading,
    isError,
    progress,
    statusMessage,
    reset,
  } = useClassifyImage();

  // Verificar /health al abrir la aplicación hasta que los pesos estén cargados
  useEffect(() => {
    let active = true;
    let timer: number;

    const checkServer = async () => {
      try {
        const res = await apiClient.get('/health', { timeout: 3000 });
        if (active && res.data?.status === 'ok') {
          setServerReady(true);
          return;
        }
      } catch {
        // El servidor aún está cargando los pesos en memoria
      }
      if (active) {
        timer = window.setTimeout(checkServer, 1500);
      }
    };

    checkServer();
    return () => {
      active = false;
      clearTimeout(timer);
    };
  }, []);

  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    setIsCameraOpen(false);
  }, []);

  const startCamera = useCallback(
    async (targetFacingMode: 'environment' | 'user' = facingMode) => {
      setCameraError(null);
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }

      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        setCameraError(
          'Tu navegador bloqueó la cámara. Asegúrate de abrir la página por HTTPS o localhost.'
        );
        return;
      }

      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: { ideal: targetFacingMode },
            width: { ideal: 1280 },
            height: { ideal: 720 },
          },
          audio: false,
        });

        streamRef.current = stream;
        setIsCameraOpen(true);
      } catch {
        setCameraError(
          'No se pudo acceder a la cámara. Revisa los permisos del navegador o cierra otras apps que la estén usando.'
        );
        setIsCameraOpen(false);
      }
    },
    [facingMode]
  );

  useEffect(() => {
    if (isCameraOpen && videoRef.current && streamRef.current) {
      videoRef.current.srcObject = streamRef.current;
    }
  }, [isCameraOpen]);

  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }
    };
  }, []);

  const processFile = useCallback(
    (file: File) => {
      if (!file.type.startsWith('image/')) return;
      stopCamera();
      setPreviewUrl((prevUrl) => {
        if (prevUrl) URL.revokeObjectURL(prevUrl);
        return URL.createObjectURL(file);
      });
      classify(file);
    },
    [classify, stopCamera]
  );

  // Captura recortando el 68% central (zona del marco guía) para eliminar ruido del fondo
  const handleCapturePhoto = () => {
    const video = videoRef.current;
    if (!video) return;

    const vw = video.videoWidth || 640;
    const vh = video.videoHeight || 480;

    // Recorte cuadrado centrado (68% del lado menor) para que el residuo llene el encuadre
    const cropSize = Math.floor(Math.min(vw, vh) * 0.68);
    const sx = Math.floor((vw - cropSize) / 2);
    const sy = Math.floor((vh - cropSize) / 2);

    const canvas = document.createElement('canvas');
    canvas.width = 448;
    canvas.height = 448;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Ligero aumento de contraste (1.08) para resaltar bordes de botellas transparentes y plástico blanco
    ctx.filter = 'contrast(1.08) saturate(1.05)';
    ctx.drawImage(video, sx, sy, cropSize, cropSize, 0, 0, 448, 448);

    canvas.toBlob(
      (blob) => {
        if (blob) {
          const file = new File([blob], `captura-${Date.now()}.jpg`, {
            type: 'image/jpeg',
          });
          processFile(file);
        }
      },
      'image/jpeg',
      0.95
    );
  };

  const handleSwitchCamera = () => {
    const nextMode = facingMode === 'environment' ? 'user' : 'environment';
    setFacingMode(nextMode);
    startCamera(nextMode);
  };

  useEffect(() => {
    const handlePaste = (e: ClipboardEvent) => {
      if (!serverReady) return;
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
  }, [processFile, serverReady]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      processFile(e.target.files[0]);
    }
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    if (!serverReady) return;
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processFile(e.dataTransfer.files[0]);
    }
  };

  const handleClear = () => {
    stopCamera();
    setCameraError(null);
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
      <div className="absolute inset-0 bg-[#0A3323]/15 pointer-events-none" />

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

          <div className="flex items-center gap-2 self-start sm:self-auto">
            <span
              className={`text-xs font-semibold flex items-center gap-1.5 px-3 py-1.5 rounded-xl border ${
                serverReady
                  ? 'bg-emerald-500/15 text-[#0A3323] border-emerald-600/30'
                  : 'bg-amber-500/20 text-amber-950 border-amber-600/30'
              }`}
            >
              {serverReady ? (
                <>
                  <CheckCircle2 className="w-3.5 h-3.5 text-emerald-700" /> Motor IA listo
                </>
              ) : (
                <>
                  <Loader2 className="w-3.5 h-3.5 animate-spin text-amber-700" /> Cargando modelo...
                </>
              )}
            </span>

            {(previewUrl || isCameraOpen) && (
              <button
                onClick={handleClear}
                className="text-xs font-semibold flex items-center gap-1.5 text-[#0A3323] bg-white/60 hover:bg-white/85 px-3 py-1.5 rounded-xl border border-white/70 transition-all cursor-pointer shadow-2xs"
              >
                <RotateCcw className="w-3.5 h-3.5" /> Nueva consulta
              </button>
            )}
          </div>
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
                <label className={!serverReady ? 'pointer-events-none opacity-55' : ''}>
                  <input
                    type="file"
                    accept="image/*"
                    disabled={!serverReady}
                    onChange={handleFileChange}
                    className="hidden"
                  />
                  <span className="w-full flex items-center justify-center gap-2 bg-[#0A3323] text-white py-3 px-4 rounded-xl font-semibold text-sm cursor-pointer hover:bg-[#072519] transition-all shadow-sm">
                    <Upload className="w-4 h-4" /> Subir Imagen
                  </span>
                </label>

                <button
                  type="button"
                  disabled={!serverReady}
                  onClick={() => (isCameraOpen ? stopCamera() : startCamera())}
                  className="w-full flex items-center justify-center gap-2 bg-white/75 text-[#0A3323] border border-white/80 py-3 px-4 rounded-xl font-semibold text-sm cursor-pointer hover:bg-white transition-all shadow-2xs disabled:opacity-55 disabled:cursor-not-allowed"
                >
                  {isCameraOpen ? (
                    <>
                      <X className="w-4 h-4" /> Cerrar Cámara
                    </>
                  ) : (
                    <>
                      <Camera className="w-4 h-4" /> Usar Cámara
                    </>
                  )}
                </button>
              </div>

              {cameraError && (
                <p className="text-xs text-red-700 bg-red-100/80 border border-red-300 rounded-xl p-2.5 mb-3">
                  {cameraError}
                </p>
              )}
            </div>

            {/* Área de cámara en vivo, vista previa o placeholder */}
            <div className="flex-1 min-h-[240px] rounded-xl border border-white/60 bg-white/30 flex items-center justify-center p-3 overflow-hidden">
              {isCameraOpen ? (
                <div className="relative w-full h-full flex flex-col items-center justify-center gap-3">
                  {/* Contenedor de video con marco guía central */}
                  <div className="relative w-full max-h-64 rounded-lg overflow-hidden shadow-sm bg-black flex items-center justify-center">
                    <video
                      ref={videoRef}
                      autoPlay
                      playsInline
                      muted
                      className="w-full max-h-64 object-cover"
                    />
                    {/* Visor guía centrado */}
                    <div className="absolute inset-0 pointer-events-none flex flex-col items-center justify-center">
                      <div className="w-44 h-44 border-2 border-dashed border-[#d9f99d] rounded-2xl shadow-[0_0_0_9999px_rgba(10,51,35,0.35)] flex items-end justify-center pb-2">
                        <span className="text-[10px] font-semibold bg-[#0A3323]/85 text-[#F7F4D5] px-2 py-0.5 rounded-full flex items-center gap-1">
                          <Scan className="w-3 h-3" /> Centra el residuo aquí
                        </span>
                      </div>
                    </div>
                  </div>

                  <div className="flex items-center gap-2 w-full">
                    <button
                      type="button"
                      onClick={handleCapturePhoto}
                      className="flex-1 flex items-center justify-center gap-2 bg-[#0A3323] text-white py-2.5 px-4 rounded-xl font-semibold text-sm cursor-pointer hover:bg-[#072519] transition-all shadow-sm"
                    >
                      <Camera className="w-4 h-4" /> Capturar y Clasificar
                    </button>
                    <button
                      type="button"
                      onClick={handleSwitchCamera}
                      title="Cambiar cámara"
                      className="p-2.5 bg-white/80 text-[#0A3323] rounded-xl border border-white hover:bg-white cursor-pointer"
                    >
                      <SwitchCamera className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              ) : previewUrl ? (
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
                      <div className="w-full h-12 bg-gradient-to-t from-[#839958]/40 via-[#839958]/12 to-transparent blur-[2px]" />
                      <div className="w-full h-[2px] bg-gradient-to-r from-transparent via-[#d9f99d] to-transparent shadow-[0_0_16px_3px_rgba(131,153,88,0.9)]" />
                    </div>
                  )}
                </div>
              ) : (
                <div className="text-center text-[#0A3323]/65 space-y-2 p-4">
                  <Camera className="w-10 h-10 mx-auto opacity-50" />
                  <p className="text-sm font-medium">
                    Sube, arrastra o pega una foto (
                    <kbd className="px-1.5 py-0.5 text-xs bg-white/70 rounded border border-[#0A3323]/20 font-mono">
                      Ctrl + V
                    </kbd>
                    )
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
