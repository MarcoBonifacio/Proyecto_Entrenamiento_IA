import React, { useEffect, useState } from 'react';

interface TypewriterTextProps {
  text: string;
  speedMs?: number;
}

export const TypewriterText: React.FC<TypewriterTextProps> = ({ text, speedMs = 55 }) => {
  const [visibleCount, setVisibleCount] = useState(0);
  const words = text.split(' ');

  useEffect(() => {
    setVisibleCount(0);
    if (!text) return;

    const interval = setInterval(() => {
      setVisibleCount((prev) => {
        if (prev >= words.length) {
          clearInterval(interval);
          return prev;
        }
        return prev + 1;
      });
    }, speedMs);

    return () => clearInterval(interval);
  }, [text, speedMs, words.length]);

  return (
    <span>
      {words.slice(0, visibleCount).join(' ')}
      {visibleCount < words.length && (
        <span className="inline-block w-1.5 h-4 ml-1 align-middle bg-[#0A3323] animate-pulse" />
      )}
    </span>
  );
};
