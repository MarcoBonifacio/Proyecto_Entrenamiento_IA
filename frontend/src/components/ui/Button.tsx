import React from 'react';
import { Loader2 } from 'lucide-react';

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary';
  isLoading?: boolean;
}

export const Button: React.FC<ButtonProps> = ({
  children,
  variant = 'primary',
  isLoading = false,
  disabled,
  className = '',
  ...props
}) => {
  const baseStyles = "inline-flex items-center justify-center px-6 py-3 rounded-xl font-medium transition-all duration-200 focus:outline-none focus:ring-4 focus:ring-[#839958]/50 disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer active:scale-95";
  
  const variants = {
    primary: "bg-[#0A3323] text-white hover:bg-[#072519] shadow-md",
    secondary: "bg-white/80 text-[#0A3323] border border-[#839958]/30 hover:bg-white shadow-sm"
  };

  return (
    <button
      disabled={disabled || isLoading}
      className={`${baseStyles} ${variants[variant]} ${className}`}
      {...props}
    >
      {isLoading ? (
        <>
          <Loader2 className="w-5 h-5 mr-2 animate-spin" />
          <span>Procesando...</span>
        </>
      ) : children}
    </button>
  );
};
