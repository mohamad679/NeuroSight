import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bgBase: "var(--color-bg)",
        bgSurface: "var(--color-surface)",
        bgElevated: "var(--color-surface-elevated)",
        borderSubtle: "var(--color-border)",
        accent: "var(--color-accent)",
        accentSoft: "var(--color-accent)",
        accentGlow: "var(--color-accent-glow)",
        success: "var(--color-success)",
        warning: "var(--color-warning)",
        danger: "var(--color-danger)",
        textPrimary: "var(--color-text-primary)",
        textSecondary: "var(--color-text-muted)",
        textMuted: "var(--color-text-soft)",
        teal: "var(--color-cyan)",
        sage: "var(--color-sage)"
      },
      fontFamily: {
        sans: ["var(--font-body)", "ui-sans-serif", "system-ui", "sans-serif"],
        display: ["var(--font-display)", "ui-sans-serif", "system-ui", "sans-serif"]
      },
      boxShadow: {
        active: "0 0 0 4px rgba(244,184,96,0.16)",
        luxury: "var(--shadow-soft)"
      },
      borderRadius: {
        luxury: "8px"
      }
    }
  },
  plugins: []
};

export default config;
