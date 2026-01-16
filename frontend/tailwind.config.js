/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./scr/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        red: {
          600: "#dc2626",
          900: "#7f1d1d",
        },
        yellow: {
          500: "#eab308",
        },
        green: {
          500: "#22c55e",
        },
        zinc: {
          800: "#27272a",
          900: "#18181b",
        },
        gray: {
          400: "#9ca3af",
          500: "#6b7280",
          600: "#4b5563",
        },
        black: "#000000",
      },
    },
  },
  plugins: [],
};
