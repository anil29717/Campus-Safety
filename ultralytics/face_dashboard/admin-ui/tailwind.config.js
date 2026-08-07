/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        campus: {
          950: "#0a0f1a",
          900: "#0f1729",
          800: "#1a2332",
          700: "#243044",
          accent: "#3b82f6",
          warn: "#f59e0b",
          danger: "#ef4444",
          ok: "#22c55e",
        },
      },
    },
  },
  plugins: [],
};
