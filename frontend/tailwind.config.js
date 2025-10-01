/** @type {import('tailwindcss').Config} */
export default {
  content: [
    './index.html',
    './src/**/*.{js,jsx,ts,tsx}'
  ],
  theme: {
    extend: {
      colors: {
        brand: {
          50: '#f0fdf4',
          500: '#16a34a',
          600: '#15803d',
          700: '#166534'
        }
      }
    },
  },
  plugins: [],
}
