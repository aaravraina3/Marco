import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */
  eslint: {
    // Ignore ESLint errors during production builds (Render/Vercel)
    ignoreDuringBuilds: true,
  },
};

export default nextConfig;
