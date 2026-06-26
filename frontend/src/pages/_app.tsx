import type { AppProps } from "next/app";
import "@/styles/globals.css";

export default function NeuroSightApp({ Component, pageProps }: AppProps) {
  return <Component {...pageProps} />;
}

