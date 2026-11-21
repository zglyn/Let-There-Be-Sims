/**
 * React Application Bootstrap Entry Point
 *
 * Mounts the root `App` component of the Sim2Real Compiler into `#root` and
 * imports the Precision Lab Dark Tailwind CSS v4 design tokens.
 *
 * Use cases:
 * - Bootstrapping either the Desktop Command Center (`/`) or the QR-paired
 *   Mobile Capture Route (`/capture?session=<id>`).
 */

import { createRoot } from 'react-dom/client';
import App from './App.tsx';
import './index.css';
import { logFunctionCall } from './utils/logger.ts';

logFunctionCall('bootstrapReactApp', { pathname: window.location.pathname });
createRoot(document.getElementById('root')!).render(<App />);
