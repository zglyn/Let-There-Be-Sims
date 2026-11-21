/**
 * Top Navigation Bar Component (`TopBar`)
 *
 * Implements the strict 1-row, 3-zone laboratory navigation bar contract:
 * - Zone 1 (Left): Single-line brand wordmark (`Sim2Real Compiler`) in Chakra Petch.
 * - Zone 2 (Center): 5 concise single-line text links switching contextual
 *   workbench views (`Command Center`, `Mobile Capture`, `Policy Diff`,
 *   `Cloud Swarm`, `Bundle`).
 * - Zone 3 (Right): Single primary CTA (`New Session`) to reset and mint a
 *   fresh QR-pairable session.
 *
 * Use cases:
 * - Seamless switching between the primary 3D Command Center and specialized
 *   inspectors without unmounting the active session SSE connection.
 */

import React from 'react';
import { APP_CONFIG } from '../config/appConfig';
import { logFunctionCall } from '../utils/logger';

export type ActiveWorkspaceView =
  | 'command_center'
  | 'mobile_capture'
  | 'policy_diff'
  | 'cloud_swarm'
  | 'bundle';

interface TopBarProps {
  activeView: ActiveWorkspaceView;
  onSelectView: (view: ActiveWorkspaceView) => void;
  onCreateNewSession: () => void;
  isCreatingSession: boolean;
}

const NAV_ITEMS: Array<{ id: ActiveWorkspaceView; label: string }> = [
  { id: 'command_center', label: 'Command Center' },
  { id: 'mobile_capture', label: 'Mobile Capture' },
  { id: 'policy_diff', label: 'Policy Diff' },
  { id: 'cloud_swarm', label: 'Cloud Swarm' },
  { id: 'bundle', label: 'Bundle' },
];

/**
 * Renders the 3-zone top bar with strict single-line controls and zero pills.
 */
export const TopBar: React.FC<TopBarProps> = ({
  activeView,
  onSelectView,
  onCreateNewSession,
  isCreatingSession,
}) => {
  return (
    <header className="sticky top-0 z-30 h-14 bg-[#07090E] border-b border-[#1E293B] px-6 flex items-center justify-between gap-8">
      {/* Zone 1: Single-line Brand Wordmark */}
      <button
        type="button"
        onClick={() => {
          logFunctionCall('TopBar.selectBrandHome');
          onSelectView('command_center');
        }}
        className="font-display text-lg font-bold tracking-tight text-slate-100 hover:text-amber-400 transition-colors whitespace-nowrap shrink-0 cursor-pointer"
      >
        {APP_CONFIG.appName}
      </button>

      {/* Zone 2: 5 Concise Single-Line Navigation Links */}
      <nav
        aria-label="Primary workbench views"
        className="flex items-center gap-6 overflow-x-auto"
      >
        {NAV_ITEMS.map((item) => {
          const isActive = activeView === item.id;
          return (
            <button
              key={item.id}
              type="button"
              onClick={() => {
                logFunctionCall('TopBar.selectView', { view: item.id });
                onSelectView(item.id);
              }}
              className={`relative py-4 text-sm font-medium transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                isActive
                  ? 'text-amber-400'
                  : 'text-slate-400 hover:text-slate-100'
              }`}
            >
              {item.label}
              {isActive && (
                <span
                  aria-hidden="true"
                  className="absolute bottom-0 left-0 right-0 h-0.5 bg-amber-400"
                />
              )}
            </button>
          );
        })}
      </nav>

      {/* Zone 3: 1 Primary Action */}
      <div className="shrink-0">
        <button
          type="button"
          onClick={() => {
            logFunctionCall('TopBar.createNewSession');
            onCreateNewSession();
          }}
          disabled={isCreatingSession}
          className="px-4 py-2 text-xs font-semibold text-slate-950 bg-amber-500 rounded-md hover:bg-amber-400 disabled:opacity-50 transition-colors whitespace-nowrap shrink-0 cursor-pointer"
        >
          {isCreatingSession ? 'Creating...' : 'New Session'}
        </button>
      </div>
    </header>
  );
};
