import { useCallback, useEffect, useMemo, useState } from 'react'

const STAGE_CONFIG = [
  { key: 'stt_ms', label: 'STT', budgetMs: 50, icon: 'mic' },
  { key: 'retrieval_ms', label: 'Retrieval', budgetMs: 30, icon: 'search' },
  { key: 'llm_ms', label: 'LLM', budgetMs: 80, icon: 'psychology' },
  { key: 'tts_ms', label: 'TTS', budgetMs: 50, icon: 'volume_up' },
]

const POLL_MS = 2000
const SLA_MS = 200

function formatMs(value) {
  if (value == null || Number.isNaN(value)) return '—'
  if (value < 1) return `${value.toFixed(2)} ms`
  if (value < 100) return `${value.toFixed(1)} ms`
  return `${Math.round(value)} ms`
}

function gaugeWidth(value, budget) {
  if (!budget || value <= 0) return 0
  return Math.min(100, (value / budget) * 100)
}

function gaugeTone(value, budget) {
  if (value <= 0) return 'idle'
  const ratio = value / budget
  if (ratio <= 0.6) return 'good'
  if (ratio <= 1) return 'warn'
  return 'hot'
}

function LatencyGauge({ label, icon, stats, budgetMs }) {
  const p50 = stats?.p50_ms ?? 0
  const tone = gaugeTone(p50, budgetMs)

  return (
    <div className={`latency-gauge tone-${tone}`}>
      <div className="latency-gauge-head">
        <span className="material-icons" aria-hidden="true">
          {icon}
        </span>
        <span>{label}</span>
        <strong>{formatMs(p50)}</strong>
      </div>
      <div className="latency-gauge-track" aria-hidden="true">
        <span className="latency-gauge-fill" style={{ width: `${gaugeWidth(p50, budgetMs)}%` }} />
        <span className="latency-gauge-budget" style={{ left: '100%' }} />
      </div>
      <div className="latency-gauge-meta">
        <span>P70 {formatMs(stats?.p70_ms)}</span>
        <span>P100 {formatMs(stats?.p100_ms)}</span>
      </div>
    </div>
  )
}

function LatencyDashboard({ liveSample = null }) {
  const [collapsed, setCollapsed] = useState(false)
  const [analytics, setAnalytics] = useState(null)
  const [status, setStatus] = useState('connecting')
  const [lastUpdated, setLastUpdated] = useState(null)

  const fetchAnalytics = useCallback(async () => {
    try {
      const response = await fetch('/api/analytics/latency')
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`)
      }
      const payload = await response.json()
      setAnalytics(payload)
      setStatus('live')
      setLastUpdated(new Date())
    } catch {
      setStatus('offline')
    }
  }, [])

  useEffect(() => {
    fetchAnalytics()
    const timer = window.setInterval(fetchAnalytics, POLL_MS)
    return () => window.clearInterval(timer)
  }, [fetchAnalytics])

  useEffect(() => {
    if (!liveSample) return
    fetchAnalytics()
  }, [liveSample, fetchAnalytics])

  const totalStats = analytics?.total_ms
  const sub200 = analytics?.sub_200ms_achieved === true

  const percentileCards = useMemo(
    () => [
      { label: 'P50', value: totalStats?.p50_ms },
      { label: 'P70', value: totalStats?.p70_ms },
      { label: 'P100', value: totalStats?.p100_ms },
    ],
    [totalStats],
  )

  return (
    <aside
      className={`latency-dashboard ${collapsed ? 'collapsed' : ''} status-${status}`}
      aria-label="Real-time latency analytics"
    >
      <header className="latency-dashboard-header">
        <div className="latency-dashboard-title">
          <span className="material-icons pulse-dot" aria-hidden="true">
            speed
          </span>
          <div>
            <strong>Latency Dashboard</strong>
            <p>
              {status === 'live' && lastUpdated
                ? `Updated ${lastUpdated.toLocaleTimeString()}`
                : status === 'offline'
                  ? 'Backend offline — showing last known metrics'
                  : 'Connecting to analytics…'}
            </p>
          </div>
        </div>
        <div className="latency-dashboard-actions">
          <span className={`sla-badge ${sub200 ? 'achieved' : 'pending'}`}>
            <span className="material-icons" aria-hidden="true">
              {sub200 ? 'verified' : 'hourglass_top'}
            </span>
            {sub200 ? 'Sub-200ms ✓' : 'Sub-200ms'}
          </span>
          <button
            aria-expanded={!collapsed}
            aria-label={collapsed ? 'Expand latency dashboard' : 'Collapse latency dashboard'}
            className="latency-toggle"
            onClick={() => setCollapsed((value) => !value)}
            type="button"
          >
            <span className="material-icons" aria-hidden="true">
              {collapsed ? 'expand_less' : 'expand_more'}
            </span>
          </button>
        </div>
      </header>

      {!collapsed && (
        <div className="latency-dashboard-body">
          <div className="latency-percentiles">
            {percentileCards.map(({ label, value }) => (
              <div className="latency-percentile-card" key={label}>
                <span>{label}</span>
                <strong>{formatMs(value)}</strong>
              </div>
            ))}
            <div className="latency-percentile-card requests">
              <span>Requests</span>
              <strong>{analytics?.request_count ?? 0}</strong>
            </div>
          </div>

          <div className="latency-stage-grid">
            {STAGE_CONFIG.map((stage) => (
              <LatencyGauge
                budgetMs={stage.budgetMs}
                icon={stage.icon}
                key={stage.key}
                label={stage.label}
                stats={analytics?.[stage.key]}
              />
            ))}
          </div>
        </div>
      )}
    </aside>
  )
}

export default LatencyDashboard
