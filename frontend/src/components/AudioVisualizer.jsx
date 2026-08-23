const bars = Array.from({ length: 28 })

function AudioVisualizer({ active = false, levels = [] }) {
  return (
    <div className={active ? 'audio-visualizer active' : 'audio-visualizer'} aria-hidden="true">
      {bars.map((_, index) => {
        const level = levels[index % Math.max(levels.length, 1)] ?? 0
        // While live, drive height purely from the measured level so the meter
        // reflects actual loudness. Idle keeps a small static bar.
        const height = active ? Math.max(6, Math.round(level * 72)) : 10
        return (
          <span
            className="audio-bar"
            key={index}
            style={{
              '--bar-height': `${height}px`,
              '--bar-delay': `${index * 38}ms`,
              // Transition off during capture: the requestAnimationFrame loop
              // already updates every frame, and a CSS transition on top of it
              // lags the meter behind the voice.
              transition: active ? 'none' : undefined,
              opacity: active ? 0.55 + level * 0.45 : undefined,
            }}
          />
        )
      })}
    </div>
  )
}

export default AudioVisualizer
