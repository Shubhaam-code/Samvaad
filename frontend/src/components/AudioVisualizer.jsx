const bars = Array.from({ length: 28 })

function AudioVisualizer({ active = false, levels = [] }) {
  return (
    <div className={active ? 'audio-visualizer active' : 'audio-visualizer'} aria-hidden="true">
      {bars.map((_, index) => {
        const level = levels[index % Math.max(levels.length, 1)] || 0.18
        return (
          <span
            className="audio-bar"
            key={index}
            style={{
              '--bar-height': `${Math.max(12, Math.round(level * 76))}px`,
              '--bar-delay': `${index * 38}ms`,
            }}
          />
        )
      })}
    </div>
  )
}

export default AudioVisualizer
