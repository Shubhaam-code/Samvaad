import { useState } from 'react'

function App() {
  const [count, setCount] = useState(0)

  return (
    <div style={{ fontFamily: 'system-ui, sans-serif', padding: '2rem' }}>
      <h1>HH Goa RAG — Frontend Skeleton</h1>
      <p>Phase 1 placeholder UI. Voice, RAG, and API integration come in later phases.</p>
      <button onClick={() => setCount((c) => c + 1)}>
        Clicks: {count}
      </button>
    </div>
  )
}

export default App
