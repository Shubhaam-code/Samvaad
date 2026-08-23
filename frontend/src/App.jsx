import { useMemo, useState } from 'react'
import ChatFeed from './components/ChatFeed.jsx'
import LatencyDashboard from './components/LatencyDashboard.jsx'
import VoiceRecorder from './components/VoiceRecorder.jsx'

const navItems = ['Platform', 'How it Works', 'Use Cases', 'Technology']
const partnerTypes = ['ENTERPRISE', 'RESEARCH', 'FINTECH', 'GOVERNMENT']

const useCases = [
  ['business', 'Enterprise Knowledge', "Talk to your company's knowledge without searching through endless documents."],
  ['science', 'Research', 'Explore complex information naturally through conversation.'],
  ['support_agent', 'Customer Support', 'Give users accurate answers grounded in your trusted knowledge.'],
  ['school', 'Education', 'Turn learning material into something students can actually talk to.'],
]

const pipeline = [
  ['mic', 'Voice'],
  ['psychology', 'Understanding'],
  ['search', 'Retrieval'],
  ['hub', 'Context'],
  ['auto_awesome', 'Grounded Generation'],
  ['volume_up', 'Voice'],
]

// No seeded citations. This previously held an invented "Q3 revenue" example
// that was rendered identically to real retrieval output and was also used as a
// fallback whenever a response carried no citations, so fabricated sources
// appeared alongside genuine ones. Citations must only ever come from the API.
const citations = []

const firstMessages = [
  {
    id: 'welcome',
    role: 'assistant',
    time: 'Now',
    text:
      'Ask a question by voice or text. Answers are grounded strictly in the indexed ' +
      'MSMARCO-XI corpus, and every claim is checked against the passages retrieved ' +
      'for it. If the corpus does not cover your question, I will say so rather than guess.',
    citations: [],
  },
]

function Waveform({ large = false }) {
  return (
    <div className={large ? 'hero-waveform large' : 'hero-waveform'} aria-hidden="true">
      {Array.from({ length: large ? 4 : 12 }).map((_, index) => (
        <span className="hero-wave-bar" key={index} style={{ animationDelay: `${(index % 6) * 0.12}s` }} />
      ))}
    </div>
  )
}

function getTime() {
  return new Intl.DateTimeFormat('en-IN', {
    hour: '2-digit',
    minute: '2-digit',
  }).format(new Date())
}

/**
 * Turn an API error body into something a person can act on.
 *
 * Raw text like "failed with status 422" told the user nothing: 422 is the
 * grounding refusal, which is the system working as intended, and it read as a
 * crash. Each code gets an explanation of what actually happened.
 */
function describeApiError(errBody, status) {
  const detail = errBody?.detail ?? {}
  const code = detail.code

  if (code === 'UNGROUNDED_ANSWER') {
    const reason = detail.reason ? ` ${detail.reason}` : ''
    return (
      'I could not verify that answer against the retrieved passages, so I did not ' +
      `read it aloud.${reason} Try rephrasing, or ask something the indexed corpus covers.`
    )
  }
  if (code === 'LLM_UNAVAILABLE') {
    return 'The language model provider is rate limited or unavailable right now, so no answer could be generated. This is a provider limit, not a problem with your question.'
  }
  if (code === 'NO_SPEECH_DETECTED') {
    return 'No speech was detected in that recording. Speak a little closer to the microphone and try again.'
  }
  if (code === 'QUERY_REJECTED') {
    return `That request was blocked by the input guardrail.${detail.reason ? ` ${detail.reason}` : ''}`
  }
  if (code === 'LLM_PROVIDER_NOT_CONFIGURED' || code === 'TTS_PROVIDER_NOT_CONFIGURED') {
    return 'A required provider is not configured on the server.'
  }
  if (status === 503) {
    return detail.message || 'The knowledge base index is unavailable, so no grounded answer can be produced yet.'
  }
  return detail.message || `Request failed with status ${status}.`
}

/** Short, readable heading for a citation card, taken from the passage text. */
function buildCitationTitle(citation, index) {
  const text = (citation.text || citation.chunk_text || citation.passage || '').trim()
  if (!text) return `Source [${index + 1}]`
  const collapsed = text.replace(/\s+/g, ' ')
  if (collapsed.length <= 72) return collapsed
  // Prefer breaking on a word boundary so the heading does not end mid-word.
  const cut = collapsed.slice(0, 72)
  const lastSpace = cut.lastIndexOf(' ')
  return `${(lastSpace > 40 ? cut.slice(0, lastSpace) : cut).trim()}…`
}

/** Render the retrieval score as a percentage, or a dash when absent. */
function formatScore(score) {
  const value = Number(score)
  if (!Number.isFinite(value)) return '—'
  const pct = value <= 1 ? value * 100 : value
  return `${Math.round(Math.max(0, Math.min(100, pct)))}%`
}

function App() {
  const [messages, setMessages] = useState(firstMessages)
  const [activeCitationId, setActiveCitationId] = useState(1)
  const [theme, setTheme] = useState('light')
  const [latencyTick, setLatencyTick] = useState(0)
  const [isProcessing, setIsProcessing] = useState(false)

  const allAssistantCitations = useMemo(() => {
    return messages
      .filter((m) => m.role === 'assistant' && Array.isArray(m.citations))
      .flatMap((m) => m.citations)
  }, [messages])

  // May be undefined until the first answer arrives; the drawer renders an
  // empty state for that case rather than dereferencing a seeded placeholder.
  const activeCitation = useMemo(() => {
    return (
      allAssistantCitations.find((citation) => citation.id === activeCitationId) ||
      allAssistantCitations[0]
    )
  }, [allAssistantCitations, activeCitationId])

  const handleRecordingComplete = async (transcript, audioBlob = null) => {
    const now = getTime()
    const userMsgId = `user-${Date.now()}`

    setMessages((current) => [
      ...current,
      {
        id: userMsgId,
        role: 'user',
        time: now,
        text: transcript,
      },
    ])

    setIsProcessing(true)

    try {
      let data = null

      if (audioBlob && audioBlob.size > 0) {
        const ext = (audioBlob.type || '').includes('webm')
          ? 'webm'
          : (audioBlob.type || '').includes('ogg')
          ? 'ogg'
          : 'wav'
        const formData = new FormData()
        formData.append('audio', audioBlob, `question.${ext}`)
        const response = await fetch('/api/voice-query', {
          method: 'POST',
          body: formData,
        })
        if (!response.ok) {
          const errBody = await response.json().catch(() => ({}))
          throw new Error(describeApiError(errBody, response.status))
        }
        data = await response.json()
      } else {
        const response = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query: transcript }),
        })
        if (!response.ok) {
          const errBody = await response.json().catch(() => ({}))
          throw new Error(describeApiError(errBody, response.status))
        }
        data = await response.json()
      }

      // Voice turns post a placeholder before the audio is transcribed. Swap in
      // the transcript Sarvam actually produced so the chat shows the question
      // that was really asked and answered.
      if (data.transcribed_text) {
        setMessages((current) =>
          current.map((msg) =>
            msg.id === userMsgId ? { ...msg, text: data.transcribed_text } : msg
          )
        )
      }

      const receivedCitations = (data.citations || []).map((c, idx) => ({
        id: idx + 1,
        // document_id is a content hash, so using it as the heading rendered an
        // unreadable 64-character string. Lead with the passage itself and keep
        // the hash as a short, clearly-truncated reference.
        title: buildCitationTitle(c, idx),
        source: c.document_id
          ? `MSMARCO-XI · doc ${String(c.document_id).slice(0, 8)}`
          : 'MSMARCO-XI Knowledge Base',
        // The API field is `score`; reading `similarity_score` never matched, so
        // every card silently displayed a hardcoded 96%.
        confidence: formatScore(c.score),
        passage: c.text || c.chunk_text || c.passage || '',
      }))

      // Never substitute placeholder citations: an answer with no evidence must
      // show none.
      const finalCitations = receivedCitations

      setMessages((current) => [
        ...current,
        {
          id: `assistant-${Date.now()}`,
          role: 'assistant',
          time: getTime(),
          text: data.answer,
          citations: finalCitations,
          audioBase64: data.audio_base64 || null,
        },
      ])

      setActiveCitationId(finalCitations[0]?.id || 1)

      // Auto-play synthesized voice response if received
      if (data.audio_base64) {
        playAudioBase64(data.audio_base64)
      }
    } catch (err) {
      setMessages((current) => [
        ...current,
        {
          id: `assistant-${Date.now()}`,
          role: 'assistant',
          time: getTime(),
          text: `⚠️ ${err.message || 'Error communicating with backend'}`,
          citations: [],
        },
      ])
    } finally {
      setIsProcessing(false)
      setLatencyTick(Date.now())
    }
  }

  const playAudioBase64 = (base64String) => {
    try {
      const audio = new Audio(`data:audio/wav;base64,${base64String}`)
      audio.play().catch(() => {})
    } catch {
      // Fallback to speech synthesis
    }
  }

  const playAnswer = async (text, audioBase64 = null) => {
    if (audioBase64) {
      playAudioBase64(audioBase64)
      return
    }
    try {
      const isHindi = /[\u0900-\u097F]/.test(text)
      const response = await fetch('/api/tts', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: text.slice(0, 400),
          language: isHindi ? 'hi-IN' : 'en-IN',
          voice: 'anushka',
        }),
      })
      if (response.ok) {
        const ttsData = await response.json()
        if (ttsData.audio_base64) {
          playAudioBase64(ttsData.audio_base64)
          setLatencyTick(Date.now())
          return
        }
      }
    } catch {
      // Fallback to browser SpeechSynthesis API
    }

    window.speechSynthesis.cancel()
    const utterance = new SpeechSynthesisUtterance(text)
    utterance.lang = /[\u0900-\u097F]/.test(text) ? 'hi-IN' : 'en-IN'
    utterance.rate = 0.94
    window.speechSynthesis.speak(utterance)
  }

  return (
    <div className={`site-shell ${theme}`} id="top">
      <nav className="nav-shell" aria-label="Primary navigation">
        <div className="nav-pill">
          <a className="brand" href="#top" aria-label="Samvaad home">
            samvaad
          </a>
          <div className="nav-links">
            {navItems.map((item) => (
              <a href={`#${item.toLowerCase().replaceAll(' ', '-')}`} key={item}>
                {item}
              </a>
            ))}
          </div>
          <div className="nav-actions">
            <button className="theme-button" onClick={() => setTheme(theme === 'light' ? 'dark' : 'light')} type="button">
              <span className="material-icons" aria-hidden="true">
                {theme === 'light' ? 'dark_mode' : 'light_mode'}
              </span>
            </button>
            <a className="button button-dark button-compact" href="#conversation">
              Try Samvaad
            </a>
          </div>
        </div>
      </nav>

      <main>
        <section className="hero gradient-mesh">
          <div className="hero-inner">
            <p className="eyebrow">Voice-First - Grounded AI</p>
            <h1>Knowledge that speaks.</h1>
            <p className="hero-copy">
              Experience your data through natural conversation. A sovereign RAG platform
              designed to turn static information into fluid, voice-driven intelligence.
            </p>
            <div className="hero-actions">
              <a className="button button-dark" href="#conversation">
                Start a Conversation
                <span className="material-icons" aria-hidden="true">
                  arrow_forward
                </span>
              </a>
              <a className="button button-light" href="#platform">
                Explore Samvaad
              </a>
            </div>
            <div className="voice-stage" aria-label="Voice prompt examples">
              <p className="floating-label label-left">"Summarize this document."</p>
              <Waveform />
              <p className="floating-label label-right">"Is there a mention of budget?"</p>
            </div>
          </div>
          <div className="partner-strip">
            <p>Built for knowledge that matters</p>
            <div>
              {partnerTypes.map((type) => (
                <span key={type}>{type}</span>
              ))}
            </div>
          </div>
        </section>

        <section className="intro-section" id="platform">
          <div>
            <h2>What if your knowledge could talk back?</h2>
            <p>
              Traditional search gives you links. Samvaad gives you answers. By combining
              state-of-the-art Voice AI with grounded Retrieval Augmented Generation, we
              have created a bridge between your data and your voice.
            </p>
          </div>
          <div className="precision-card">
            <span className="material-icons" aria-hidden="true">
              auto_awesome
            </span>
            <p>Human-like nuance.</p>
            <p>Machine-like precision.</p>
          </div>
        </section>

        <section className="conversation-section" id="conversation">
          <div className="section-heading">
            <h2>Ask. Listen. Follow up.</h2>
            <p>A seamless, distraction-free voice interface.</p>
          </div>

          <div className="session-shell">
            <div className="session-titlebar">
              <div className="window-dots" aria-hidden="true">
                <span />
                <span />
              </div>
              <p>Active Session</p>
              <span />
            </div>
            <VoiceRecorder compact onComplete={handleRecordingComplete} />
            <div className="session-grid">
              <ChatFeed
                activeCitationId={activeCitationId}
                messages={messages}
                onCitationSelect={setActiveCitationId}
                onPlayAnswer={playAnswer}
              />
              <aside className="grounding-drawer" aria-label="Citation passage drawer">
                <div className="drawer-heading">
                  <p className="eyebrow">Grounding Drawer</p>
                  <strong>{activeCitation ? `Source [${activeCitation.id}]` : 'No source selected'}</strong>
                  <span>{activeCitation?.confidence ?? ''}</span>
                </div>
                {activeCitation ? (
                  <article className="passage-card">
                    <span className="material-icons" aria-hidden="true">
                      article
                    </span>
                    <h3>{activeCitation.title}</h3>
                    <p className="source-name">{activeCitation.source}</p>
                    <p>{activeCitation.passage}</p>
                  </article>
                ) : (
                  <article className="passage-card">
                    <span className="material-icons" aria-hidden="true">
                      article
                    </span>
                    <p>
                      Ask a question to see the retrieved passages that grounded the answer.
                    </p>
                  </article>
                )}
              </aside>
            </div>
          </div>
        </section>

        <section className="sources-section" id="how-it-works">
          <h2>Every answer has a source.</h2>
          <div className="source-flow">
            <article className="source-card">
              <span className="material-icons orange" aria-hidden="true">
                folder
              </span>
              <h3>Knowledge Base</h3>
              <p>PDFs, Docs, Audio, Video</p>
            </article>
            <div className="source-line" aria-hidden="true" />
            <div className="brain-node">
              <span className="material-icons" aria-hidden="true">
                psychology
              </span>
            </div>
            <article className="source-card">
              <span className="material-icons lavender" aria-hidden="true">
                record_voice_over
              </span>
              <h3>Grounded Answer</h3>
              <p>Natural, Auditable Speech</p>
            </article>
          </div>
        </section>

        <section className="use-cases" id="use-cases">
          {useCases.map(([icon, title, body], index) => (
            <article key={title}>
              <span className={`material-icons ${index % 2 === 0 ? 'orange' : 'lavender'}`} aria-hidden="true">
                {icon}
              </span>
              <h3>{title}</h3>
              <p>{body}</p>
            </article>
          ))}
        </section>

        <section className="pipeline-section" id="technology">
          <h2>Built for grounded conversations</h2>
          <div className="pipeline">
            {pipeline.map(([icon, label], index) => (
              <div className="pipeline-item" key={`${icon}-${label}`}>
                <span className={`material-icons ${index % 2 === 0 ? 'orange' : 'lavender'}`} aria-hidden="true">
                  {icon}
                </span>
                <p>{label}</p>
                {index < pipeline.length - 1 && (
                  <span className="material-icons pipeline-arrow" aria-hidden="true">
                    arrow_forward
                  </span>
                )}
              </div>
            ))}
          </div>
        </section>

        <section className="cta-section gradient-mesh">
          <div className="cta-wave">
            <Waveform large />
          </div>
          <div className="cta-content">
            <h2>Let your knowledge speak.</h2>
            <p>Meet Samvaad, a voice-first way to interact with the information that matters.</p>
            <a className="button button-dark" href="#conversation">
              Start Now
            </a>
          </div>
        </section>
      </main>

      <footer className="site-footer">
        <div className="footer-grid">
          <div>
            <a className="brand footer-brand" href="#top">
              samvaad
            </a>
            <p>Built for India, designed for the world. The leading sovereign AI voice RAG platform.</p>
          </div>
          <div>
            <h3>Platform</h3>
            <a href="#technology">Technology</a>
            <a href="#how-it-works">Security</a>
            <a href="#conversation">Pricing</a>
            <a href="#conversation">API Docs</a>
          </div>
          <div>
            <h3>Company</h3>
            <a href="#platform">About</a>
            <a href="#use-cases">Careers</a>
            <a href="#conversation">Contact</a>
            <a href="#top">Press Kit</a>
          </div>
          <div>
            <h3>Follow</h3>
            <div className="social-row">
              <a href="#top" aria-label="Samvaad website">
                <span className="material-icons" aria-hidden="true">
                  public
                </span>
              </a>
              <a href="#top" aria-label="Email Samvaad">
                <span className="material-icons" aria-hidden="true">
                  alternate_email
                </span>
              </a>
            </div>
          </div>
        </div>
        <div className="footer-bottom">
          <p>Copyright 2026 Samvaad AI. All rights reserved.</p>
          <div>
            <a href="#top">Privacy Policy</a>
            <a href="#top">Terms of Service</a>
          </div>
        </div>
      </footer>

      <LatencyDashboard liveSample={latencyTick} />
    </div>
  )
}

export default App
