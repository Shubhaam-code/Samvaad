const navItems = ['Platform', 'How it Works', 'Use Cases', 'Technology']
const partnerTypes = ['ENTERPRISE', 'RESEARCH', 'FINTECH', 'GOVERNMENT']

const useCases = [
  ['Enterprise', 'Secure, internal knowledge retrieval for teams.'],
  ['Research', 'Synthesize thousands of papers through dialogue.'],
  ['Support', 'The next generation of grounded customer care.'],
  ['Education', 'Interactive learning agents based on curriculum.'],
]

const footerLinks = {
  Platform: ['Technology', 'Security', 'Pricing', 'API Docs'],
  Company: ['About', 'Careers', 'Contact', 'Press Kit'],
}

function Waveform({ large = false }) {
  return (
    <div className={large ? 'waveform waveform-large' : 'waveform'} aria-hidden="true">
      {Array.from({ length: large ? 4 : 12 }).map((_, index) => (
        <span
          className="waveform-bar"
          key={index}
          style={{ animationDelay: `${(index % 6) * 0.1}s` }}
        />
      ))}
    </div>
  )
}

function App() {
  return (
    <>
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
          <a className="button button-dark button-compact" href="#start">
            Try Samvaad
          </a>
        </div>
      </nav>

      <main id="top">
        <section className="hero section-fade">
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

        <section className="intro" id="platform">
          <div className="intro-copy">
            <h2>What if your knowledge could talk back?</h2>
            <p>
              Traditional search gives you links. Samvaad gives you answers. By combining
              state-of-the-art Voice AI with grounded Retrieval Augmented Generation, we have
              created a bridge between your data and your voice.
            </p>
          </div>
          <div className="precision-panel" aria-label="Samvaad precision promise">
            <span className="material-icons sparkle-main" aria-hidden="true">
              auto_awesome
            </span>
            <p>Human-like nuance.</p>
            <p>Machine-like precision.</p>
          </div>
        </section>

        <section className="conversation" id="conversation">
          <div className="section-heading">
            <h2>Ask. Listen. Follow up.</h2>
            <p>A seamless, distraction-free voice interface.</p>
          </div>

          <article className="session-card" aria-label="Example Samvaad session">
            <header>
              <div className="window-dots" aria-hidden="true">
                <span />
                <span />
              </div>
              <p>Active Session</p>
              <span aria-hidden="true" />
            </header>
            <div className="session-body">
              <div className="message">
                <span className="avatar">
                  <span className="material-icons" aria-hidden="true">
                    person
                  </span>
                </span>
                <p>"What was our Q3 revenue growth and the main drivers behind it?"</p>
              </div>
              <div className="message answer">
                <span className="avatar voice">
                  <span className="material-icons" aria-hidden="true">
                    graphic_eq
                  </span>
                </span>
                <div>
                  <p>
                    "Revenue grew by 24% year-over-year. The primary drivers were the
                    expansion into the EMEA market and a 15% increase in retention from
                    enterprise clients."
                  </p>
                  <span className="source-pill">
                    <span className="material-icons" aria-hidden="true">
                      verified
                    </span>
                    Grounded in 4 sources
                  </span>
                </div>
              </div>
              <div className="progress-track" aria-hidden="true">
                <span />
              </div>
            </div>
          </article>
        </section>

        <section className="sources" id="how-it-works">
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
          {useCases.map(([title, body]) => (
            <article key={title}>
              <h3>{title}</h3>
              <p>{body}</p>
            </article>
          ))}
        </section>

        <section className="cta" id="start">
          <div className="cta-wave">
            <Waveform large />
          </div>
          <div className="cta-content">
            <h2>Let your knowledge speak.</h2>
            <a className="button button-dark" href="#conversation">
              Start Now
            </a>
          </div>
        </section>
      </main>

      <footer className="site-footer" id="technology">
        <div className="footer-grid">
          <div>
            <a className="brand footer-brand" href="#top">
              samvaad
            </a>
            <p>
              Built for India, designed for the world. The leading sovereign AI voice RAG
              platform.
            </p>
          </div>
          {Object.entries(footerLinks).map(([heading, links]) => (
            <div className="footer-links" key={heading}>
              <h2>{heading}</h2>
              {links.map((link) => (
                <a href="#" key={link}>
                  {link}
                </a>
              ))}
            </div>
          ))}
          <div className="footer-links">
            <h2>Follow</h2>
            <div className="social-row">
              <a href="#" aria-label="Samvaad website">
                <span className="material-icons" aria-hidden="true">
                  public
                </span>
              </a>
              <a href="#" aria-label="Email Samvaad">
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
            <a href="#">Privacy Policy</a>
            <a href="#">Terms of Service</a>
          </div>
        </div>
      </footer>
    </>
  )
}

export default App
