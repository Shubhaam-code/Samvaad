import CitationCard from './CitationCard.jsx'

function ChatFeed({ messages, activeCitationId, onCitationSelect, onPlayAnswer }) {
  return (
    <section className="chat-panel" aria-label="Conversation history">
      <header className="panel-header">
        <div>
          <p className="eyebrow">Live Session</p>
          <h2>Conversation</h2>
        </div>
        <span className="status-pill">
          <span className="status-dot" />
          Grounded
        </span>
      </header>

      <div className="message-list">
        {messages.map((message) => (
          <article className={`chat-message ${message.role}`} key={message.id}>
            <div className="message-avatar">
              <span className="material-icons" aria-hidden="true">
                {message.role === 'user' ? 'person' : 'graphic_eq'}
              </span>
            </div>
            <div className="message-content">
              <div className="message-meta">
                <span>{message.role === 'user' ? 'You' : 'Samvaad'}</span>
                <time>{message.time}</time>
              </div>
              <p>{message.text}</p>
              {message.role === 'assistant' && (
                <div className="answer-actions">
                  <button className="icon-button" type="button" onClick={() => onPlayAnswer(message.text, message.audioBase64)}>
                    <span className="material-icons" aria-hidden="true">
                      volume_up
                    </span>
                    Play voice
                  </button>
                  <div className="citation-badges" aria-label="Citations">
                    {message.citations.map((citation) => (
                      <button
                        className={activeCitationId === citation.id ? 'citation-badge active' : 'citation-badge'}
                        key={citation.id}
                        onClick={() => onCitationSelect(citation.id)}
                        type="button"
                      >
                        [{citation.id}]
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </article>
        ))}
      </div>

      <div className="citation-strip">
        {messages
          .filter((message) => message.role === 'assistant')
          .flatMap((message) => message.citations)
          .map((citation) => (
            <CitationCard
              active={activeCitationId === citation.id}
              citation={citation}
              key={citation.id}
              onSelect={onCitationSelect}
            />
          ))}
      </div>
    </section>
  )
}

export default ChatFeed
