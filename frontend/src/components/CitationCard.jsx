function CitationCard({ citation, active, onSelect }) {
  return (
    <button
      className={active ? 'citation-card active' : 'citation-card'}
      onClick={() => onSelect(citation.id)}
      type="button"
    >
      <span className="citation-index">[{citation.id}]</span>
      <span>
        <strong>{citation.title}</strong>
        <small>{citation.source}</small>
      </span>
      <span className="material-icons" aria-hidden="true">
        chevron_right
      </span>
    </button>
  )
}

export default CitationCard
