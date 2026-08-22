import { useEffect, useRef, useState } from 'react'
import AudioVisualizer from './AudioVisualizer.jsx'

const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition

function VoiceRecorder({ compact = false, onComplete }) {
  const [isRecording, setIsRecording] = useState(false)
  const [permissionState, setPermissionState] = useState('idle')
  const [status, setStatus] = useState('Tap the mic and ask your question.')
  const [levels, setLevels] = useState(Array.from({ length: 14 }, () => 0.25))
  const mediaRecorderRef = useRef(null)
  const recognitionRef = useRef(null)
  const streamRef = useRef(null)
  const analyserRef = useRef(null)
  const animationRef = useRef(null)
  const audioChunksRef = useRef([])

  useEffect(() => {
    return () => {
      cancelAnimationFrame(animationRef.current)
      streamRef.current?.getTracks().forEach((track) => track.stop())
      recognitionRef.current?.abort?.()
    }
  }, [])

  const startVisualizer = (stream) => {
    const AudioContextConstructor = window.AudioContext || window.webkitAudioContext
    const context = new AudioContextConstructor()
    const analyser = context.createAnalyser()
    const source = context.createMediaStreamSource(stream)
    const data = new Uint8Array(analyser.frequencyBinCount)

    analyser.fftSize = 64
    source.connect(analyser)
    analyserRef.current = { analyser, context, data }

    const tick = () => {
      analyser.getByteFrequencyData(data)
      setLevels(Array.from(data.slice(0, 14), (value) => value / 255))
      animationRef.current = requestAnimationFrame(tick)
    }

    tick()
  }

  const startSpeechRecognition = () => {
    if (!SpeechRecognition) {
      transcriptRef.current = ''
      return
    }

    const recognition = new SpeechRecognition()
    recognition.continuous = true
    recognition.interimResults = true
    recognition.lang = 'en-IN'

    recognition.onresult = (event) => {
      transcriptRef.current = Array.from(event.results)
        .map((result) => result[0].transcript)
        .join(' ')
        .trim()
    }

    recognition.onerror = () => {
      setStatus('Speech recognition paused. Recording is still active.')
    }

    recognition.start()
    recognitionRef.current = recognition
  }

  const fileInputRef = useRef(null)

  const startRecording = async () => {
    try {
      if (!navigator?.mediaDevices?.getUserMedia) {
        throw new Error(
          'Microphone access requires a secure connection (HTTPS or localhost). Use https:// or upload an audio file below.'
        )
      }
      setPermissionState('requesting')
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const recorder = new MediaRecorder(stream)

      audioChunksRef.current = []
      recorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          audioChunksRef.current.push(event.data)
        }
      }

      streamRef.current = stream
      mediaRecorderRef.current = recorder
      transcriptRef.current = ''
      recorder.start(100)
      startVisualizer(stream)
      startSpeechRecognition()
      setPermissionState('granted')
      setIsRecording(true)
      setStatus('Listening... release or tap again to send.')
    } catch (error) {
      setPermissionState('denied')
      const msg = error.message?.includes('secure')
        ? error.message
        : 'Microphone permission blocked. Click 🔒 in address bar to allow or upload audio below.'
      setStatus(msg)
    }
  }

  const stopRecording = () => {
    const recorder = mediaRecorderRef.current
    recognitionRef.current?.stop?.()
    streamRef.current?.getTracks().forEach((track) => track.stop())
    analyserRef.current?.context?.close()
    cancelAnimationFrame(animationRef.current)

    setIsRecording(false)
    setLevels(Array.from({ length: 14 }, () => 0.25))
    setStatus('Processing voice question through Sarvam AI...')

    const finish = () => {
      const mimeType = recorder?.mimeType || 'audio/webm'
      const audioBlob =
        audioChunksRef.current.length > 0
          ? new Blob(audioChunksRef.current, { type: mimeType })
          : null
      const transcript =
        transcriptRef.current ||
        'Summarize the latest knowledge base update and show citations.'

      onComplete(transcript, audioBlob)
      setStatus('Tap the mic and ask your next question.')
    }

    if (recorder && recorder.state !== 'inactive') {
      recorder.onstop = finish
      recorder.stop()
    } else {
      finish()
    }
  }

  const toggleRecording = () => {
    if (isRecording) {
      stopRecording()
      return
    }
    startRecording()
  }

  const handleFileUpload = (e) => {
    const file = e.target.files?.[0]
    if (!file) return
    setStatus(`Uploading audio: ${file.name}...`)
    onComplete(file.name, file)
    e.target.value = ''
  }

  const [customText, setCustomText] = useState('')

  const handleTextSubmit = (e) => {
    e.preventDefault()
    const trimmed = customText.trim()
    if (!trimmed) return
    onComplete(trimmed, null)
    setCustomText('')
    setStatus('Question sent to knowledge base.')
  }

  const handleSampleClick = (sampleQuery) => {
    onComplete(sampleQuery, null)
    setStatus(`Querying: "${sampleQuery}"`)
  }

  return (
    <section className={compact ? 'recorder-panel compact' : 'recorder-panel'} aria-label="Voice recorder">
      <div className="recorder-copy">
        <p className="eyebrow">One-click Voice RAG</p>
        <h1>Ask your knowledge base out loud.</h1>
        <p>
          Record a question, receive a grounded answer, play it back as voice, and inspect
          the exact passages used for citation.
        </p>
      </div>

      <div className="recorder-console">
        <AudioVisualizer active={isRecording} levels={levels} />
        <div style={{ display: 'flex', gap: '10px', alignItems: 'center', justifyContent: 'center', flexWrap: 'wrap' }}>
          <button
            aria-pressed={isRecording}
            className={isRecording ? 'mic-button recording' : 'mic-button'}
            onClick={toggleRecording}
            type="button"
          >
            <span className="pulse-ring" aria-hidden="true" />
            <span className="material-icons" aria-hidden="true">
              {isRecording ? 'stop' : 'mic'}
            </span>
            <span>{isRecording ? 'Recording (Tap to send)' : 'Hold / Tap to Speak'}</span>
          </button>

          <button
            type="button"
            className="upload-audio-button"
            onClick={() => fileInputRef.current?.click()}
            title="Upload audio file (WAV, MP3, WebM, M4A)"
          >
            <span className="material-icons" aria-hidden="true">
              audio_file
            </span>
            <span>Upload Audio</span>
          </button>
        </div>

        <input
          ref={fileInputRef}
          type="file"
          accept="audio/*,.wav,.mp3,.webm,.m4a,.ogg"
          onChange={handleFileUpload}
          style={{ display: 'none' }}
        />

        <p className={`recorder-status ${permissionState}`}>{status}</p>

        <form className="query-input-form" onSubmit={handleTextSubmit}>
          <input
            type="text"
            className="query-text-input"
            placeholder="Or type a question (e.g. गोवा की राजधानी क्या है?)..."
            value={customText}
            onChange={(e) => setCustomText(e.target.value)}
          />
          <button type="submit" className="query-send-button" disabled={!customText.trim()}>
            <span className="material-icons" aria-hidden="true">
              send
            </span>
          </button>
        </form>

        <div className="sample-queries-strip">
          <span>Quick Samples:</span>
          <button
            type="button"
            className="sample-pill"
            onClick={() => handleSampleClick('What is the capital of Goa?')}
          >
            "Capital of Goa?"
          </button>
          <button
            type="button"
            className="sample-pill"
            onClick={() => handleSampleClick('गोवा की राजधानी क्या है?')}
          >
            "गोवा की राजधानी?"
          </button>
          <button
            type="button"
            className="sample-pill"
            onClick={() => handleSampleClick('What are the best beaches in North Goa?')}
          >
            "North Goa Beaches"
          </button>
        </div>
      </div>
    </section>
  )
}

export default VoiceRecorder
