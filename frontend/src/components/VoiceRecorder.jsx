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
  const transcriptRef = useRef('')

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

  const startRecording = async () => {
    try {
      setPermissionState('requesting')
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const recorder = new MediaRecorder(stream)

      streamRef.current = stream
      mediaRecorderRef.current = recorder
      transcriptRef.current = ''
      recorder.start()
      startVisualizer(stream)
      startSpeechRecognition()
      setPermissionState('granted')
      setIsRecording(true)
      setStatus('Listening... release or tap again to send.')
    } catch (error) {
      setPermissionState('denied')
      setStatus('Microphone permission is required to record voice.')
    }
  }

  const stopRecording = () => {
    mediaRecorderRef.current?.stop()
    recognitionRef.current?.stop?.()
    streamRef.current?.getTracks().forEach((track) => track.stop())
    analyserRef.current?.context?.close()
    cancelAnimationFrame(animationRef.current)

    setIsRecording(false)
    setLevels(Array.from({ length: 14 }, () => 0.25))
    setStatus('Processing your question...')

    const transcript =
      transcriptRef.current ||
      'Summarize the latest revenue update and show the sources behind the answer.'

    window.setTimeout(() => {
      onComplete(transcript)
      setStatus('Tap the mic and ask your next question.')
    }, 480)
  }

  const toggleRecording = () => {
    if (isRecording) {
      stopRecording()
      return
    }

    startRecording()
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
          <span>{isRecording ? 'Recording' : 'Hold / Tap to Speak'}</span>
        </button>
        <p className={`recorder-status ${permissionState}`}>{status}</p>
      </div>
    </section>
  )
}

export default VoiceRecorder
