import { useEffect, useRef, useState } from 'react'
import AudioVisualizer from './AudioVisualizer.jsx'

const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition

const BAR_COUNT = 28
const IDLE_LEVELS = Array.from({ length: BAR_COUNT }, () => 0)

function VoiceRecorder({ compact = false, onComplete }) {
  const [isRecording, setIsRecording] = useState(false)
  const [permissionState, setPermissionState] = useState('idle')
  const [status, setStatus] = useState('Tap the mic and ask your question.')
  const [levels, setLevels] = useState(IDLE_LEVELS)
  const mediaRecorderRef = useRef(null)
  const recognitionRef = useRef(null)
  const streamRef = useRef(null)
  const analyserRef = useRef(null)
  const animationRef = useRef(null)
  const audioChunksRef = useRef([])
  // Holds the live Web Speech transcript. This ref was missing entirely, so
  // assigning to it threw a ReferenceError inside startRecording(); the catch
  // block then reported it as a denied microphone even though getUserMedia had
  // already succeeded.
  const transcriptRef = useRef('')

  useEffect(() => {
    return () => {
      cancelAnimationFrame(animationRef.current)
      streamRef.current?.getTracks().forEach((track) => track.stop())
      recognitionRef.current?.abort?.()
    }
  }, [])

  const startVisualizer = async (stream) => {
    const AudioContextConstructor = window.AudioContext || window.webkitAudioContext
    const context = new AudioContextConstructor()

    // Chrome creates the context suspended until a user gesture is observed;
    // without this the analyser only ever reports silence.
    if (context.state === 'suspended') {
      await context.resume().catch(() => {})
    }

    const analyser = context.createAnalyser()
    // fftSize must be set BEFORE allocating the buffers: frequencyBinCount is
    // derived from it, and the previous code sized the array against the
    // default 2048 and then shrank the FFT, leaving most bins permanently zero.
    analyser.fftSize = 1024
    analyser.smoothingTimeConstant = 0.7

    const source = context.createMediaStreamSource(stream)
    source.connect(analyser)

    const freqData = new Uint8Array(analyser.frequencyBinCount)
    const timeData = new Uint8Array(analyser.fftSize)
    analyserRef.current = { analyser, context, source }

    // Speech energy sits low in the spectrum, so sampling the whole range
    // leaves most bars flat. Use the lowest ~40% of bins and spread them.
    const usableBins = Math.floor(analyser.frequencyBinCount * 0.4)
    const binsPerBar = Math.max(1, Math.floor(usableBins / BAR_COUNT))
    const smoothed = new Array(BAR_COUNT).fill(0)

    const tick = () => {
      analyser.getByteFrequencyData(freqData)
      analyser.getByteTimeDomainData(timeData)

      // Overall loudness (RMS around the 128 midpoint) drives a gain factor so
      // quiet speech still visibly moves the bars.
      let sumSquares = 0
      for (let i = 0; i < timeData.length; i += 1) {
        const centered = (timeData[i] - 128) / 128
        sumSquares += centered * centered
      }
      const rms = Math.sqrt(sumSquares / timeData.length)
      const loudness = Math.min(1, rms * 6)

      const next = new Array(BAR_COUNT)
      for (let bar = 0; bar < BAR_COUNT; bar += 1) {
        let sum = 0
        const start = bar * binsPerBar
        for (let j = 0; j < binsPerBar; j += 1) {
          sum += freqData[start + j] || 0
        }
        const magnitude = sum / binsPerBar / 255
        // Blend spectral shape with overall loudness, then ease so the meter
        // rises quickly and falls back smoothly instead of flickering.
        const target = Math.min(1, magnitude * 1.6 * (0.35 + loudness))
        smoothed[bar] = smoothed[bar] * 0.6 + target * 0.4
        next[bar] = smoothed[bar]
      }

      setLevels(next)
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
    // Only used for the provisional bubble shown while the request is in
    // flight; Sarvam STT on the backend produces the authoritative transcript.
    recognition.lang = 'hi-IN'

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

      // Recording is already live at this point. Visualiser and speech
      // recognition are cosmetic, so a failure in either must not be reported
      // as a microphone problem or abort the recording.
      startVisualizer(stream).catch((err) => {
        console.warn('Audio visualiser unavailable:', err)
      })
      try {
        startSpeechRecognition()
      } catch (err) {
        console.warn('Live speech recognition unavailable:', err)
      }

      setPermissionState('granted')
      setIsRecording(true)
      setStatus('Listening... tap again to send.')
    } catch (error) {
      // Only a genuine permission/device fault should surface as one. Anything
      // else is a bug in our own code and must say so rather than sending the
      // user to fiddle with browser settings that are already correct.
      const name = error?.name
      let msg
      if (name === 'NotAllowedError' || name === 'SecurityError') {
        setPermissionState('denied')
        msg = 'Microphone permission blocked. Click the lock icon in the address bar to allow, or upload audio below.'
      } else if (name === 'NotFoundError' || name === 'DevicesNotFoundError') {
        setPermissionState('denied')
        msg = 'No microphone was found. Connect one or upload an audio file below.'
      } else if (error?.message?.includes('secure')) {
        setPermissionState('denied')
        msg = error.message
      } else {
        setPermissionState('error')
        msg = `Could not start recording: ${error?.message || name || 'unknown error'}`
      }
      console.error('startRecording failed:', error)
      setStatus(msg)
      streamRef.current?.getTracks().forEach((track) => track.stop())
    }
  }

  const stopRecording = () => {
    const recorder = mediaRecorderRef.current
    recognitionRef.current?.stop?.()
    cancelAnimationFrame(animationRef.current)
    streamRef.current?.getTracks().forEach((track) => track.stop())
    analyserRef.current?.source?.disconnect?.()
    analyserRef.current?.context?.close?.().catch?.(() => {})
    analyserRef.current = null

    setIsRecording(false)
    setLevels(IDLE_LEVELS)
    setStatus('Transcribing and searching the knowledge base...')

    const finish = () => {
      const mimeType = recorder?.mimeType || 'audio/webm'
      const audioBlob =
        audioChunksRef.current.length > 0
          ? new Blob(audioChunksRef.current, { type: mimeType })
          : null

      if (!audioBlob) {
        setStatus('No audio was captured. Try again, or upload a file below.')
        return
      }

      // The backend transcribes with Sarvam STT and returns the authoritative
      // transcript, so never substitute a canned question here: doing so sent a
      // query the user never asked. The browser's live transcript is only a
      // placeholder shown until the real one arrives.
      const provisional = transcriptRef.current?.trim() || '🎤 Voice question'
      onComplete(provisional, audioBlob)
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
            placeholder="Or type a question (e.g. कॉर्पोरेशन क्या है?)..."
            value={customText}
            onChange={(e) => setCustomText(e.target.value)}
          />
          <button type="submit" className="query-send-button" disabled={!customText.trim()}>
            <span className="material-icons" aria-hidden="true">
              send
            </span>
          </button>
        </form>

        {/* Samples must be answerable from the indexed corpus. These previously
            asked about Goa, which the MSMARCO-XI index does not cover, so every
            suggested question produced a refusal and looked like a broken app. */}
        <div className="sample-queries-strip">
          <span>Quick Samples:</span>
          <button
            type="button"
            className="sample-pill"
            onClick={() => handleSampleClick('कॉर्पोरेशन क्या है?')}
          >
            "कॉर्पोरेशन क्या है?"
          </button>
          <button
            type="button"
            className="sample-pill"
            onClick={() => handleSampleClick('ईमानदारी या सच्चाई की परिभाषा')}
          >
            "ईमानदारी की परिभाषा"
          </button>
          <button
            type="button"
            className="sample-pill"
            onClick={() => handleSampleClick('फ्रैंक गिफोर्ड ने कितनी महिलाओं से शादी की')}
          >
            "फ्रैंक गिफोर्ड"
          </button>
        </div>
      </div>
    </section>
  )
}

export default VoiceRecorder
