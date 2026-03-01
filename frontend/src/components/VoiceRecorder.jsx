import React, { useState, useRef } from "react";
import { FiMic, FiSquare, FiSend, FiX } from "react-icons/fi";
import "./VoiceRecorder.css";

export default function VoiceRecorder({ onRecorded, onCancel, disabled }) {
    const [recording, setRecording] = useState(false);
    const [audioURL, setAudioURL] = useState(null);
    const [audioBlob, setAudioBlob] = useState(null);
    const mediaRecorder = useRef(null);
    const chunks = useRef([]);

    const startRecording = async () => {
        chunks.current = [];
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder.current = new MediaRecorder(stream);
            mediaRecorder.current.ondataavailable = (e) => chunks.current.push(e.data);
            mediaRecorder.current.onstop = () => {
                const blob = new Blob(chunks.current, { type: "audio/webm" });
                setAudioBlob(blob);
                setAudioURL(URL.createObjectURL(blob));
                stream.getTracks().forEach((t) => t.stop());
            };
            mediaRecorder.current.start();
            setRecording(true);
        } catch {
            alert("Microphone access denied.");
        }
    };

    const stopRecording = () => {
        mediaRecorder.current?.stop();
        setRecording(false);
    };

    const handleSend = () => {
        if (audioBlob) onRecorded(audioBlob);
    };

    const handleCancel = () => {
        setAudioURL(null);
        setAudioBlob(null);
        onCancel();
    };

    return (
        <div className="voice-recorder">
            {!recording && !audioURL && (
                <button className="btn-record" onClick={startRecording} disabled={disabled}>
                    <FiMic /> Start Recording
                </button>
            )}
            {recording && (
                <button className="btn-stop" onClick={stopRecording}>
                    <FiSquare /> Stop
                    <span className="pulse-dot" />
                </button>
            )}
            {audioURL && (
                <div className="preview-row">
                    <audio controls src={audioURL} className="audio-preview" />
                    <button className="btn-send-voice" onClick={handleSend}><FiSend /> Send</button>
                    <button className="btn-cancel" onClick={handleCancel}><FiX /></button>
                </div>
            )}
        </div>
    );
}
