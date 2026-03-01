import React, { useState, useRef } from "react";
import { FiSend, FiMic, FiMicOff } from "react-icons/fi";
import VoiceRecorder from "./VoiceRecorder";
import "./InputBox.css";

export default function InputBox({ onSendText, onSendVoice, disabled }) {
    const [text, setText] = useState("");
    const [voiceMode, setVoiceMode] = useState(false);

    const handleSubmit = (e) => {
        e.preventDefault();
        const trimmed = text.trim();
        if (!trimmed || disabled) return;
        onSendText(trimmed);
        setText("");
    };

    const handleKeyDown = (e) => {
        if (e.key === "Enter" && !e.shiftKey) handleSubmit(e);
    };

    return (
        <div className="input-bar">
            {voiceMode ? (
                <VoiceRecorder
                    onRecorded={(blob) => { onSendVoice(blob); setVoiceMode(false); }}
                    onCancel={() => setVoiceMode(false)}
                    disabled={disabled}
                />
            ) : (
                <form className="input-form" onSubmit={handleSubmit}>
                    <textarea
                        className="input-text"
                        rows={1}
                        placeholder="Ask about a medicine…"
                        value={text}
                        onChange={(e) => setText(e.target.value)}
                        onKeyDown={handleKeyDown}
                        disabled={disabled}
                    />
                    <span className="char-count">{text.length}/500</span>
                    <button
                        type="button"
                        className="btn-icon"
                        onClick={() => setVoiceMode(true)}
                        disabled={disabled}
                        title="Switch to voice input"
                    >
                        <FiMic />
                    </button>
                    <button type="submit" className="btn-send" disabled={disabled || !text.trim()}>
                        <FiSend /> Send
                    </button>
                </form>
            )}
        </div>
    );
}
