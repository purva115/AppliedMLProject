import React, { useState, useEffect, useRef } from "react";
import MessageBubble from "./MessageBubble";
import InputBox from "./InputBox";
import { sendTextMessage, sendVoiceMessage } from "../services/api";
import "./ChatInterface.css";

const WELCOME = {
    id: 0,
    role: "bot",
    text: "👋 Hi! I'm **PharmaLLM**, your medicine information assistant. Ask me about any medicine — its uses, composition, dosage, or side effects.",
    ts: new Date().toISOString(),
};

export default function ChatInterface() {
    const [messages, setMessages] = useState([WELCOME]);
    const [loading, setLoading] = useState(false);
    const bottomRef = useRef(null);

    useEffect(() => {
        bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [messages]);

    const addMessage = (role, text) =>
        setMessages((prev) => [
            ...prev,
            { id: Date.now(), role, text, ts: new Date().toISOString() },
        ]);

    const handleTextSend = async (text) => {
        addMessage("user", text);
        setLoading(true);
        try {
            const data = await sendTextMessage(text);
            addMessage("bot", data.response);
        } catch {
            addMessage("bot", "⚠️ Sorry, I couldn't reach the server. Please try again.");
        } finally {
            setLoading(false);
        }
    };

    const handleVoiceSend = async (audioBlob) => {
        addMessage("user", "🎤 *Voice message sent*");
        setLoading(true);
        try {
            const data = await sendVoiceMessage(audioBlob);
            if (data.transcribed) addMessage("user", `📝 "${data.transcribed}"`);
            addMessage("bot", data.response);
        } catch {
            addMessage("bot", "⚠️ Voice processing failed. Please try again.");
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="chat-container">
            <header className="chat-header">
                <div className="header-logo">💊</div>
                <div>
                    <h1>PharmaLLM</h1>
                    <span className="status-dot" /> <small>Online</small>
                </div>
            </header>

            <main className="chat-messages">
                {messages.map((m) => (
                    <MessageBubble key={m.id} message={m} />
                ))}
                {loading && (
                    <div className="typing-indicator">
                        <span /><span /><span />
                    </div>
                )}
                <div ref={bottomRef} />
            </main>

            <InputBox onSendText={handleTextSend} onSendVoice={handleVoiceSend} disabled={loading} />
        </div>
    );
}
