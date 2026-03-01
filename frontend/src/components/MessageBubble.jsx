import React from "react";
import ReactMarkdown from "react-markdown";
import "./MessageBubble.css";

export default function MessageBubble({ message }) {
    const isBot = message.role === "bot";
    const time = new Date(message.ts).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

    return (
        <div className={`bubble-row ${isBot ? "bot-row" : "user-row"}`}>
            {isBot && <div className="avatar">💊</div>}
            <div className={`bubble ${isBot ? "bubble-bot" : "bubble-user"}`}>
                {isBot ? <ReactMarkdown>{message.text}</ReactMarkdown> : <p>{message.text}</p>}
                <span className="timestamp">{time}</span>
            </div>
        </div>
    );
}
