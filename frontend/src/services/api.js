import axios from "axios";

const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:5000/api";

const api = axios.create({ baseURL: BASE_URL });

export const sendTextMessage = async (message, language = "en") => {
    const { data } = await api.post("/chat/text", { message, language });
    return data;
};

export const sendVoiceMessage = async (audioBlob) => {
    const form = new FormData();
    form.append("audio", audioBlob, "recording.webm");
    const { data } = await api.post("/chat/speech", form, {
        headers: { "Content-Type": "multipart/form-data" },
    });
    return data;
};

export const fetchTTS = async (text, lang = "en") => {
    const { data } = await api.post(
        "/chat/tts",
        { text, lang },
        { responseType: "blob" }
    );
    return URL.createObjectURL(data);
};

export const checkHealth = async () => {
    const { data } = await api.get("/health");
    return data;
};

export default api;
