
"use strict";

// -------------------------------------------
// ELEMENT HELPERS
// -------------------------------------------

const $ = (id) => document.getElementById(id);

const chatForm = $("chat-form");
const questionInput = $("research-question");
const sendButton = $("ask-button");
const messages = $("messages");
const welcomeScreen = $("welcome-screen");
const chatContainer = $("chat-container");
const newChatButton = $("new-chat-button");
const statusMessage = $("chat-message");

let isGenerating = false;
let activeRequest = null;
let chatVersion = 0;


// -------------------------------------------
// API HELPER
// -------------------------------------------

async function apiRequest(url, options = {}) {
    const response = await fetch(url, options);

    let data;

    try {
        data = await response.json();
    } catch {
        throw new Error(`Server error: HTTP ${response.status}`);
    }

    if (!response.ok) {
        const detail = data.detail;

        throw new Error(
            typeof detail === "string"
                ? detail
                : JSON.stringify(detail ?? data)
        );
    }

    return data;
}


// -------------------------------------------
// UI HELPERS
// -------------------------------------------

function setStatus(message = "", type = "") {
    statusMessage.textContent = message;
    statusMessage.className = `chat-message ${type}`;
}


function setBusy(busy) {
    isGenerating = busy;

    sendButton.disabled = busy;
    questionInput.disabled = busy;

    sendButton.textContent = busy ? "…" : "↑";
    sendButton.setAttribute(
        "aria-label",
        busy ? "Generating answer" : "Send message"
    );

    newChatButton.disabled = false;
}


function scrollToBottom() {
    chatContainer.scrollTop = chatContainer.scrollHeight;
}


function createElement(tag, className = "", text = "") {
    const element = document.createElement(tag);

    element.className = className;
    element.textContent = text;

    return element;
}


// -------------------------------------------
// CREATE CHAT MESSAGE
// -------------------------------------------

function addMessage(role, content, isThinking = false) {
    const row = createElement(
        "div",
        `chat-message-row ${role}`
    );

    const bubble = createElement("div", "message-bubble");

    if (role === "assistant") {
        const label = createElement(
            "div",
            "assistant-label",
            "Research Bot"
        );

        const answer = createElement(
            "div",
            isThinking
                ? "assistant-content thinking"
                : "assistant-content",
            content
        );

        bubble.append(label, answer);

    } else {
        bubble.textContent = content;
    }

    row.appendChild(bubble);
    messages.appendChild(row);

    scrollToBottom();

    return row;
}


// -------------------------------------------
// AUTO-RESIZE MESSAGE INPUT
// -------------------------------------------

function resizeInput() {
    questionInput.style.height = "auto";

    questionInput.style.height =
        Math.min(questionInput.scrollHeight, 170) + "px";
}


// -------------------------------------------
// ASK RESEARCH BOT
// -------------------------------------------

async function askQuestion(question) {
    question = question.trim();

    if (!question || isGenerating) {
        return;
    }

    // Hide welcome screen after the first message.
    welcomeScreen.hidden = true;

    setStatus("");

    // Display the user's question.
    addMessage("user", question);

    // Clear input.
    questionInput.value = "";
    resizeInput();

    // Show loading message inside conversation.
    const thinkingRow = addMessage(
        "assistant",
        "Research Bot is thinking...",
        true
    );

    setBusy(true);

    // Capture the current chat version.
    // This prevents old requests from appearing after New Chat.
    const requestVersion = chatVersion;

    const controller = new AbortController();
    activeRequest = controller;

    try {
        const data = await apiRequest("/ask", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                question: question,
                top_k: 3
            }),
            signal: controller.signal
        });

        if (requestVersion !== chatVersion) {
            return;
        }

        // Display only the generated answer.
        // Retrieved chunks and similarity scores remain internal.
        const answer =
            data.answer || "No answer was generated.";

        const answerContent = thinkingRow.querySelector(
            ".assistant-content"
        );

        answerContent.textContent = answer;
        answerContent.classList.remove("thinking");

        scrollToBottom();

    } catch (error) {
        if (
            error.name === "AbortError" ||
            requestVersion !== chatVersion
        ) {
            return;
        }

        console.error("Research Bot request failed:", error);

        const answerContent = thinkingRow.querySelector(
            ".assistant-content"
        );

        answerContent.textContent =
            "Sorry, I couldn't generate a response. Please try again.";

        answerContent.classList.remove("thinking");

        setStatus(error.message, "error");

    } finally {
        if (requestVersion === chatVersion) {
            activeRequest = null;
            setBusy(false);
            questionInput.focus();
        }
    }
}


// -------------------------------------------
// NEW CHAT
// -------------------------------------------

function startNewChat() {
    // Invalidate any response from the previous chat.
    chatVersion++;

    // Cancel the browser's pending request.
    // This does not guarantee cancellation of backend generation.
    if (activeRequest) {
        activeRequest.abort();
        activeRequest = null;
    }

    messages.replaceChildren();

    welcomeScreen.hidden = false;

    questionInput.value = "";
    resizeInput();

    setStatus("");
    setBusy(false);

    questionInput.focus();
}


// -------------------------------------------
// FORM SUBMISSION
// -------------------------------------------

function handleSubmit(event) {
    event.preventDefault();

    askQuestion(questionInput.value);
}


// -------------------------------------------
// INITIALIZATION
// -------------------------------------------

function initializeChat() {
    chatForm.addEventListener("submit", handleSubmit);

    newChatButton.addEventListener("click", startNewChat);

    // Enter sends the message.
    // Shift + Enter creates a new line.
    questionInput.addEventListener("keydown", (event) => {
        if (
            event.key === "Enter" &&
            !event.shiftKey &&
            !event.isComposing
        ) {
            event.preventDefault();

            askQuestion(questionInput.value);
        }
    });

    questionInput.addEventListener("input", resizeInput);

    // Suggested questions on the welcome screen.
    document.querySelectorAll(".suggestion-card").forEach(
        (button) => {
            button.addEventListener("click", () => {
                askQuestion(button.dataset.question || "");
            });
        }
    );

    resizeInput();
}


if (document.readyState === "loading") {
    document.addEventListener(
        "DOMContentLoaded",
        initializeChat
    );
} else {
    initializeChat();
}
