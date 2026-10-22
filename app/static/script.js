
"use strict";

// Get an HTML element using its ID.
const $ = (id) => document.getElementById(id);


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

function setMessage(message, type = "") {
    const element = $("chat-message");

    element.textContent = message;
    element.className = `message ${type}`;
}


function setBusy(busy) {
    const button = $("ask-button");

    button.disabled = busy;
    button.textContent = busy
        ? "Generating answer..."
        : "Ask Research Assistant";
}


function makeElement(tag, className = "", text = "") {
    const element = document.createElement(tag);

    element.className = className;
    element.textContent = text;

    return element;
}


// -------------------------------------------
// DATABASE STATUS
// -------------------------------------------

async function checkDatabase() {
    const indicator = $("database-indicator");

    try {
        const data = await apiRequest("/database/status");

        if (!data.connected) {
            throw new Error("Database disconnected");
        }

        indicator.textContent = "MongoDB Connected";
        indicator.className = "status-pill connected";

        $("total-chunks-stat").textContent = data.total_chunks;

        if (!data.vector_index_ready) {
            indicator.textContent = "Vector Index Not Ready";
            indicator.className = "status-pill disconnected";
        }

    } catch (error) {
        indicator.textContent = "MongoDB Disconnected";
        indicator.className = "status-pill disconnected";

        $("total-chunks-stat").textContent = "--";

        console.error("Database status error:", error);
    }
}


// -------------------------------------------
// CREATE A SOURCE CARD
// -------------------------------------------

function createSourceCard(source, index) {
    const card = makeElement("article", "source-card");

    const title = makeElement(
        "h4",
        "",
        `Source ${index + 1}: ${source.source || "Unknown Paper"}`
    );

    const meta = makeElement("div", "source-meta");

    const details = [
        `Page: ${source.page_number ?? "N/A"}`,
        `Section: ${source.section || "General"}`
    ];

    if (source.similarity_score != null) {
        details.push(
            `Similarity: ${Number(source.similarity_score).toFixed(4)}`
        );
    }

    details.forEach((detail) => {
        meta.appendChild(makeElement("span", "", detail));
    });

    // The text is inserted safely as text, not interpreted as HTML.
    const passage = makeElement(
        "p",
        "",
        source.text || "No passage preview available."
    );

    card.append(title, meta, passage);

    return card;
}


// -------------------------------------------
// DISPLAY THE GENERATED ANSWER
// -------------------------------------------

function displayAnswer(data) {
    const answerSection = $("answer-section");
    const answerContent = $("answer-content");
    const sourcesList = $("sources-list");

    answerContent.textContent =
        data.answer || "No answer was generated.";

    sourcesList.replaceChildren();

    const sources = Array.isArray(data.sources)
        ? data.sources
        : [];

    if (sources.length === 0) {
        sourcesList.appendChild(
            makeElement(
                "p",
                "muted",
                "No supporting sources were returned."
            )
        );
    } else {
        sources.forEach((source, index) => {
            sourcesList.appendChild(
                createSourceCard(source, index)
            );
        });
    }

    $("model-info").textContent =
        `Generated using: ${data.model || "Unknown model"}`;

    answerSection.hidden = false;
}


// -------------------------------------------
// ASK THE RESEARCH ASSISTANT
// -------------------------------------------

async function askQuestion(event) {
    event.preventDefault();

    const question = $("research-question").value.trim();
    const topK = Number($("top-k").value);

    if (!question) {
        setMessage("Please enter a research question.", "error");
        return;
    }

    // Hide the previous answer while processing a new question.
    $("answer-section").hidden = true;

    setBusy(true);

    setMessage(
        "Retrieving relevant research passages and generating an answer. Please wait..."
    );

    try {
        const data = await apiRequest("/ask", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                question: question,
                top_k: topK
            })
        });

        displayAnswer(data);

        setMessage(
            "Answer generated successfully.",
            "success"
        );

    } catch (error) {
        setMessage(error.message, "error");

        console.error("RAG request failed:", error);

    } finally {
        setBusy(false);
    }
}


// -------------------------------------------
// INITIALIZATION
// -------------------------------------------

function initializeDashboard() {
    $("chat-form").addEventListener("submit", askQuestion);

    checkDatabase();
}


if (document.readyState === "loading") {
    document.addEventListener(
        "DOMContentLoaded",
        initializeDashboard
    );
} else {
    initializeDashboard();
}
