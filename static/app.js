const analysisForm = document.querySelector("#analysis-form");
const reviewInput = document.querySelector("#review-input");
const characterCount = document.querySelector("#character-count");
const submitButton = document.querySelector("#submit-button");
const formMessage = document.querySelector("#form-message");
const resultCard = document.querySelector("#result-card");
const historyRows = document.querySelector("#history-rows");
const historyMessage = document.querySelector("#history-message");
const refreshButton = document.querySelector("#refresh-button");

let selectedEngine = "gemini";
let savedAnalyses = [];

function setMessage(element, message, isError = true) {
  element.textContent = message;
  element.style.color = isError ? "#ec7d89" : "#65c99a";
}

function formatDate(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return "—";
  }

  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    year: "numeric",
  }).format(date);
}

function setCount(id, value) {
  document.querySelector(id).textContent = String(value);
}

function updateDashboard(analyses) {
  const counts = { positive: 0, neutral: 0, negative: 0 };
  for (const item of analyses) {
    const label = item.label?.toLowerCase();
    if (Object.hasOwn(counts, label)) {
      counts[label] += 1;
    }
  }

  const total = analyses.length;
  setCount("#total-count", total);
  setCount("#positive-count", counts.positive);
  setCount("#neutral-count", counts.neutral);
  setCount("#negative-count", counts.negative);
}

function createCell(text, className = "") {
  const cell = document.createElement("td");
  cell.textContent = text;
  if (className) {
    cell.className = className;
  }
  return cell;
}

function renderHistory(analyses) {
  historyRows.replaceChildren();

  if (!analyses.length) {
    const row = document.createElement("tr");
    const cell = createCell("No saved reviews yet. Analyze one to get started.", "empty-state");
    cell.colSpan = 6;
    row.append(cell);
    historyRows.append(row);
    return;
  }

  for (const item of analyses.slice(0, 8)) {
    const row = document.createElement("tr");
    row.append(createCell(item.review, "review-cell"));

    const sentimentCell = document.createElement("td");
    const badge = document.createElement("span");
    const label = item.label?.toLowerCase() ?? "neutral";
    badge.className = `sentiment-chip ${label}`;
    badge.textContent = label;
    sentimentCell.append(badge);
    row.append(sentimentCell);

    row.append(createCell(String(item.score), ""));
    row.append(createCell(item.theme, "theme-cell"));
    row.append(createCell(item.source));
    row.append(createCell(formatDate(item.created_at)));
    historyRows.append(row);
  }
}

async function loadAnalyses() {
  refreshButton.disabled = true;
  historyMessage.textContent = "";

  try {
    const response = await fetch("/analyses");
    if (!response.ok) {
      throw new Error(`Could not load saved reviews (${response.status}).`);
    }

    const data = await response.json();
    savedAnalyses = data.analyses;
    updateDashboard(savedAnalyses);
    renderHistory(savedAnalyses);
  } catch (error) {
    historyRows.replaceChildren();
    const row = document.createElement("tr");
    const cell = createCell("Saved reviews could not be loaded.", "empty-state");
    cell.colSpan = 6;
    row.append(cell);
    historyRows.append(row);
    setMessage(historyMessage, error.message);
  } finally {
    refreshButton.disabled = false;
  }
}

function showResult(result, engine) {
  const label = result.label.toLowerCase();
  document.querySelector("#result-label").textContent = label;
  document.querySelector("#result-badge").textContent = label;
  document.querySelector("#result-badge").className = `sentiment-badge ${label}`;
  document.querySelector("#result-score").textContent = String(result.score);
  document.querySelector("#result-theme").textContent = result.theme;
  document.querySelector("#result-source").textContent =
    engine === "gemini" ? "Gemini AI" : "Python rules";
  resultCard.hidden = false;
}

reviewInput.addEventListener("input", () => {
  characterCount.textContent = `${reviewInput.value.length} / 5000`;
});

document.querySelectorAll(".engine-button").forEach((button) => {
  button.addEventListener("click", () => {
    selectedEngine = button.dataset.engine;
    document.querySelectorAll(".engine-button").forEach((engineButton) => {
      engineButton.setAttribute(
        "aria-pressed",
        String(engineButton === button),
      );
    });
    formMessage.textContent = "";
  });
});

analysisForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const review = reviewInput.value.trim();
  formMessage.textContent = "";

  if (!review) {
    setMessage(formMessage, "Please enter a review before analyzing it.");
    reviewInput.focus();
    return;
  }

  submitButton.disabled = true;
  submitButton.querySelector("span").textContent = "Analyzing…";
  resultCard.hidden = true;

  try {
    const endpoint = selectedEngine === "gemini" ? "/analyze" : "/analyze/rules";
    const response = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ review }),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "The review could not be analyzed.");
    }

    showResult(data, selectedEngine);
    setMessage(formMessage, "Analysis saved to your review history.", false);
    await loadAnalyses();
  } catch (error) {
    setMessage(formMessage, error.message);
  } finally {
    submitButton.disabled = false;
    submitButton.querySelector("span").textContent = "Analyze review";
  }
});

refreshButton.addEventListener("click", loadAnalyses);

loadAnalyses();
