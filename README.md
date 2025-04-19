# CourtRoomNewASAP
# 🏛️ CourtRoom AI

**CourtRoom AI** is an autonomous legal reasoning simulation built using Large Language Models (LLMs). It replicates courtroom trials involving multiple agents (Lawyers, Judge, Plaintiff, Defendant), reflecting real-world legal proceedings including planning, evidence review, argumentation, and final verdicts.

---

## 📚 Overview

This project simulates courtroom trials based on real or mock legal cases. Using LLM-powered agents, it produces:
- Structured courtroom dialogue
- Evidence-backed arguments and rebuttals
- Reflective reasoning from the judge
- A final verdict in JSON format

---

## ⚙️ Architecture

### 👨‍⚖️ Agents
Each role is represented as an LLM-based agent:
- **Plaintiff (Eleanor Reed)** – Empathetic, story-driven
- **Prosecution (Jordan Blake)** – Formal, assertive
- **Defense (Alex Carter)** – Rational, doubt-raising
- **Defendant (Julian St. Clair)** – Humble, assertive
- **Judge (Evelyn Thompson)** – Reflective, neutral, verdict-rendering

Agents use techniques like:
- **ReAct** (Reasoning + Acting)
- **Reflection + Deliberation** for the judge

---

## 🧠 Features

- 💬 Natural language arguments, rebuttals, and closings
- 🔍 Access to case metadata and legal databases
- 🔄 Dynamic verdicts based on legal reasoning
- 🧾 Full courtroom history and dialogue tracking
- 🛠️ Modular structure for easy expansion

---

# 🔄 Trial Flow

1. 🧾 **Opening Statements**
2. ⚔️ **Argument Rounds** (configurable)
3. ❗ **Judge Interjection** (if inconsistencies arise)
4. ↩️ **Rebuttals**
5. 🔚 **Closing Statements**
6. ⚖️ **Reflection + Final Verdict**

---

## 🗃️ Sample `cases.csv` Format

```csv
id,text
case_001,"Plaintiff was struck by the defendant's car at 9PM..."
case_002,"Defendant was accused of embezzling $50,000 from the company..."

