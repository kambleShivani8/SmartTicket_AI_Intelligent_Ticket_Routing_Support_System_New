🎫 SmartTicket AI

AI-Powered Ticket Classification, Priority Prediction & Auto-Routing System

🚀 Overview

SmartTicket AI is an end-to-end intelligent helpdesk automation system that uses Machine Learning, NLP, and Sentence Transformers (SBERT) to analyze customer support tickets and streamline the entire workflow.

Instead of manually reading and routing tickets, this system automatically understands, categorizes, prioritizes, and assigns tickets, while also suggesting solutions based on past data.

✨ Key Capabilities

✔️ Automatically classifies support tickets
✔️ Predicts ticket priority using ML
✔️ Routes tickets to the correct team
✔️ Extracts important entities from text
✔️ Finds similar past tickets using semantic search
✔️ Suggests possible solutions
✔️ Decides whether to auto-assign or send for human review

⚙️ Features Breakdown
📌 1. Ticket Classification

Automatically categorizes tickets into:

Network & VPN Issues
Server & Infrastructure Issues
Billing & Payment Issues
Software & Application Issues
Hardware Issues
Security Issues
General Support
⚠️ 2. Priority Prediction (ML-Based)

Predicts urgency level:

🟢 Low
🟡 Medium
🟠 High
🔴 Critical

Tech Used:

TF-IDF Vectorization
Feature Engineering
Logistic Regression
🏢 3. Auto Team Routing

Automatically assigns tickets to the appropriate team:

Category	Assigned Team
Network Issues	Network Team
Server Issues	Infrastructure Team
Billing Issues	Finance Team
Software Issues	App Support Team
Security Issues	Security Team
🔍 4. Similar Ticket Retrieval (SBERT)

Uses Sentence Transformers + Cosine Similarity to:

Find semantically similar tickets
Return Top 3 related issues
Improve resolution speed
💡 5. Solution Recommendation

Suggests fixes based on historical tickets:

Restart server
Fix API timeout
Reset VPN
Process refund
🧠 6. Entity Extraction

Extracts key information from ticket text:

Module → Payment, Login
Platform → Android, Web
Issue Type → Crash, Error
🎯 7. Confidence-Based Decision System
✅ High Confidence → Auto Assignment
⚠️ Low Confidence → Sent for Human Review
🔄 System Workflow
Input Ticket
   ↓
Text Cleaning (SpaCy)
   ↓
Category Prediction (ML)
   ↓
Priority Prediction (ML)
   ↓
Team Routing
   ↓
SBERT Similarity Search
   ↓
Solution Suggestion
   ↓
Confidence Check
   ↓
Final Decision (Auto / Human)
🖥️ User Interface (Streamlit)
💬 Chat-style interface
⚡ Real-time AI analysis
📊 Insights panel showing:
Category
Priority
Assigned Team
Confidence Score
Similar Tickets
Suggested Solutions
## 🚀 Deployment

Deployed on Render:

https://smartticket-ai-intelligent-ticket-b526.onrender.com/

---

## 📂 Project Structure

```text
SmartTicket_AI/
│
├── app.py
├── requirements.txt
├── data/
├── models/
├── src/
└── README.md
|__ packages.txt
|__ render.yaml
